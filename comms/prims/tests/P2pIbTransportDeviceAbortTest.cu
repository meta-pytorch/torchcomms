// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/tests/P2pIbTransportDeviceAbortTest.cuh"

#include <cstddef>

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/common/fault_tolerance/AbortMacros.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/transport/P2pIbTransportDevice.cuh"
#include "comms/prims/transport/ibrc/P2pIbrcTransportDevice.cuh"

namespace comms::prims::test {

namespace {

// Queue and channel state backing the transport under test.
//
// Must be one block-wide object: `wait_signal` is a group primitive whose
// leader reads this state on behalf of every thread. Per-thread locals would
// give each thread its own copy, so the leader would be polling storage no
// other thread can see.
struct IbrcScratch {
  IbrcCmdQueueDevice queues[2];
  IbLocalChannel channels[1];
};

__device__ void zeroScratch(ThreadGroup& group, IbrcScratch& scratch) {
  auto* raw = reinterpret_cast<char*>(&scratch);
  for (std::size_t i = group.thread_id_in_group; i < sizeof(IbrcScratch);
       i += group.group_size) {
    raw[i] = 0;
  }
  group.sync();
}

__device__ P2pIbrcTransportDevice makeLocalIbrcTransport(
    IbrcScratch& scratch,
    comms::fault_tolerance::AbortDevice abort = {}) {
  return P2pIbrcTransportDevice(
      DeviceSpan<IbrcCmdQueueDevice>{scratch.queues, 2},
      /*nics=*/1,
      /*maxChannels=*/1,
      /*qpsPerConnection=*/1,
      DeviceSpan<IbLocalChannel>{scratch.channels, 1},
      /*ownedRemoteSignalBuf=*/{},
      /*ownedLocalSignalBuf=*/{},
      /*ownedCounterDeviceBuf=*/{},
      /*ownedCounterHostBuf=*/{},
      /*numSignalSlots=*/0,
      /*numCounterSlots=*/0,
      /*channelLayout=*/{},
      abort);
}

constexpr uint32_t kIbrcTestQueueDepth = 4;

// Ring storage for the queue-full test. Global rather than shared because the
// transport reaches pi/ci with system-scope atomics, which are meaningful only
// on memory the host could also map; a __shared__ ring would make the test
// depend on undefined behavior rather than on the code under test.
__device__ IbrcDesc gTestDescs[kIbrcTestQueueDepth];
__device__ uint64_t gTestPi;
__device__ uint64_t gTestCi;

// Which of the two production call paths reaches the same IBRC wait.
enum class IbEntryPoint { Wrapper, Ibrc };

template <IbEntryPoint kEntry>
__global__ void waitSignalKernel(
    uint64_t* signal,
    bool* waitResult,
    uint64_t expected,
    comms::fault_tolerance::AbortDevice abort,
    uint32_t* enteredWait) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);

  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(scratch);
  IbgdaLocalBuffer localSignal{signal, NetworkLKeys{}};

  abort.start();
  // Published last, after the handle is armed, so a host that sees it knows
  // every precondition of the wait is already in place.
  if (group.is_leader() && enteredWait != nullptr) {
    __threadfence_system();
    *static_cast<volatile uint32_t*>(enteredWait) = 1U;
  }
  group.sync();

  if constexpr (kEntry == IbEntryPoint::Wrapper) {
    P2pIbTransportDevice transport(&ibrc);
    transport.wait_signal(group, localSignal, expected, abort);
  } else {
    ibrc.wait_signal(group, localSignal, expected, abort);
  }
  // Reaching this line at all is the liveness guarantee under test: the wait
  // reports no status, so a wait that failed to terminate hangs the kernel
  // rather than returning something for the host to inspect.
  //
  // Whether the condition actually held on exit is read back from the signal
  // itself. That is what separates "the signal arrived" from "the abort
  // released us", without the wait having to hand back a status.
  if (group.is_leader()) {
    const uint64_t current = *static_cast<volatile uint64_t*>(signal);
    *waitResult = current >= expected;
  }
}

// Body of the queue-full producer, shared by the two kernels below. The
// transport is built with `abort` exactly as handed in -- unstarted -- because
// that is what a real IBRC transport holds: the communicator flag, never a
// deadline.
__device__ void runPutUntilQueueFull(
    ThreadGroup& group,
    IbrcScratch& scratch,
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    comms::fault_tolerance::AbortDevice abort) {
  if (group.is_leader()) {
    gTestPi = 0;
    gTestCi = 0;
    for (uint32_t i = 0; i < kIbrcTestQueueDepth; ++i) {
      gTestDescs[i].ready_seq = kIbrcInvalidReadySeq;
    }
    // Lane 0 of channel 0 is where a single-NIC, single-QP put lands.
    scratch.queues[0].descs = gTestDescs;
    scratch.queues[0].pi = &gTestPi;
    scratch.queues[0].ci = &gTestCi;
    scratch.queues[0].status = nullptr;
    scratch.queues[0].depth = kIbrcTestQueueDepth;
    scratch.queues[0].mask = kIbrcTestQueueDepth - 1;
  }
  group.sync();

  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(scratch, abort);
  IbgdaLocalBuffer localBuf{dataBuf, NetworkLKeys{/*n=*/1}};
  IbgdaRemoteBuffer remoteBuf{dataBuf, NetworkRKeys{/*n=*/1}};

  // Deliberately no start() here. The local copy is not the transport's, so
  // arming it would only look like a deadline while changing nothing -- which
  // is what made the earlier version of these tests pass on the explicit host
  // abort alone.
  uint32_t posted = 0;
  for (uint32_t i = 0; i < attempts; ++i) {
    // A skipped put returns the default ticket, whose value is 0; a posted one
    // is always seq + 1, so a nonzero value means the descriptor was published.
    const IbLocalCompletionTicket ticket = ibrc.put(
        group,
        localBuf,
        remoteBuf,
        sizeof(uint64_t),
        /*signalBuf=*/IbgdaRemoteBuffer{},
        /*signalVal=*/0);
    if (ticket.value != 0) {
      ++posted;
    }
  }
  if (group.is_leader()) {
    *postedOut = posted;
  }
}

__global__ void putUntilQueueFullKernel(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);
  runPutUntilQueueFull(group, scratch, dataBuf, postedOut, attempts, abort);
}

// The division of labour the FT contract actually specifies, in one kernel.
//
// Block 0 is an IBRC producer parked in `reserve()` on a full ring, holding a
// flag-only handle. Block 1 is the collective: it owns the deadline, arms the
// only started handle in the launch, and waits on a signal that never arrives.
//
// A block parked in a proxy-facing wait cannot latch its own timeout -- that is
// the property under test. Block 1's deadline expires, latches TIMED_OUT into
// the shared state, and block 0 leaves on the flag. Nothing calls setAbort().
//
// The two blocks race only in the benign direction: block 0 fills a 4-entry
// ring in microseconds against block 1's millisecond deadline, and if that
// order were ever inverted the test fails loudly on the posted count rather
// than hanging.
//
// Block 0 is the producer because the transport derives its channel id from
// the group id, and this fixture is wired for a single channel.
//
// The collective block spins on the signal directly rather than through a
// transport: all it has to model is "a wait in this kernel owns the armed
// handle", and a bare loop does that without needing a second channel's worth
// of queue geometry.
__global__ void queueFullReleasedByCollectiveDeadlineKernel(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    uint64_t* signal,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);

  if (blockIdx.x == 0) {
    runPutUntilQueueFull(group, scratch, dataBuf, postedOut, attempts, abort);
    return;
  }

  abort.start();
  if (group.is_leader()) {
    const auto* observed = static_cast<volatile uint64_t*>(signal);
    while (*observed < 1U) {
      FT_ABORT_BREAK(abort, "test collective wait on a signal that never sets");
    }
  }
  group.sync();
}

// Ben's case: a kernel that *ends* in flush() rather than parking in reserve().
//
// The proxy never advances `ci`, so the drain cannot complete. Before the
// watchdog was made unconditional, FT-on removed the only bound here -- the
// legacy cycle deadline was gated on `!abort.isEnabled()` and the caller's
// deadline was dropped on the IBRC branch of P2pIbTransportDevice::flush --
// leaving an explicit host abort as the sole exit. Nothing calls setAbort().
__global__ void flushNeverDrainsKernel(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);

  // One successful put, so the drain has something to wait on.
  runPutUntilQueueFull(
      group, scratch, dataBuf, postedOut, /*attempts=*/1, abort);

  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(scratch, abort);
  // `ci` is never advanced by anyone, so this can only end on the fixed proxy
  // watchdog -- flush takes no deadline, by design.
  ibrc.flush(group, IbDirection::Send);
}

} // namespace

uint32_t ibrcTestQueueDepth() {
  return kIbrcTestQueueDepth;
}

void launchIbrcPutUntilQueueFull(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    comms::fault_tolerance::AbortDevice abort) {
  putUntilQueueFullKernel<<<1, 32>>>(dataBuf, postedOut, attempts, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcFlushNeverDrains(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    comms::fault_tolerance::AbortDevice abort) {
  flushNeverDrainsKernel<<<1, 32>>>(dataBuf, postedOut, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcQueueFullReleasedByCollectiveDeadline(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    uint64_t* signal,
    comms::fault_tolerance::AbortDevice abort) {
  queueFullReleasedByCollectiveDeadlineKernel<<<2, 32>>>(
      dataBuf, postedOut, attempts, signal, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbWrapperWaitSignal(
    uint64_t* signal,
    bool* waitResult,
    uint64_t expected,
    comms::fault_tolerance::AbortDevice abort,
    uint32_t* enteredWait) {
  waitSignalKernel<IbEntryPoint::Wrapper>
      <<<1, 32>>>(signal, waitResult, expected, abort, enteredWait);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcWaitSignal(
    uint64_t* signal,
    bool* waitResult,
    uint64_t expected,
    comms::fault_tolerance::AbortDevice abort,
    uint32_t* enteredWait) {
  waitSignalKernel<IbEntryPoint::Ibrc>
      <<<1, 32>>>(signal, waitResult, expected, abort, enteredWait);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
