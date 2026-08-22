// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/tests/P2pIbTransportDeviceAbortTest.cuh"

#include <cstddef>

#include "comms/common/fault_tolerance/AbortDevice.cuh"
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

__device__ P2pIbrcTransportDevice makeLocalIbrcTransport(IbrcScratch& scratch) {
  return P2pIbrcTransportDevice(
      DeviceSpan<IbrcCmdQueueDevice>{scratch.queues, 2},
      /*nics=*/1,
      /*maxChannels=*/1,
      /*qpsPerConnection=*/1,
      DeviceSpan<IbLocalChannel>{scratch.channels, 1});
}

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

} // namespace

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
