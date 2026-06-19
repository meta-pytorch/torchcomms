// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// CudaHipCompat must come before Checks.h so the `cuda*` -> `hip*`
// macro renames apply on AMD builds (Checks.h uses `cudaError_t` /
// `cudaSuccess` / `cudaGetErrorString` / `cudaGetLastError` directly).
#include "comms/prims/transport/amd/HipHostCompat.h"

#include "comms/prims/core/TimeoutUtils.h"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/P2pIbgdaTransportDeviceTest.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::tests {

// =============================================================================
// Device-side test kernels
// =============================================================================

__global__ void testP2pTransportConstruction(bool* success) {
  // Create transport on device with empty NIC span
  P2pIbgdaTransportDevice transport(DeviceSpan<NicDeviceIbgdaResources>{});

  // If we get here, construction succeeded
  *success = true;
}

__global__ void testP2pTransportDefaultConstruction(bool* success) {
  // Default construction should initialize all members
  P2pIbgdaTransportDevice transport;

  // If we get here, default construction succeeded
  *success = true;
}

__global__ void testP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* success) {
  // Construct transport with ownedLocalSignalBuf pointing to d_signalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      numSignals);

  *success = true;

  // Test read_signal for each slot via slot-index API
  for (int i = 0; i < numSignals; ++i) {
    uint64_t expected = static_cast<uint64_t>(i + 1) * 100;
    uint64_t actual = transport.read_signal(i);
    if (actual != expected) {
      *success = false;
    }
  }
}

// =============================================================================
// wait_signal test kernels
// =============================================================================

__global__ void
testWaitSignalGE(uint64_t* d_signalBuf, uint64_t targetValue, bool* success) {
  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Signal buffer is pre-set to a value >= targetValue by host
  // wait_signal should return immediately (slot 0)
  transport.wait_signal(0, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* success) {
  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      numSignals);

  *success = true;

  // Signal buffer is pre-set: slot[i] = (i + 1) * 100
  // Test wait_signal on each slot with matching GE condition
  for (int i = 0; i < numSignals; ++i) {
    uint64_t expectedValue = static_cast<uint64_t>(i + 1) * 100;
    transport.wait_signal(i, expectedValue);

    // Verify read_signal returns the same value
    uint64_t readValue = transport.read_signal(i);
    if (readValue != expectedValue) {
      *success = false;
    }
  }
}

// =============================================================================
// Wrapper functions to launch the kernels (called from .cc test file)
// =============================================================================

void runTestP2pTransportConstruction(bool* d_success) {
  testP2pTransportConstruction<<<1, 1>>>(d_success);
}

void runTestP2pTransportDefaultConstruction(bool* d_success) {
  testP2pTransportDefaultConstruction<<<1, 1>>>(d_success);
}

void runTestP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* d_success) {
  testP2pTransportReadSignal<<<1, 1>>>(d_signalBuf, numSignals, d_success);
}

void runTestWaitSignalGE(
    uint64_t* d_signalBuf,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalGE<<<1, 1>>>(d_signalBuf, targetValue, d_success);
}

void runTestWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* d_success) {
  testWaitSignalMultipleSlots<<<1, 1>>>(d_signalBuf, numSignals, d_success);
}

// =============================================================================
// Group-level API test kernels
// =============================================================================

__global__ void testPutCooperativePartitioning(bool* success) {
  *success = true;

  auto group = comms::prims::make_warp_group();
  if (group.group_size != comms::prims::kWarpSize) {
    *success = false;
    return;
  }

  constexpr std::size_t kTotalBytes = 1024; // 1KB
  constexpr std::size_t kChunkSize = kTotalBytes / comms::prims::kWarpSize;

  std::size_t expectedOffset = group.thread_id_in_group * kChunkSize;
  std::size_t expectedChunk = kChunkSize;

  char baseData[8];
  void* basePtr = baseData;

  comms::prims::IbgdaLocalBuffer baseBuf(
      basePtr, comms::prims::NetworkLKeys{comms::prims::NetworkLKey(0x1111)});
  comms::prims::IbgdaLocalBuffer laneBuf = baseBuf.subBuffer(expectedOffset);

  auto* expectedPtr = static_cast<char*>(basePtr) + expectedOffset;
  if (laneBuf.ptr != expectedPtr) {
    *success = false;
  }

  if (laneBuf.lkey_per_device[0] != baseBuf.lkey_per_device[0]) {
    *success = false;
  }

  if (expectedChunk != kChunkSize) {
    *success = false;
  }
}

__global__ void testPutSignalGroupBroadcast(bool* success) {
  *success = true;

  auto group = comms::prims::make_warp_group();
  if (group.group_size != comms::prims::kWarpSize) {
    *success = false;
    return;
  }

  uint64_t signalTicket = 0;
  if (group.is_leader()) {
    signalTicket = 0xCAFEBABE12345678ULL;
  }

  signalTicket = group.broadcast<uint64_t>(signalTicket);

  if (signalTicket != 0xCAFEBABE12345678ULL) {
    *success = false;
  }
}

// =============================================================================
// Group-level test wrapper functions
// =============================================================================

void runTestPutCooperativePartitioning(bool* d_success) {
  testPutCooperativePartitioning<<<1, comms::prims::kWarpSize>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void runTestPutSignalGroupBroadcast(bool* d_success) {
  testPutSignalGroupBroadcast<<<1, comms::prims::kWarpSize>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// broadcast test kernels for BLOCK and MULTIWARP scopes
// =============================================================================

__global__ void testBroadcast64Block(bool* success) {
  auto group = comms::prims::make_block_group();

  uint64_t val = 0;
  if (group.is_leader()) {
    val = 0xDEADBEEF42424242ULL;
  }

  val = group.broadcast<uint64_t>(val);

  if (val != 0xDEADBEEF42424242ULL) {
    *success = false;
  }
}

__global__ void testBroadcast64Multiwarp(bool* success) {
  auto group = comms::prims::make_multiwarp_group();

  uint64_t val = 0;
  if (group.is_leader()) {
    val = 0xAAAABBBB00000000ULL + group.group_id;
  }

  val = group.broadcast<uint64_t>(val);

  uint64_t expected = 0xAAAABBBB00000000ULL + group.group_id;
  if (val != expected) {
    *success = false;
  }
}

__global__ void testBroadcast64DoubleSafety(bool* success) {
  auto group = comms::prims::make_block_group();

  uint64_t val1 = 0;
  if (group.is_leader()) {
    val1 = 0x1111111111111111ULL;
  }
  val1 = group.broadcast<uint64_t>(val1);

  if (val1 != 0x1111111111111111ULL) {
    *success = false;
  }

  uint64_t val2 = 0;
  if (group.is_leader()) {
    val2 = 0x2222222222222222ULL;
  }
  val2 = group.broadcast<uint64_t>(val2);

  if (val2 != 0x2222222222222222ULL) {
    *success = false;
  }
}

__global__ void testPutCooperativePartitioningBlock(bool* success) {
  auto group = comms::prims::make_block_group();

  constexpr std::size_t kTotalBytes = 4096; // 4KB
  std::size_t chunkSize = kTotalBytes / group.group_size;
  std::size_t expectedOffset = group.thread_id_in_group * chunkSize;

  char baseData[8];
  void* basePtr = baseData;

  comms::prims::IbgdaLocalBuffer baseBuf(
      basePtr, comms::prims::NetworkLKeys{comms::prims::NetworkLKey(0x1111)});
  comms::prims::IbgdaLocalBuffer laneBuf = baseBuf.subBuffer(expectedOffset);

  auto* expectedPtr = static_cast<char*>(basePtr) + expectedOffset;
  if (laneBuf.ptr != expectedPtr) {
    *success = false;
  }

  if (laneBuf.lkey_per_device[0] != baseBuf.lkey_per_device[0]) {
    *success = false;
  }
}

// =============================================================================
// broadcast / block-scope test wrapper functions
// =============================================================================

void runTestBroadcast64Block(bool* d_success) {
  testBroadcast64Block<<<4, 256>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void runTestBroadcast64Multiwarp(bool* d_success) {
  testBroadcast64Multiwarp<<<2, 512>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void runTestBroadcast64DoubleSafety(bool* d_success) {
  testBroadcast64DoubleSafety<<<4, 256>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void runTestPutCooperativePartitioningBlock(bool* d_success) {
  testPutCooperativePartitioningBlock<<<4, 256>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// trace_ibgda_event test kernel
// =============================================================================

__global__ void testTraceIbgdaEvent(PipesTraceHandle trace) {
#if PIPES_IS_DEVICE_COMPILE
  trace_ibgda_event(
      trace,
      /*self_rank=*/7,
      PipesTraceEventType::kIbSendBegin,
      /*step=*/0x12345678,
      /*group_id=*/0x4321);
#endif
}

void runTestTraceIbgdaEvent(PipesTraceHandle trace) {
  testTraceIbgdaEvent<<<1, 1>>>(trace);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// wait_signal timeout test kernels
// =============================================================================

__global__ void testWaitSignalTimeout(uint64_t* d_signalBuf, Timeout timeout) {
  // Start the timeout timer
  timeout.start();

  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Signal buffer is pre-set to 0 by host.
  // Waiting for >= 999 will never succeed, so timeout should fire.
  transport.wait_signal(0, 999, timeout);
}

__global__ void
testWaitSignalNoTimeout(uint64_t* d_signalBuf, Timeout timeout, bool* success) {
  // Start the timeout timer
  timeout.start();

  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Signal buffer is pre-set to 42 by host.
  // Waiting for >= 42 will succeed immediately, no timeout.
  transport.wait_signal(0, 42, timeout);

  *success = true;
}

// =============================================================================
// wait_signal timeout test wrapper functions
// =============================================================================

cudaError_t runTestWaitSignalTimeout(
    uint64_t* d_signalBuf,
    int device,
    uint32_t timeout_ms) {
  Timeout timeout = makeTimeout(timeout_ms, device);

  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  testWaitSignalTimeout<<<1, 1>>>(d_signalBuf, timeout);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  return cudaDeviceSynchronize();
}

void runTestWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    int device,
    uint32_t timeout_ms,
    bool* d_success) {
  Timeout timeout = makeTimeout(timeout_ms, device);

  testWaitSignalNoTimeout<<<1, 1>>>(d_signalBuf, timeout, d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Resumable-forward (init_forward_progress / progress_forward_once) unit tests
//
// These exercise ONLY the no-NIC paths: zero-byte init, init state layout, and
// the first DATA_READY yield (which returns before any signal/put). The
// NIC-touching paths (CopyOp::forward, prev SLOT_FREE signal, fwd put, and the
// NIC_DONE/SLOT_FREE backpressure waits — which only arise once the pipeline
// fills) require real QPs and are covered by the distributed
// recv_forward_chain_test. Guarded NVIDIA-only because progress_forward_once
// carries a dependent `static_assert(sizeof(CopyOp)==0)` on AMD.
// =============================================================================
#ifndef __HIP_PLATFORM_AMD__
namespace {

// Build a transport handle over caller-provided device buffers, with a
// populated IbSendRecvState but an empty NIC span (no QPs).
__device__ __forceinline__ P2pIbgdaTransportDevice makeUnitForwardTransport(
    IbSendRecvState::ProgressSlot* state,
    uint64_t* signalBuf,
    uint64_t* counterBuf,
    char* sendStaging,
    char* recvStaging,
    int maxGroups,
    int pipelineDepth,
    std::size_t dataBufferSize) {
  // Aggregate (designated) init: DeviceSpan has a const data member, so its
  // copy-assignment is deleted — `s.state = ...` won't compile; set every field
  // at construction instead. Designators are in declaration order.
  IbSendRecvState s = {
      .sendStagingPtr = sendStaging,
      .recvStagingPtr = recvStaging,
      .localSignalBuf = IbgdaLocalBuffer(signalBuf, NetworkLKeys{}),
      .localCounterBuf = IbgdaLocalBuffer(counterBuf, NetworkLKeys{}),
      .state = DeviceSpan<IbSendRecvState::ProgressSlot>(
          state, static_cast<uint32_t>(2 * maxGroups)),
      .maxGroups = maxGroups,
      .pipelineDepth = pipelineDepth,
      .dataBufferSize = dataBufferSize,
  };
  return P2pIbgdaTransportDevice(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      IbgdaLocalBuffer{},
      IbgdaLocalBuffer{},
      /*numSignalSlots=*/0,
      /*numCounterSlots=*/0,
      s);
}

} // namespace

// scenario: 0 = zero-byte init -> Done; 1 = init state layout; 2 = DATA_READY
// yield (recvSignal[groupId] preset just below the chunk's streamEnd by host).
__global__ void testForwardProgressNoQp(
    IbSendRecvState::ProgressSlot* recvState,
    uint64_t* recvSignal,
    uint64_t* recvCounter,
    char* recvStaging,
    IbSendRecvState::ProgressSlot* fwdState,
    uint64_t* fwdSignal,
    uint64_t* fwdCounter,
    char* fwdStaging,
    int maxGroups,
    int pipelineDepth,
    std::size_t dataBufferSize,
    int scenario,
    std::size_t nbytes,
    bool* success) {
  auto group = make_block_group();
  P2pIbgdaTransportDevice self = makeUnitForwardTransport(
      recvState,
      recvSignal,
      recvCounter,
      /*sendStaging=*/recvStaging,
      /*recvStaging=*/recvStaging,
      maxGroups,
      pipelineDepth,
      dataBufferSize);
  P2pIbgdaTransportDevice fwd = makeUnitForwardTransport(
      fwdState,
      fwdSignal,
      fwdCounter,
      /*sendStaging=*/fwdStaging,
      /*recvStaging=*/fwdStaging,
      maxGroups,
      pipelineDepth,
      dataBufferSize);

  // Self's recv slot index and fwd's send slot index for group_id 0.
  const int recvSlotIdx = maxGroups; // recv base + group_id(0)
  const int sendSlotIdx = 0; // send base + group_id(0)

  if (scenario == 0) {
    self.init_forward_progress(group, fwd, /*nbytes=*/0, maxGroups);
    group.sync();
    IbgdaSendRecvProgressStatus st = self.progress_forward_once(
        group, nullptr, fwd, /*nbytes=*/0, maxGroups);
    if (group.is_leader()) {
      if (recvState[recvSlotIdx].activeStage !=
          detail::IbSendRecvProgressStage::Done) {
        *success = false;
      }
      if (fwdState[sendSlotIdx].activeStage !=
          detail::IbSendRecvProgressStage::Done) {
        *success = false;
      }
      if (st != IbgdaSendRecvProgressStatus::Done) {
        *success = false;
      }
    }
  } else if (scenario == 1) {
    self.init_forward_progress(group, fwd, nbytes, maxGroups);
    group.sync();
    if (group.is_leader()) {
      const auto& rs = recvState[recvSlotIdx];
      const auto& fs = fwdState[sendSlotIdx];
      if (rs.activeStage != detail::IbSendRecvProgressStage::FwdWaitDataReady) {
        *success = false;
      }
      if (rs.activeNextByte != 0 || rs.activeBaseStep != 0) {
        *success = false;
      }
      if (rs.nextStep != static_cast<int64_t>(nbytes)) {
        *success = false; // recv cursor reserved at init
      }
      if (fs.activeStage != detail::IbSendRecvProgressStage::Busy) {
        *success = false;
      }
      if (fs.activeNextByte != 0 || fs.activeBaseStep != 0) {
        *success = false;
      }
      if (fs.nextStep != static_cast<int64_t>(nbytes)) {
        *success = false; // send cursor reserved at init (lockstep base)
      }
    }
  } else if (scenario == 2) {
    // scenario 2: DATA_READY not yet at the chunk's streamEnd -> Waiting, no
    // side effect, stage unchanged.
    self.init_forward_progress(group, fwd, nbytes, maxGroups);
    group.sync();
    IbgdaSendRecvProgressStatus st =
        self.progress_forward_once(group, nullptr, fwd, nbytes, maxGroups);
    if (group.is_leader()) {
      if (st != IbgdaSendRecvProgressStatus::Waiting) {
        *success = false;
      }
      if (recvState[recvSlotIdx].activeStage !=
          detail::IbSendRecvProgressStage::FwdWaitDataReady) {
        *success = false;
      }
      if (recvState[recvSlotIdx].activeNextByte != 0) {
        *success = false;
      }
    }
  } else {
    // scenario 3: DATA_READY satisfied (recvSignal preset >= streamEnd) but fwd
    // NIC_DONE not ready -> the call advances FwdWaitDataReady ->
    // FwdWaitNicDone (a side-effect-free transition) and returns Progressed
    // with the new stage persisted. To make the NIC_DONE wait actually run on
    // the first chunk (otherwise skipped while the pipeline is not full),
    // pre-advance the fwd send slot's reserved base so fwdStreamEnd >
    // fwdPipelineBytes. No NIC: the counter/signal reads are plain memory; the
    // call returns before any put.
    const std::size_t perBlockSlot =
        (dataBufferSize / maxGroups) & ~static_cast<std::size_t>(15);
    const std::size_t fwdPipelineBytes =
        perBlockSlot * static_cast<std::size_t>(pipelineDepth);
    if (group.is_leader()) {
      fwdState[sendSlotIdx].nextStep =
          static_cast<int64_t>(8 * fwdPipelineBytes);
    }
    group.sync();
    self.init_forward_progress(group, fwd, nbytes, maxGroups);
    group.sync();
    IbgdaSendRecvProgressStatus st =
        self.progress_forward_once(group, nullptr, fwd, nbytes, maxGroups);
    if (group.is_leader()) {
      if (st != IbgdaSendRecvProgressStatus::Progressed) {
        *success = false;
      }
      if (recvState[recvSlotIdx].activeStage !=
          detail::IbSendRecvProgressStage::FwdWaitNicDone) {
        *success = false;
      }
      if (recvState[recvSlotIdx].activeNextByte != 0) {
        *success = false;
      }
    }
  }
}

void runTestForwardProgressNoQp(
    IbSendRecvState::ProgressSlot* recvState,
    uint64_t* recvSignal,
    uint64_t* recvCounter,
    char* recvStaging,
    IbSendRecvState::ProgressSlot* fwdState,
    uint64_t* fwdSignal,
    uint64_t* fwdCounter,
    char* fwdStaging,
    int maxGroups,
    int pipelineDepth,
    std::size_t dataBufferSize,
    int scenario,
    std::size_t nbytes,
    bool* d_success) {
  testForwardProgressNoQp<<<1, 128>>>(
      recvState,
      recvSignal,
      recvCounter,
      recvStaging,
      fwdState,
      fwdSignal,
      fwdCounter,
      fwdStaging,
      maxGroups,
      pipelineDepth,
      dataBufferSize,
      scenario,
      nbytes,
      d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}
#endif // !__HIP_PLATFORM_AMD__

} // namespace comms::prims::tests
