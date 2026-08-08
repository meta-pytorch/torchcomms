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
// Resumable-forward (init_forward_progress / progress_forward_once) no-QP tests
//
// These exercise ONLY the no-NIC control paths: zero-byte init, init state
// layout, the Done/idle-pairing early-return, and the paired-slot desync trap.
// They never reach a signal wait or RDMA put (which need real QPs); those are
// covered by the distributed recv_forward_chain_test. A geometry-only
// IbChannelLayout + a zeroed IbLocalChannel array is enough because the trap
// and init paths touch only progress-slot state, not staging/signal buffers.
// Guarded NVIDIA-only because progress_forward_once carries a dependent
// static_assert(sizeof(CopyOp)==0) on AMD.
// =============================================================================
#ifndef __HIP_PLATFORM_AMD__
namespace {

// Build a QP-less transport over a caller-provided (zeroed) IbLocalChannel
// array with a geometry-only channel layout.
__device__ __forceinline__ P2pIbgdaTransportDevice makeNoQpForwardTransport(
    IbLocalChannel* channels,
    int maxChannels,
    int pipelineDepth,
    std::size_t perChannelBufferSize) {
  IbChannelLayout layout{};
  layout.maxChannels = maxChannels;
  layout.numLanes = 1;
  layout.pipelineDepth = pipelineDepth;
  layout.perChannelBufferSize = perChannelBufferSize;
  layout.perChannelSize = perChannelBufferSize;
  return P2pIbgdaTransportDevice(
      DeviceSpan<NicDeviceIbgdaResources>{}, // empty NIC span => no QPs
      IbgdaRemoteBuffer{},
      IbgdaLocalBuffer{},
      IbgdaLocalBuffer{},
      /*numSignalSlots=*/0,
      /*numCounterSlots=*/0,
      maxChannels,
      /*qpsPerConnection=*/1,
      /*qpDirectionCount=*/kIbDirections,
      DeviceSpan<IbLocalChannel>(channels, static_cast<uint32_t>(maxChannels)),
      layout);
}

} // namespace

// scenario: 0 = zero-byte init -> both slots Done, both nextStep preserved;
// 1 = non-zero init state layout; 2 = completed op (both slots Done, equal
// cursors) -> progress_forward_once returns Done.
__global__ void testForwardProgressNoQp(
    int scenario,
    IbLocalChannel* self_channels,
    IbLocalChannel* fwd_channels,
    int maxChannels,
    int pipelineDepth,
    unsigned long long perChannelBufferSize,
    unsigned long long nbytes,
    bool* success) {
  auto group = make_block_group();
  P2pIbgdaTransportDevice self = makeNoQpForwardTransport(
      self_channels,
      maxChannels,
      pipelineDepth,
      static_cast<std::size_t>(perChannelBufferSize));
  P2pIbgdaTransportDevice fwd = makeNoQpForwardTransport(
      fwd_channels,
      maxChannels,
      pipelineDepth,
      static_cast<std::size_t>(perChannelBufferSize));
  IbLocalChannel& selfCh = self.local_channel(0u);
  IbLocalChannel& fwdCh = fwd.local_channel(0u);
  const std::size_t proto = (static_cast<std::size_t>(nbytes) + 15ULL) & ~15ULL;

  bool ok = true;
  if (scenario == 0) {
    if (group.is_leader()) {
      selfCh.recvProgress.nextStep = 777;
      fwdCh.sendProgress.nextStep = 888;
    }
    group.sync();
    self.init_forward_progress(group, fwd, /*nbytes=*/0);
    if (group.is_leader()) {
      ok = selfCh.recvProgress.activeStage ==
              detail::IbSendRecvProgressStage::Done &&
          fwdCh.sendProgress.activeStage ==
              detail::IbSendRecvProgressStage::Done &&
          selfCh.recvProgress.nextStep == 777 &&
          fwdCh.sendProgress.nextStep == 888;
    }
  } else if (scenario == 1) {
    self.init_forward_progress(group, fwd, static_cast<std::size_t>(nbytes));
    if (group.is_leader()) {
      ok = selfCh.recvProgress.activeStage ==
              detail::IbSendRecvProgressStage::FwdWaitDataReady &&
          fwdCh.sendProgress.activeStage ==
              detail::IbSendRecvProgressStage::Busy &&
          selfCh.recvProgress.activeNextByte == 0 &&
          fwdCh.sendProgress.activeNextByte == 0 &&
          static_cast<std::size_t>(selfCh.recvProgress.nextStep) >= proto &&
          static_cast<std::size_t>(fwdCh.sendProgress.nextStep) >= proto;
    }
  } else if (scenario == 2) {
    if (group.is_leader()) {
      selfCh.recvProgress.activeStage = detail::IbSendRecvProgressStage::Done;
      selfCh.recvProgress.activeNextByte = proto;
      fwdCh.sendProgress.activeStage = detail::IbSendRecvProgressStage::Done;
      fwdCh.sendProgress.activeNextByte = proto;
    }
    group.sync();
    const IbgdaSendRecvProgressStatus st = self.progress_forward_once(
        group, nullptr, fwd, static_cast<std::size_t>(nbytes));
    if (group.is_leader()) {
      ok = st == IbgdaSendRecvProgressStatus::Done &&
          selfCh.recvProgress.activeNextByte ==
              fwdCh.sendProgress.activeNextByte;
    }
  }
  if (group.is_leader()) {
    *success = ok;
  }
}

// scenario: 0 = paired-slot cursor desync (init, then corrupt the fwd send
// slot's cursor) -> validate_forward_paired_slots traps; 1 = Done recv slot
// paired with a still-Busy fwd slot -> Done-pairing assert traps.
__global__ void testForwardProgressTrap(
    int scenario,
    IbLocalChannel* self_channels,
    IbLocalChannel* fwd_channels,
    int maxChannels,
    int pipelineDepth,
    unsigned long long perChannelBufferSize,
    unsigned long long nbytes) {
  auto group = make_block_group();
  P2pIbgdaTransportDevice self = makeNoQpForwardTransport(
      self_channels,
      maxChannels,
      pipelineDepth,
      static_cast<std::size_t>(perChannelBufferSize));
  P2pIbgdaTransportDevice fwd = makeNoQpForwardTransport(
      fwd_channels,
      maxChannels,
      pipelineDepth,
      static_cast<std::size_t>(perChannelBufferSize));
  IbLocalChannel& selfCh = self.local_channel(0u);
  IbLocalChannel& fwdCh = fwd.local_channel(0u);

  if (scenario == 0) {
    self.init_forward_progress(group, fwd, static_cast<std::size_t>(nbytes));
    if (group.is_leader()) {
      fwdCh.sendProgress.activeNextByte += 16; // desync the paired cursor
    }
    group.sync();
    self.progress_forward_once(
        group, nullptr, fwd, static_cast<std::size_t>(nbytes));
  } else if (scenario == 1) {
    if (group.is_leader()) {
      selfCh.recvProgress.activeStage = detail::IbSendRecvProgressStage::Done;
      fwdCh.sendProgress.activeStage = detail::IbSendRecvProgressStage::Busy;
    }
    group.sync();
    self.progress_forward_once(
        group, nullptr, fwd, static_cast<std::size_t>(nbytes));
  }
}

void runTestForwardProgressNoQp(
    int scenario,
    IbLocalChannel* self_channels,
    IbLocalChannel* fwd_channels,
    int maxChannels,
    int pipelineDepth,
    unsigned long long perChannelBufferSize,
    unsigned long long nbytes,
    bool* d_success) {
  testForwardProgressNoQp<<<1, 1>>>(
      scenario,
      self_channels,
      fwd_channels,
      maxChannels,
      pipelineDepth,
      perChannelBufferSize,
      nbytes,
      d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

cudaError_t runTestForwardProgressTrap(
    int scenario,
    IbLocalChannel* self_channels,
    IbLocalChannel* fwd_channels,
    int maxChannels,
    int pipelineDepth,
    unsigned long long perChannelBufferSize,
    unsigned long long nbytes) {
  // Intentionally unchecked - we expect the kernel to trap.
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  testForwardProgressTrap<<<1, 1>>>(
      scenario,
      self_channels,
      fwd_channels,
      maxChannels,
      pipelineDepth,
      perChannelBufferSize,
      nbytes);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  return cudaDeviceSynchronize();
}
#endif // !__HIP_PLATFORM_AMD__

} // namespace comms::prims::tests
