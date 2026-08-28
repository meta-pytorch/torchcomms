// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// CudaHipCompat must come before Checks.h so the `cuda*` -> `hip*`
// macro renames apply on AMD builds (Checks.h uses `cudaError_t` /
// `cudaSuccess` / `cudaGetErrorString` / `cudaGetLastError` directly).
#include "comms/prims/transport/amd/HipHostCompat.h"

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/P2pIbgdaTransportDeviceTest.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

#include <chrono>

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

__global__ void testWaitSignalWithDisabledAbort(
    uint64_t* d_signalBuf,
    bool* success) {
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  comms::fault_tolerance::AbortDevice abort;
  transport.wait_signal(0, 0, abort);
  *success = true;
}

__global__ void testWaitSignalUntilAbort(
    uint64_t* d_signalBuf,
    comms::fault_tolerance::AbortDevice abort,
    bool* success,
    uint32_t* enteredWait) {
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Arm the deadline the way a production kernel does. Without this the
  // handle observes explicit aborts only, and a communicator abortDevice never
  // reaches the wait.
  abort.start();
  // Published after the handle is armed, so a host that sees it knows every
  // precondition of the wait is already in place.
  if (enteredWait != nullptr) {
    __threadfence_system();
    *static_cast<volatile uint32_t*>(enteredWait) = 1U;
  }
  transport.wait_signal(0, 1, abort);
  // Reaching this line is the liveness guarantee: the wait reports no status,
  // so one that failed to terminate hangs the kernel instead. The signal slot
  // is what distinguishes the two ways out -- it is still short of the expected
  // value here, so the abort released the wait rather than the signal landing.
  *success = *static_cast<volatile uint64_t*>(localSigBuf.ptr) < 1;
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

void runTestWaitSignalWithDisabledAbort(
    uint64_t* d_signalBuf,
    bool* d_success) {
  testWaitSignalWithDisabledAbort<<<1, 1>>>(d_signalBuf, d_success);
}

void runTestWaitSignalUntilAbort(
    uint64_t* d_signalBuf,
    comms::fault_tolerance::AbortDevice abort,
    bool* d_success,
    uint32_t* d_enteredWait) {
  testWaitSignalUntilAbort<<<1, 1>>>(
      d_signalBuf, abort, d_success, d_enteredWait);
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
// wait_signal abortDevice test kernels
// =============================================================================

__global__ void testWaitSignalTimeout(
    uint64_t* d_signalBuf,
    AbortDevice abortDevice) {
  // Start the abortDevice timer
  abortDevice.start();

  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Signal buffer is pre-set to 0 by host.
  // Waiting for >= 999 will never succeed, so abortDevice should fire.
  transport.wait_signal(0, 999, abortDevice);
}

__global__ void testWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    AbortDevice abortDevice,
    bool* success) {
  // Start the abortDevice timer
  abortDevice.start();

  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Signal buffer is pre-set to 42 by host.
  // Waiting for >= 42 will succeed immediately, no abort handle.
  transport.wait_signal(0, 42, abortDevice);

  *success = true;
}

// =============================================================================
// wait_signal abortDevice test wrapper functions
// =============================================================================

cudaError_t runTestWaitSignalTimeout(
    uint64_t* d_signalBuf,
    int device,
    uint32_t timeout_ms) {
  auto status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  comms::fault_tolerance::Abort abort{
      /*enabled=*/true, comms::fault_tolerance::AbortBehavior::TRAP};
  abort.setDefaultTimeout(std::chrono::milliseconds{timeout_ms});
  AbortDevice abortDevice = abort.getDeviceHandle();

  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  testWaitSignalTimeout<<<1, 1>>>(d_signalBuf, abortDevice);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  return cudaDeviceSynchronize();
}

void runTestWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    int /*device*/,
    uint32_t /*timeout_ms*/,
    bool* d_success) {
  AbortDevice abortDevice;

  testWaitSignalNoTimeout<<<1, 1>>>(d_signalBuf, abortDevice, d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::tests
