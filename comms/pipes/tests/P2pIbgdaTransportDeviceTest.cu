// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pIbgdaTransportDeviceTest.cuh"

namespace comms::pipes::tests {

// =============================================================================
// Device-side test kernels
// =============================================================================

__global__ void testP2pTransportConstruction(bool* success) {
  // Create transport on device with null QPs
  P2pIbgdaTransportDevice transport({}, {});

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
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKey(0));
  P2pIbgdaTransportDevice transport(
      {},
      {},
      NetworkLKey{},
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
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKey(0));
  P2pIbgdaTransportDevice transport(
      {},
      {},
      NetworkLKey{},
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
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKey(0));
  P2pIbgdaTransportDevice transport(
      {},
      {},
      NetworkLKey{},
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

__global__ void testPutGroupPartitioning(bool* success) {
  *success = true;

  auto group = comms::pipes::make_warp_group();
  if (group.group_size != comms::pipes::kWarpSize) {
    *success = false;
    return;
  }

  constexpr std::size_t kTotalBytes = 1024; // 1KB
  constexpr std::size_t kChunkSize = kTotalBytes / comms::pipes::kWarpSize;

  std::size_t expectedOffset = group.thread_id_in_group * kChunkSize;
  std::size_t expectedChunk = kChunkSize;

  char baseData[8];
  void* basePtr = baseData;

  comms::pipes::IbgdaLocalBuffer baseBuf(
      basePtr, comms::pipes::NetworkLKey(0x1111));
  comms::pipes::IbgdaLocalBuffer laneBuf = baseBuf.subBuffer(expectedOffset);

  auto* expectedPtr = static_cast<char*>(basePtr) + expectedOffset;
  if (laneBuf.ptr != expectedPtr) {
    *success = false;
  }

  if (laneBuf.lkey != baseBuf.lkey) {
    *success = false;
  }

  if (expectedChunk != kChunkSize) {
    *success = false;
  }
}

__global__ void testPutSignalGroupBroadcast(bool* success) {
  *success = true;

  auto group = comms::pipes::make_warp_group();
  if (group.group_size != comms::pipes::kWarpSize) {
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

void runTestPutGroupPartitioning(bool* d_success) {
  testPutGroupPartitioning<<<1, comms::pipes::kWarpSize>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void runTestPutSignalGroupBroadcast(bool* d_success) {
  testPutSignalGroupBroadcast<<<1, comms::pipes::kWarpSize>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// broadcast test kernels for BLOCK and MULTIWARP scopes
// =============================================================================

__global__ void testBroadcast64Block(bool* success) {
  auto group = comms::pipes::make_block_group();

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
  auto group = comms::pipes::make_multiwarp_group();

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
  auto group = comms::pipes::make_block_group();

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

__global__ void testPutGroupPartitioningBlock(bool* success) {
  auto group = comms::pipes::make_block_group();

  constexpr std::size_t kTotalBytes = 4096; // 4KB
  std::size_t chunkSize = kTotalBytes / group.group_size;
  std::size_t expectedOffset = group.thread_id_in_group * chunkSize;

  char baseData[8];
  void* basePtr = baseData;

  comms::pipes::IbgdaLocalBuffer baseBuf(
      basePtr, comms::pipes::NetworkLKey(0x1111));
  comms::pipes::IbgdaLocalBuffer laneBuf = baseBuf.subBuffer(expectedOffset);

  auto* expectedPtr = static_cast<char*>(basePtr) + expectedOffset;
  if (laneBuf.ptr != expectedPtr) {
    *success = false;
  }

  if (laneBuf.lkey != baseBuf.lkey) {
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

void runTestPutGroupPartitioningBlock(bool* d_success) {
  testPutGroupPartitioningBlock<<<4, 256>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// wait_signal timeout test kernels
// =============================================================================

__global__ void testWaitSignalTimeout(uint64_t* d_signalBuf, Timeout timeout) {
  // Start the timeout timer
  timeout.start();

  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKey(0));
  P2pIbgdaTransportDevice transport(
      {},
      {},
      NetworkLKey{},
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
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKey(0));
  P2pIbgdaTransportDevice transport(
      {},
      {},
      NetworkLKey{},
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

void runTestWaitSignalTimeout(
    uint64_t* d_signalBuf,
    int device,
    uint32_t timeout_ms) {
  Timeout timeout = makeTimeout(timeout_ms, device);

  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  testWaitSignalTimeout<<<1, 1>>>(d_signalBuf, timeout);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaDeviceSynchronize();
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

} // namespace comms::pipes::tests
