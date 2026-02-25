// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pIbgdaTransportDeviceTest.cuh"

namespace comms::pipes::tests {

// =============================================================================
// Device-side test kernels
// =============================================================================

__global__ void testP2pTransportConstruction(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    bool* success) {
  // Create transport on device with the given buffers (no real QP)
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf);

  *success = true;

  // Verify buffer accessors work
  auto localSig = transport.getLocalSignalBuffer();
  auto remoteSig = transport.getRemoteSignalBuffer();

  if (localSig.ptr != localBuf.ptr || localSig.lkey != localBuf.lkey) {
    *success = false;
  }
  if (remoteSig.ptr != remoteBuf.ptr || remoteSig.rkey != remoteBuf.rkey) {
    *success = false;
  }

  // QP should be null in this test (no real DOCA setup)
  if (transport.getQp() != nullptr) {
    *success = false;
  }
}

__global__ void testP2pTransportDefaultConstruction(bool* success) {
  // Default construction should initialize all members
  P2pIbgdaTransportDevice transport;

  *success = true;

  // QP should be null
  if (transport.getQp() != nullptr) {
    *success = false;
  }

  // Local signal buffer should have null ptr
  auto localSig = transport.getLocalSignalBuffer();
  if (localSig.ptr != nullptr) {
    *success = false;
  }

  // Remote signal buffer should have null ptr
  auto remoteSig = transport.getRemoteSignalBuffer();
  if (remoteSig.ptr != nullptr) {
    *success = false;
  }

  // Default numSignals should be 1
  if (transport.getNumSignals() != 1) {
    *success = false;
  }
}

__global__ void testP2pTransportNumSignals(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, numSignals);

  *success = (transport.getNumSignals() == numSignals);
}

__global__ void testP2pTransportSignalPointerArithmetic(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, numSignals);

  *success = true;

  // The transport stores base pointers and calculates signal[i] as base + i
  // We can verify this by checking the buffer accessors still point to base
  auto localSig = transport.getLocalSignalBuffer();
  auto remoteSig = transport.getRemoteSignalBuffer();

  // Base pointers should match what we passed in
  if (localSig.ptr != localBuf.ptr) {
    *success = false;
  }
  if (remoteSig.ptr != remoteBuf.ptr) {
    *success = false;
  }

  // Keys should be preserved
  if (localSig.lkey != localBuf.lkey) {
    *success = false;
  }
  if (remoteSig.rkey != remoteBuf.rkey) {
    *success = false;
  }
}

__global__ void testP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* success) {
  // The localBuf should point to d_signalBuf which is pre-initialized with
  // known values: d_signalBuf[i] = (i + 1) * 100
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, numSignals);

  *success = true;

  // Test read_signal for each slot
  for (int i = 0; i < numSignals; ++i) {
    uint64_t expected = static_cast<uint64_t>(i + 1) * 100;
    uint64_t actual = transport.read_signal(i);
    if (actual != expected) {
      *success = false;
    }
  }
}

__global__ void testIbgdaWork(bool* success) {
  *success = true;

  // Test default construction
  IbgdaWork defaultWork;
  if (defaultWork.value != 0) {
    *success = false;
  }

  // Test explicit construction with a value
  doca_gpu_dev_verbs_ticket_t testTicket = 12345;
  IbgdaWork workWithValue(testTicket);
  if (workWithValue.value != testTicket) {
    *success = false;
  }

  // Test copy
  IbgdaWork copiedWork = workWithValue;
  if (copiedWork.value != testTicket) {
    *success = false;
  }
}

// =============================================================================
// wait_signal test kernels
// =============================================================================

__global__ void testWaitSignalEQ(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to targetValue by host
  // wait_signal with EQ should return immediately
  transport.wait_signal(0, IbgdaCmpOp::EQ, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalNE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which != targetValue)
  // wait_signal with NE should return immediately
  transport.wait_signal(0, IbgdaCmpOp::NE, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalGE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which >= targetValue)
  // wait_signal with GE should return immediately
  transport.wait_signal(0, IbgdaCmpOp::GE, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalGT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which > targetValue)
  // wait_signal with GT should return immediately
  transport.wait_signal(0, IbgdaCmpOp::GT, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalLE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which <= targetValue)
  // wait_signal with LE should return immediately
  transport.wait_signal(0, IbgdaCmpOp::LE, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalLT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which < targetValue)
  // wait_signal with LT should return immediately
  transport.wait_signal(0, IbgdaCmpOp::LT, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, numSignals);

  *success = true;

  // Signal buffer is pre-set: slot[i] = (i + 1) * 100
  // Test wait_signal on each slot with matching EQ condition
  for (int i = 0; i < numSignals; ++i) {
    uint64_t expectedValue = static_cast<uint64_t>(i + 1) * 100;
    transport.wait_signal(i, IbgdaCmpOp::EQ, expectedValue);

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

void runTestP2pTransportConstruction(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    bool* d_success) {
  testP2pTransportConstruction<<<1, 1>>>(localBuf, remoteBuf, d_success);
}

void runTestP2pTransportDefaultConstruction(bool* d_success) {
  testP2pTransportDefaultConstruction<<<1, 1>>>(d_success);
}

void runTestP2pTransportNumSignals(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success) {
  testP2pTransportNumSignals<<<1, 1>>>(
      localBuf, remoteBuf, numSignals, d_success);
}

void runTestP2pTransportSignalPointerArithmetic(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success) {
  testP2pTransportSignalPointerArithmetic<<<1, 1>>>(
      localBuf, remoteBuf, numSignals, d_success);
}

void runTestP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success) {
  testP2pTransportReadSignal<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, numSignals, d_success);
}

void runTestIbgdaWork(bool* d_success) {
  testIbgdaWork<<<1, 1>>>(d_success);
}

void runTestWaitSignalEQ(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalEQ<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, targetValue, d_success);
}

void runTestWaitSignalNE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalNE<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalGE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalGE<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalGT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalGT<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalLE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalLE<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalLT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalLT<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success) {
  testWaitSignalMultipleSlots<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, numSignals, d_success);
}

// =============================================================================
// Group-level API test kernels
// =============================================================================

// Test that put_group correctly partitions data across warp lanes.
// We can't call the real DOCA put_warp without a real QP, so this test
// verifies the partitioning logic (offset/chunk calculation and
// subBuffer arithmetic) which is the GPU-side logic we can test.
__global__ void testPutGroupPartitioning(bool* success) {
  *success = true;

  // Simulate the partitioning logic that put_group does
  auto group = comms::pipes::make_warp_group();
  if (group.group_size != comms::pipes::kWarpSize) {
    *success = false;
    return;
  }

  constexpr std::size_t kTotalBytes = 1024; // 1KB
  constexpr std::size_t kChunkSize = kTotalBytes / comms::pipes::kWarpSize;

  // Verify each lane gets the right offset and chunk size
  std::size_t expectedOffset = group.thread_id_in_group * kChunkSize;
  std::size_t expectedChunk = kChunkSize;

  // Create a mock buffer at a known base address
  // Use a stack-local array as a stand-in for the buffer pointer
  char baseData[8]; // just need an address
  void* basePtr = baseData;

  comms::pipes::IbgdaLocalBuffer baseBuf(
      basePtr, comms::pipes::NetworkLKey(0x1111));
  comms::pipes::IbgdaLocalBuffer laneBuf = baseBuf.subBuffer(expectedOffset);

  // Verify the sub-buffer pointer is at the correct offset
  auto* expectedPtr = static_cast<char*>(basePtr) + expectedOffset;
  if (laneBuf.ptr != expectedPtr) {
    *success = false;
  }

  // Verify the key is preserved
  if (laneBuf.lkey != baseBuf.lkey) {
    *success = false;
  }

  // Verify chunk size is correct
  if (expectedChunk != kChunkSize) {
    *success = false;
  }
}

// Test that put_signal_group correctly broadcasts the signal ticket
// from lane 0 to all lanes. Simulates the broadcast pattern without
// calling actual DOCA operations.
__global__ void testPutSignalGroupBroadcast(bool* success) {
  *success = true;

  auto group = comms::pipes::make_warp_group();
  if (group.group_size != comms::pipes::kWarpSize) {
    *success = false;
    return;
  }

  // Simulate what put_signal_group does for the signal broadcast:
  // Leader produces a ticket, broadcasts to all threads via broadcast_64
  uint64_t signalTicket = 0;
  if (group.is_leader()) {
    signalTicket = 0xCAFEBABE12345678ULL;
  }

  // Broadcast from leader to all threads
  signalTicket = group.broadcast_64(signalTicket);

  // Every thread should see the leader's value
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
// broadcast_64 test kernels for BLOCK and MULTIWARP scopes
// =============================================================================

// Test broadcast_64 with BLOCK scope
// Launched with multiple blocks: only writes false on failure (initialized to
// true by host) to avoid inter-block data races.
__global__ void testBroadcast64Block(bool* success) {
  auto group = comms::pipes::make_block_group();

  // Leader produces a value, broadcasts to all threads
  uint64_t val = 0;
  if (group.is_leader()) {
    val = 0xDEADBEEF42424242ULL;
  }

  val = group.broadcast_64(val);

  if (val != 0xDEADBEEF42424242ULL) {
    *success = false;
  }
}

// Test broadcast_64 with MULTIWARP scope
// Launched with multiple blocks: only writes false on failure.
__global__ void testBroadcast64Multiwarp(bool* success) {
  auto group = comms::pipes::make_multiwarp_group();

  // Each multiwarp leader produces a unique value based on group_id
  uint64_t val = 0;
  if (group.is_leader()) {
    val = 0xAAAABBBB00000000ULL + group.group_id;
  }

  val = group.broadcast_64(val);

  // All threads in the multiwarp should see their leader's value
  uint64_t expected = 0xAAAABBBB00000000ULL + group.group_id;
  if (val != expected) {
    *success = false;
  }
}

// Test double-broadcast safety (the double-sync pattern)
// Two consecutive broadcasts with different values should not race.
// Launched with multiple blocks: only writes false on failure.
__global__ void testBroadcast64DoubleSafety(bool* success) {
  auto group = comms::pipes::make_block_group();

  // First broadcast
  uint64_t val1 = 0;
  if (group.is_leader()) {
    val1 = 0x1111111111111111ULL;
  }
  val1 = group.broadcast_64(val1);

  if (val1 != 0x1111111111111111ULL) {
    *success = false;
  }

  // Second broadcast with different value — must not race with first
  uint64_t val2 = 0;
  if (group.is_leader()) {
    val2 = 0x2222222222222222ULL;
  }
  val2 = group.broadcast_64(val2);

  if (val2 != 0x2222222222222222ULL) {
    *success = false;
  }
}

// Test put_group partitioning logic with block-sized groups.
// Launched with multiple blocks: only writes false on failure.
__global__ void testPutGroupPartitioningBlock(bool* success) {
  auto group = comms::pipes::make_block_group();

  constexpr std::size_t kTotalBytes = 4096; // 4KB
  std::size_t chunkSize = kTotalBytes / group.group_size;
  std::size_t expectedOffset = group.thread_id_in_group * chunkSize;

  // Create a mock buffer at a known base address
  char baseData[8]; // just need an address
  void* basePtr = baseData;

  comms::pipes::IbgdaLocalBuffer baseBuf(
      basePtr, comms::pipes::NetworkLKey(0x1111));
  comms::pipes::IbgdaLocalBuffer laneBuf = baseBuf.subBuffer(expectedOffset);

  // Verify the sub-buffer pointer is at the correct offset
  auto* expectedPtr = static_cast<char*>(basePtr) + expectedOffset;
  if (laneBuf.ptr != expectedPtr) {
    *success = false;
  }

  // Verify the key is preserved
  if (laneBuf.lkey != baseBuf.lkey) {
    *success = false;
  }
}

// =============================================================================
// broadcast_64 / block-scope test wrapper functions
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

} // namespace comms::pipes::tests
