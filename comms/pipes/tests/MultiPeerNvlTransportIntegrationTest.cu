// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultiPeerNvlTransportIntegrationTest.cuh"

#include "comms/pipes/DeviceSignal.cuh"
#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes::test {

// =============================================================================
// MultiPeerDeviceTransport Accessors Test
// =============================================================================

__global__ void multiPeerDeviceTransportAccessorsKernel(
    MultiPeerDeviceTransport& transport,
    int* results) {
  results[0] = transport.rank();
  results[1] = transport.n_ranks();
  results[2] = transport.num_peers();
}

void testMultiPeerDeviceTransportAccessors(
    MultiPeerDeviceTransport& transport,
    int* results) {
  multiPeerDeviceTransportAccessorsKernel<<<1, 1>>>(transport, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Signal/Wait Test
// =============================================================================

__global__ void signalWaitKernel(
    MultiPeerDeviceTransport& transport,
    int peer,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    // Signal the peer's inbox (peer is peer index, not rank)
    transport.signal_peer(peer, group, signalIdx, SignalOp::SIGNAL_ADD, 1);
    *result = 1;
  } else {
    // Wait for signal (peer signals us, we wait on our inbox)
    transport.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
    *result = 1;
  }
}

void testSignalWait(
    MultiPeerDeviceTransport& transport,
    int peer,
    int signalIdx,
    bool isSignaler,
    int* result) {
  signalWaitKernel<<<1, 32>>>(transport, peer, signalIdx, isSignaler, result);
}

// =============================================================================
// Single-Peer Send/Recv Tests
// =============================================================================

__global__ void singlePeerSendKernel(
    MultiPeerDeviceTransport& transport,
    int peer,
    void* srcBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  transport.send(peer, group, srcBuff, nbytes);
}

void testSinglePeerSend(
    MultiPeerDeviceTransport& transport,
    int peer,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerSendKernel<<<numBlocks, blockSize>>>(
      transport, peer, srcBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

__global__ void singlePeerRecvKernel(
    MultiPeerDeviceTransport& transport,
    int peer,
    void* dstBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  transport.recv(peer, group, dstBuff, nbytes);
}

void testSinglePeerRecv(
    MultiPeerDeviceTransport& transport,
    int peer,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerRecvKernel<<<numBlocks, blockSize>>>(
      transport, peer, dstBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Multi-Peer Send/Recv Tests (Peer Iteration)
// =============================================================================

__global__ void multiPeerSendAllPeersKernel(
    MultiPeerDeviceTransport& transport,
    void** srcBuffs,
    std::size_t nbytesPerPeer) {
  auto group = make_warp_group();

  // Use peer iteration helpers to send to all peers
  // send() now takes peer index directly
  int numPeers = transport.num_peers();
  for (int i = 0; i < numPeers; ++i) {
    transport.send(i, group, srcBuffs[i], nbytesPerPeer, i);
  }
}

void testMultiPeerSendAllPeers(
    MultiPeerDeviceTransport& transport,
    void** srcBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize) {
  multiPeerSendAllPeersKernel<<<numBlocks, blockSize>>>(
      transport, srcBuffs, nbytesPerPeer);
  CUDACHECK_TEST(cudaGetLastError());
}

__global__ void multiPeerRecvAllPeersKernel(
    MultiPeerDeviceTransport& transport,
    void** dstBuffs,
    std::size_t nbytesPerPeer) {
  auto group = make_warp_group();

  // Use peer iteration helpers to receive from all peers
  // recv() now takes peer index directly
  int numPeers = transport.num_peers();
  for (int i = 0; i < numPeers; ++i) {
    transport.recv(i, group, dstBuffs[i], nbytesPerPeer, i);
  }
}

void testMultiPeerRecvAllPeers(
    MultiPeerDeviceTransport& transport,
    void** dstBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize) {
  multiPeerRecvAllPeersKernel<<<numBlocks, blockSize>>>(
      transport, dstBuffs, nbytesPerPeer);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Concurrent Signal Multi-Block Test
// =============================================================================

__global__ void concurrentSignalMultiBlockKernel(
    MultiPeerDeviceTransport& transport,
    int peer,
    int numSlots,
    bool isSignaler,
    int* results) {
  auto group = make_warp_group();

  // Each block uses a different signal slot (blockIdx.x % numSlots)
  auto slotId = blockIdx.x % numSlots;

  if (isSignaler) {
    // Signal the peer's inbox at this slot (peer is peer index, not rank)
    transport.signal_peer(peer, group, slotId, SignalOp::SIGNAL_ADD, 1);
  } else {
    // Wait for signal at this slot (peer signals us, we wait on our inbox)
    transport.wait_signal(group, slotId, CmpOp::CMP_GE, 1);
  }

  // Record success for this block
  if (threadIdx.x == 0) {
    results[blockIdx.x] = 1;
  }
}

void testConcurrentSignalMultiBlock(
    MultiPeerDeviceTransport& transport,
    int peer,
    int numSlots,
    bool isSignaler,
    int* results,
    int numBlocks) {
  concurrentSignalMultiBlockKernel<<<numBlocks, 32>>>(
      transport, peer, numSlots, isSignaler, results);
}

// =============================================================================
// Transport Types Test
// =============================================================================

__global__ void transportTypesKernel(
    const MultiPeerDeviceTransport& transport,
    int* results) {
  // Output numPeers in results[0]
  results[0] = transport.num_peers();

  // Self transport type
  int myRank = transport.rank();
  results[1 + myRank] = static_cast<int>(transport.get_self_transport()->type);

  // Peer transport types
  int numPeers = transport.num_peers();
  for (int peerIdx = 0; peerIdx < numPeers; ++peerIdx) {
    int rank = transport.peer_index_to_rank(peerIdx);
    results[1 + rank] =
        static_cast<int>(transport.get_peer_transport(peerIdx)->type);
  }
}

void testTransportTypes(
    const MultiPeerDeviceTransport& transport,
    int* results) {
  transportTypesKernel<<<1, 1>>>(transport, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Concurrent Signal Multi-Warp Test
// =============================================================================

__global__ void concurrentSignalWaitMultiWarpKernel(
    MultiPeerDeviceTransport& transport,
    int peer,
    int numSlots,
    bool isSignaler,
    int* results,
    int warpsPerBlock) {
  // Each warp uses a different signal slot based on its warp index
  uint32_t warpIdx = threadIdx.x / 32;
  uint32_t laneIdx = threadIdx.x % 32;

  // Only process if this warp is within the configured range
  if (warpIdx >= warpsPerBlock) {
    return;
  }

  // Create a warp-level thread group
  auto group = make_warp_group();

  // Use different slot per warp (warpIdx % numSlots)
  int slotId = warpIdx % numSlots;

  if (isSignaler) {
    // Signal the peer's inbox at this slot (peer is peer index, not rank)
    transport.signal_peer(peer, group, slotId, SignalOp::SIGNAL_ADD, 1);
  } else {
    // Wait for signal at this slot (peer signals us, we wait on our inbox)
    transport.wait_signal(group, slotId, CmpOp::CMP_GE, 1);
  }

  // Record success for this warp (only lane 0 writes)
  if (laneIdx == 0) {
    results[warpIdx] = 1;
  }
}

void testConcurrentSignalWaitMultiWarp(
    MultiPeerDeviceTransport& transport,
    int peer,
    int numSlots,
    bool isSignaler,
    int* results,
    int warpsPerBlock) {
  int blockSize = warpsPerBlock * 32; // 32 threads per warp
  concurrentSignalWaitMultiWarpKernel<<<1, blockSize>>>(
      transport, peer, numSlots, isSignaler, results, warpsPerBlock);
}

// =============================================================================
// signal_all() Test
// =============================================================================

__global__ void signalAllKernel(
    MultiPeerDeviceTransport& transport,
    int signalerRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = transport.rank();

  if (myRank == signalerRank) {
    // This rank signals all peers
    transport.get_signal().signal_all(
        group, signalIdx, SignalOp::SIGNAL_ADD, 1);
  } else {
    // Wait for signal from the signaler (signaler writes to our inbox)
    transport.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
  }

  *result = 1;
}

void testSignalAll(
    MultiPeerDeviceTransport& transport,
    int signalerRank,
    int signalIdx,
    int* result) {
  signalAllKernel<<<1, 32>>>(transport, signalerRank, signalIdx, result);
}

// =============================================================================
// wait_signal_from_all() Test
// =============================================================================

__global__ void waitSignalFromAllKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = transport.rank();

  if (myRank == targetRank) {
    // Wait for accumulated signals from all (nRanks - 1) peers
    int nRanks = transport.n_ranks();
    transport.wait_signal(group, signalIdx, CmpOp::CMP_GE, nRanks - 1);
  } else {
    // Signal the target rank - convert rank to peer index
    int peer = transport.rank_to_peer_index(targetRank);
    transport.signal_peer(peer, group, signalIdx, SignalOp::SIGNAL_ADD, 1);
  }

  *result = 1;
}

void testWaitSignalFromAll(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalFromAllKernel<<<1, 32>>>(transport, targetRank, signalIdx, result);
}

// =============================================================================
// Wait with CMP_EQ Test
// =============================================================================

__global__ void waitWithCmpEqKernel(
    MultiPeerDeviceTransport& transport,
    int peer,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    // Signal with the exact expected value using SIGNAL_SET
    transport.signal_peer(
        peer, group, signalIdx, SignalOp::SIGNAL_SET, expectedValue);
  } else {
    // Wait for the exact value with CMP_EQ
    transport.wait_signal(group, signalIdx, CmpOp::CMP_EQ, expectedValue);
  }

  *result = 1;
}

void testWaitWithCmpEq(
    MultiPeerDeviceTransport& transport,
    int peer,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result) {
  waitWithCmpEqKernel<<<1, 32>>>(
      transport, peer, signalIdx, expectedValue, isSignaler, result);
}

// =============================================================================
// Monotonic Wait Values Test
// =============================================================================

__global__ void monotonicWaitValuesKernel(
    MultiPeerDeviceTransport& transport,
    int peer,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  for (int i = 0; i < numIterations; ++i) {
    if (isSignaler) {
      // Signal with value=1 (adds to accumulated value)
      transport.signal_peer(peer, group, signalIdx, SignalOp::SIGNAL_ADD, 1);
    } else {
      // Wait for accumulated value (i+1)
      transport.wait_signal(group, signalIdx, CmpOp::CMP_GE, i + 1);
    }
    group.sync();
  }

  *result = 1;
}

void testMonotonicWaitValues(
    MultiPeerDeviceTransport& transport,
    int peer,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result) {
  monotonicWaitValuesKernel<<<1, 32>>>(
      transport, peer, signalIdx, numIterations, isSignaler, result);
}

// =============================================================================
// SIGNAL_SET Integration Test
// =============================================================================

__global__ void signalWithSetKernel(
    MultiPeerDeviceTransport& transport,
    int peer,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    // Use SIGNAL_SET instead of SIGNAL_ADD
    transport.signal_peer(
        peer, group, signalIdx, SignalOp::SIGNAL_SET, setValue);
  } else {
    // Wait for the set value
    transport.wait_signal(group, signalIdx, CmpOp::CMP_GE, setValue);
  }

  *result = 1;
}

void testSignalWithSet(
    MultiPeerDeviceTransport& transport,
    int peer,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result) {
  signalWithSetKernel<<<1, 32>>>(
      transport, peer, signalIdx, setValue, isSignaler, result);
}

} // namespace comms::pipes::test
