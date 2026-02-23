// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultiPeerNvlTransportIntegrationTest.cuh"

#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/window/DeviceWindowBarrier.cuh"
#include "comms/pipes/window/DeviceWindowSignal.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes::test {

// =============================================================================
// MultiPeerDeviceTransport Accessors Test
// =============================================================================

__global__ void multiPeerDeviceTransportAccessorsKernel(
    const MultiPeerDeviceTransport& transport,
    int* results) {
  results[0] = transport.rank();
  results[1] = transport.n_ranks();
  results[2] = transport.num_peers();
}

void testMultiPeerDeviceTransportAccessors(
    const MultiPeerDeviceTransport& transport,
    int* results) {
  multiPeerDeviceTransportAccessorsKernel<<<1, 1>>>(transport, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Signal/Wait Test
// =============================================================================

__global__ void signalWaitKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    // Signal the peer's inbox (peer is peer index, not rank)
    transport.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
    *result = 1;
  } else {
    // Wait for signal (peer signals us, we wait on our inbox)
    transport.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
    *result = 1;
  }
}

void testSignalWait(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  signalWaitKernel<<<1, 32>>>(
      transport, targetRank, signalIdx, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Test
// =============================================================================

__global__ void barrierKernel(
    MultiPeerDeviceTransport& transport,
    int barrierIdx,
    int* result) {
  auto group = make_warp_group();

  // Execute barrier
  transport.barrier(group, barrierIdx);

  // If we get here, barrier succeeded
  *result = 1;
}

void testBarrier(
    MultiPeerDeviceTransport& transport,
    int barrierIdx,
    int* result) {
  barrierKernel<<<1, 32>>>(transport, barrierIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Single-Peer Send/Recv Tests
// =============================================================================

__global__ void singlePeerSendKernel(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    void* srcBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  transport.send(peerRank, group, srcBuff, nbytes);
}

void testSinglePeerSend(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerSendKernel<<<numBlocks, blockSize>>>(
      transport, peerRank, srcBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

__global__ void singlePeerRecvKernel(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    void* dstBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  transport.recv(peerRank, group, dstBuff, nbytes);
}

void testSinglePeerRecv(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerRecvKernel<<<numBlocks, blockSize>>>(
      transport, peerRank, dstBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Multi-Peer Send/Recv Test (Parallel via Partition)
// =============================================================================

__global__ void multiPeerSendRecvAllPeersKernel(
    MultiPeerDeviceTransport& transport,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer) {
  auto group = make_warp_group();

  int myRank = transport.rank();
  int numPeers = transport.num_peers();

  // Partition into send and recv groups (interleaved for SM balance)
  auto [partition_id, send_recv_group] = group.partition_interleaved(2);

  // Further partition across peers
  auto [peer_idx, group_per_peer] =
      send_recv_group.partition_interleaved(numPeers);

  // Map peer_idx to actual rank (skip self)
  int peer_rank = peer_idx < myRank ? peer_idx : peer_idx + 1;

  if (partition_id == 0) {
    transport.send(
        peer_rank, group_per_peer, srcBuffs[peer_idx], nbytesPerPeer, peer_idx);
  } else {
    transport.recv(
        peer_rank, group_per_peer, dstBuffs[peer_idx], nbytesPerPeer, peer_idx);
  }
}

void testMultiPeerSendRecvAllPeers(
    MultiPeerDeviceTransport& transport,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize) {
  multiPeerSendRecvAllPeersKernel<<<numBlocks, blockSize>>>(
      transport, srcBuffs, dstBuffs, nbytesPerPeer);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Concurrent Signal Multi-Block Test
// =============================================================================

__global__ void concurrentSignalMultiBlockKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results) {
  auto group = make_warp_group();

  // Each block uses a different signal slot (blockIdx.x % numSlots)
  auto slotId = blockIdx.x % numSlots;

  if (isSignaler) {
    // Signal the peer's inbox at this slot
    transport.signal_peer(group, targetRank, slotId, SignalOp::SIGNAL_ADD, 1);
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
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int numBlocks) {
  concurrentSignalMultiBlockKernel<<<numBlocks, 32>>>(
      transport, targetRank, numSlots, isSignaler, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Put Operation Test
// =============================================================================

__global__ void putOperationKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    void* remoteDst,
    const void* localSrc,
    std::size_t nbytes,
    int signalId,
    bool isWriter,
    int* result) {
  auto group = make_warp_group();

  if (isWriter) {
    // Use put_signal to write and signal completion
    transport.put_signal(
        targetRank, group, remoteDst, localSrc, nbytes, signalId, 1);
  } else {
    // Wait for signal indicating data is ready
    transport.wait_signal(group, signalId, CmpOp::CMP_GE, 1);
  }

  if (threadIdx.x == 0) {
    *result = 1;
  }
}

void testPutOperation(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    void* remoteDst,
    const void* localSrc,
    std::size_t nbytes,
    int signalId,
    bool isWriter,
    int* result) {
  putOperationKernel<<<1, 32>>>(
      transport,
      targetRank,
      remoteDst,
      localSrc,
      nbytes,
      signalId,
      isWriter,
      result);
  CUDACHECK_TEST(cudaGetLastError());
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
  int nRanks = transport.n_ranks();
  for (int r = 0; r < nRanks; ++r) {
    if (r == myRank)
      continue;
    results[1 + r] = static_cast<int>(transport.get_peer_transport(r)->type);
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
    int targetRank,
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
    // Signal the target rank's inbox at this slot (peer is peer index, not
    // rank)
    transport.signal_peer(group, targetRank, slotId, SignalOp::SIGNAL_ADD, 1);
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
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int warpsPerBlock) {
  int blockSize = warpsPerBlock * 32; // 32 threads per warp
  concurrentSignalWaitMultiWarpKernel<<<1, blockSize>>>(
      transport, targetRank, numSlots, isSignaler, results, warpsPerBlock);
  CUDACHECK_TEST(cudaGetLastError());
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
  CUDACHECK_TEST(cudaGetLastError());
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
    // Signal the target rank
    transport.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
  }

  *result = 1;
}

void testWaitSignalFromAll(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalFromAllKernel<<<1, 32>>>(transport, targetRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Wait with CMP_EQ Test
// =============================================================================

__global__ void waitWithCmpEqKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    // Signal with the exact expected value using SIGNAL_SET
    transport.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_SET, expectedValue);
  } else {
    // Wait for the exact value with CMP_EQ
    transport.wait_signal(group, signalIdx, CmpOp::CMP_EQ, expectedValue);
  }

  *result = 1;
}

void testWaitWithCmpEq(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result) {
  waitWithCmpEqKernel<<<1, 32>>>(
      transport, targetRank, signalIdx, expectedValue, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Monotonic Wait Values Test
// =============================================================================

__global__ void monotonicWaitValuesKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  for (int i = 0; i < numIterations; ++i) {
    if (isSignaler) {
      // Signal with value=1 (adds to accumulated value)
      transport.signal_peer(
          group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
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
    int targetRank,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result) {
  monotonicWaitValuesKernel<<<1, 32>>>(
      transport, targetRank, signalIdx, numIterations, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// SIGNAL_SET Integration Test
// =============================================================================

__global__ void signalWithSetKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    // Use SIGNAL_SET instead of SIGNAL_ADD
    transport.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_SET, setValue);
  } else {
    // Wait for the set value
    transport.wait_signal(group, signalIdx, CmpOp::CMP_GE, setValue);
  }

  *result = 1;
}

void testSignalWithSet(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result) {
  signalWithSetKernel<<<1, 32>>>(
      transport, targetRank, signalIdx, setValue, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Monotonic Counters Test
// =============================================================================

__global__ void barrierMonotonicKernel(
    MultiPeerDeviceTransport& transport,
    int barrierIdx,
    int numPhases,
    int* result) {
  auto group = make_warp_group();

  // Perform multiple barrier synchronizations on the same slot.
  // Counters accumulate monotonically — no reset needed.
  for (int phase = 0; phase < numPhases; ++phase) {
    transport.barrier(group, barrierIdx);
  }

  if (threadIdx.x == 0) {
    *result = 1;
  }
}

void testBarrierMonotonic(
    MultiPeerDeviceTransport& transport,
    int barrierIdx,
    int numPhases,
    int* result) {
  barrierMonotonicKernel<<<1, 32>>>(transport, barrierIdx, numPhases, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Multi-Block Stress Test
// =============================================================================

__global__ void barrierMultiBlockStressKernel(
    MultiPeerDeviceTransport& transport,
    int numSlots,
    int* results) {
  auto group = make_warp_group();

  // Each block uses a different barrier slot (blockIdx.x % numSlots)
  uint32_t slotId = blockIdx.x % numSlots;

  // Perform barrier synchronization
  transport.barrier(group, slotId);

  // Record success for this block
  if (threadIdx.x == 0) {
    results[blockIdx.x] = 1;
  }
}

void testBarrierMultiBlockStress(
    MultiPeerDeviceTransport& transport,
    int numSlots,
    int* results,
    int numBlocks) {
  barrierMultiBlockStressKernel<<<numBlocks, 32>>>(
      transport, numSlots, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Peer Test (Two-Sided Barrier)
// =============================================================================

__global__ void barrierPeerKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int barrierIdx,
    int* result) {
  auto group = make_warp_group();

  // Two-sided barrier: synchronize with specific peer
  transport.barrier_peer(targetRank, group, barrierIdx);

  // If we get here, barrier_peer succeeded
  if (threadIdx.x == 0) {
    *result = 1;
  }
}

void testBarrierPeer(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int barrierIdx,
    int* result) {
  barrierPeerKernel<<<1, 32>>>(transport, targetRank, barrierIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal_from() Basic Test
// =============================================================================

__global__ void waitSignalFromPeerKernel(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    // Signal the peer's inbox
    transport.signal_peer(group, peerRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
  } else {
    // Wait for signal from the specific peer using wait_signal_from
    transport.wait_signal_from(group, peerRank, signalIdx, CmpOp::CMP_GE, 1);
    // Verify read_signal_from also returns the correct value
    uint64_t val = transport.read_signal_from(peerRank, signalIdx);
    if (val < 1) {
      *result = 0;
      return;
    }
  }

  *result = 1;
}

void testWaitSignalFromPeer(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  waitSignalFromPeerKernel<<<1, 32>>>(
      transport, peerRank, signalIdx, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal_from() Multi-Peer Isolation Test
// =============================================================================

__global__ void waitSignalFromMultiPeerIsolationKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = transport.rank();
  int nRanks = transport.n_ranks();

  if (myRank == targetRank) {
    // Wait for signals from all peers, each with value = peerRank + 1
    for (int r = 0; r < nRanks; ++r) {
      if (r == myRank) {
        continue;
      }
      uint64_t expectedValue = static_cast<uint64_t>(r + 1);
      transport.wait_signal_from(
          group, r, signalIdx, CmpOp::CMP_GE, expectedValue);
      // Verify isolation: each peer's value is independent
      uint64_t val = transport.read_signal_from(r, signalIdx);
      if (val != expectedValue) {
        *result = 0;
        return;
      }
    }
  } else {
    // Signal the target rank with value = myRank + 1 using SIGNAL_SET
    uint64_t signalValue = static_cast<uint64_t>(myRank + 1);
    transport.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_SET, signalValue);
  }

  *result = 1;
}

void testWaitSignalFromMultiPeerIsolation(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalFromMultiPeerIsolationKernel<<<1, 32>>>(
      transport, targetRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal() and wait_signal_from() Both Work Test
// =============================================================================

__global__ void waitSignalAndWaitSignalFromBothWorkKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = transport.rank();
  int nRanks = transport.n_ranks();

  if (myRank == targetRank) {
    // Wait for accumulated signals from all (nRanks - 1) peers
    transport.wait_signal(group, signalIdx, CmpOp::CMP_GE, nRanks - 1);

    // Also verify wait_signal_from succeeds for each individual peer
    for (int r = 0; r < nRanks; ++r) {
      if (r == myRank) {
        continue;
      }
      transport.wait_signal_from(group, r, signalIdx, CmpOp::CMP_GE, 1);
    }
  } else {
    // Signal the target rank with SIGNAL_ADD, 1
    transport.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
  }

  *result = 1;
}

void testWaitSignalAndWaitSignalFromBothWork(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalAndWaitSignalFromBothWorkKernel<<<1, 32>>>(
      transport, targetRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Signal/Wait Test (BLOCK Scope - exercises fallback path)
// =============================================================================

__global__ void signalWaitBlockScopeKernel(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_block_group(); // Uses SyncScope::BLOCK

  if (isSignaler) {
    transport.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
    if (threadIdx.x == 0) {
      *result = 1;
    }
  } else {
    // This exercises the fallback path (non-WARP scope) in wait_signal()
    transport.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
    if (threadIdx.x == 0) {
      *result = 1;
    }
  }
}

void testSignalWaitBlockScope(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  // Launch with 1 block of 128 threads (BLOCK scope = all 128 threads in block)
  signalWaitBlockScopeKernel<<<1, 128>>>(
      transport, targetRank, signalIdx, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

} // namespace comms::pipes::test
