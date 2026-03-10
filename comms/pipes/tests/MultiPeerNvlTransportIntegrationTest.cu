// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultiPeerNvlTransportIntegrationTest.cuh"

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes::test {

// =============================================================================
// TestDeviceBundle Accessors Test
// =============================================================================

__global__ void multiPeerDeviceTransportAccessorsKernel(
    const TestDeviceBundle& bundle,
    int* results) {
  results[0] = bundle.rank();
  results[1] = bundle.n_ranks();
  results[2] = bundle.num_peers();
}

void testMultiPeerDeviceTransportAccessors(
    const TestDeviceBundle& bundle,
    int* results) {
  multiPeerDeviceTransportAccessorsKernel<<<1, 1>>>(bundle, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Signal/Wait Test
// =============================================================================

__global__ void signalWaitKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    bundle.signal.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
    *result = 1;
  } else {
    bundle.signal.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
    *result = 1;
  }
}

void testSignalWait(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  signalWaitKernel<<<1, 32>>>(
      bundle, targetRank, signalIdx, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Test
// =============================================================================

__global__ void
barrierKernel(TestDeviceBundle bundle, int barrierIdx, int* result) {
  auto group = make_warp_group();

  bundle.barrier.barrier(group, barrierIdx);

  *result = 1;
}

void testBarrier(TestDeviceBundle& bundle, int barrierIdx, int* result) {
  barrierKernel<<<1, 32>>>(bundle, barrierIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Single-Peer Send/Recv Tests
// =============================================================================

__global__ void singlePeerSendKernel(
    TestDeviceBundle bundle,
    int peerRank,
    void* srcBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  bundle.handle.get_nvl(peerRank).send(group, srcBuff, nbytes);
}

void testSinglePeerSend(
    TestDeviceBundle& bundle,
    int peerRank,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerSendKernel<<<numBlocks, blockSize>>>(
      bundle, peerRank, srcBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

__global__ void singlePeerRecvKernel(
    TestDeviceBundle bundle,
    int peerRank,
    void* dstBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  bundle.handle.get_nvl(peerRank).recv(group, dstBuff, nbytes);
}

void testSinglePeerRecv(
    TestDeviceBundle& bundle,
    int peerRank,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerRecvKernel<<<numBlocks, blockSize>>>(
      bundle, peerRank, dstBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Multi-Peer Send/Recv Test (Parallel via Partition)
// =============================================================================

__global__ void multiPeerSendRecvAllPeersKernel(
    TestDeviceBundle bundle,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer) {
  auto group = make_warp_group();

  int myRank = bundle.rank();
  int numPeers = bundle.num_peers();

  // Partition into send and recv groups (interleaved for SM balance)
  auto [partition_id, send_recv_group] = group.partition_interleaved(2);

  // Further partition across peers
  auto [peer_idx, group_per_peer] =
      send_recv_group.partition_interleaved(numPeers);

  // Map peer_idx to actual rank (skip self)
  int peer_rank = peer_idx < myRank ? peer_idx : peer_idx + 1;

  if (partition_id == 0) {
    bundle.handle.get_nvl(peer_rank).send(
        group_per_peer, srcBuffs[peer_idx], nbytesPerPeer);
  } else {
    bundle.handle.get_nvl(peer_rank).recv(
        group_per_peer, dstBuffs[peer_idx], nbytesPerPeer);
  }
}

void testMultiPeerSendRecvAllPeers(
    TestDeviceBundle& bundle,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize) {
  multiPeerSendRecvAllPeersKernel<<<numBlocks, blockSize>>>(
      bundle, srcBuffs, dstBuffs, nbytesPerPeer);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Concurrent Signal Multi-Block Test
// =============================================================================

__global__ void concurrentSignalMultiBlockKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results) {
  auto group = make_warp_group();

  // Each block uses a different signal slot (blockIdx.x % numSlots)
  auto slotId = blockIdx.x % numSlots;

  if (isSignaler) {
    bundle.signal.signal_peer(
        group, targetRank, slotId, SignalOp::SIGNAL_ADD, 1);
  } else {
    bundle.signal.wait_signal(group, slotId, CmpOp::CMP_GE, 1);
  }

  // Record success for this block
  if (threadIdx.x == 0) {
    results[blockIdx.x] = 1;
  }
}

void testConcurrentSignalMultiBlock(
    TestDeviceBundle& bundle,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int numBlocks) {
  concurrentSignalMultiBlockKernel<<<numBlocks, 32>>>(
      bundle, targetRank, numSlots, isSignaler, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Put Operation Test
// =============================================================================

__global__ void putOperationKernel(
    TestDeviceBundle bundle,
    int targetRank,
    void* remoteDst,
    const void* localSrc,
    std::size_t nbytes,
    int signalId,
    bool isWriter,
    int* result) {
  auto group = make_warp_group();

  if (isWriter) {
    // put_signal: put + sync + signal_peer
    bundle.handle.get_nvl(targetRank)
        .put(
            group,
            static_cast<char*>(remoteDst),
            static_cast<const char*>(localSrc),
            nbytes);
    group.sync();
    bundle.signal.signal_peer(
        group, targetRank, signalId, SignalOp::SIGNAL_ADD, 1);
  } else {
    bundle.signal.wait_signal(group, signalId, CmpOp::CMP_GE, 1);
  }

  if (threadIdx.x == 0) {
    *result = 1;
  }
}

void testPutOperation(
    TestDeviceBundle& bundle,
    int targetRank,
    void* remoteDst,
    const void* localSrc,
    std::size_t nbytes,
    int signalId,
    bool isWriter,
    int* result) {
  putOperationKernel<<<1, 32>>>(
      bundle,
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
    const TestDeviceBundle& bundle,
    int* results) {
  // Output numPeers in results[0]
  results[0] = bundle.num_peers();

  // Self transport type
  int myRank = bundle.rank();
  results[1 + myRank] = static_cast<int>(bundle.handle.transports[myRank].type);

  // Peer transport types
  int nRanks = bundle.n_ranks();
  for (int r = 0; r < nRanks; ++r) {
    if (r == myRank)
      continue;
    results[1 + r] = static_cast<int>(bundle.handle.transports[r].type);
  }
}

void testTransportTypes(const TestDeviceBundle& bundle, int* results) {
  transportTypesKernel<<<1, 1>>>(bundle, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Concurrent Signal Multi-Warp Test
// =============================================================================

__global__ void concurrentSignalWaitMultiWarpKernel(
    TestDeviceBundle bundle,
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
    bundle.signal.signal_peer(
        group, targetRank, slotId, SignalOp::SIGNAL_ADD, 1);
  } else {
    bundle.signal.wait_signal(group, slotId, CmpOp::CMP_GE, 1);
  }

  // Record success for this warp
  if (laneIdx == 0) {
    results[warpIdx] = 1;
  }
}

void testConcurrentSignalWaitMultiWarp(
    TestDeviceBundle& bundle,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int warpsPerBlock) {
  int blockSize = warpsPerBlock * 32; // 32 threads per warp
  concurrentSignalWaitMultiWarpKernel<<<1, blockSize>>>(
      bundle, targetRank, numSlots, isSignaler, results, warpsPerBlock);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// signal_all() Test
// =============================================================================

__global__ void signalAllKernel(
    TestDeviceBundle bundle,
    int signalerRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = bundle.rank();

  if (myRank == signalerRank) {
    bundle.signal.signal_all(group, signalIdx, SignalOp::SIGNAL_ADD, 1);
  } else {
    bundle.signal.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
  }

  *result = 1;
}

void testSignalAll(
    TestDeviceBundle& bundle,
    int signalerRank,
    int signalIdx,
    int* result) {
  signalAllKernel<<<1, 32>>>(bundle, signalerRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// signal_all() + read_signal() Aggregate Test
// =============================================================================

__global__ void signalAllAggregateDistributedKernel(
    TestDeviceBundle bundle,
    int signalIdx,
    uint64_t* result) {
  auto group = make_warp_group();

  // Every rank signals all peers with value 1
  bundle.signal.signal_all(group, signalIdx, SignalOp::SIGNAL_ADD, 1);

  // Wait until aggregate reaches nRanks-1 (all peers have signaled us)
  bundle.signal.wait_signal(
      group, signalIdx, CmpOp::CMP_GE, bundle.num_peers());

  // Read aggregate (thread-level API)
  if (group.is_leader()) {
    *result = bundle.signal.read_signal(signalIdx);
  }
}

void testSignalAllAggregateDistributed(
    TestDeviceBundle& bundle,
    int signalIdx,
    uint64_t* result) {
  signalAllAggregateDistributedKernel<<<1, 32>>>(bundle, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal_from_all() Test
// =============================================================================

__global__ void waitSignalFromAllKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = bundle.rank();

  if (myRank == targetRank) {
    int nRanks = bundle.n_ranks();
    bundle.signal.wait_signal(group, signalIdx, CmpOp::CMP_GE, nRanks - 1);
  } else {
    bundle.signal.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
  }

  *result = 1;
}

void testWaitSignalFromAll(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalFromAllKernel<<<1, 32>>>(bundle, targetRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Wait with CMP_EQ Test
// =============================================================================

__global__ void waitWithCmpEqKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    bundle.signal.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_SET, expectedValue);
  } else {
    bundle.signal.wait_signal(group, signalIdx, CmpOp::CMP_EQ, expectedValue);
  }

  *result = 1;
}

void testWaitWithCmpEq(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result) {
  waitWithCmpEqKernel<<<1, 32>>>(
      bundle, targetRank, signalIdx, expectedValue, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Monotonic Wait Values Test
// =============================================================================

__global__ void monotonicWaitValuesKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  for (int i = 0; i < numIterations; ++i) {
    if (isSignaler) {
      bundle.signal.signal_peer(
          group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
    } else {
      bundle.signal.wait_signal(group, signalIdx, CmpOp::CMP_GE, i + 1);
    }
    group.sync();
  }

  *result = 1;
}

void testMonotonicWaitValues(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result) {
  monotonicWaitValuesKernel<<<1, 32>>>(
      bundle, targetRank, signalIdx, numIterations, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// SIGNAL_SET Integration Test
// =============================================================================

__global__ void signalWithSetKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    bundle.signal.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_SET, setValue);
  } else {
    bundle.signal.wait_signal(group, signalIdx, CmpOp::CMP_GE, setValue);
  }

  *result = 1;
}

void testSignalWithSet(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result) {
  signalWithSetKernel<<<1, 32>>>(
      bundle, targetRank, signalIdx, setValue, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Monotonic Counters Test
// =============================================================================

__global__ void barrierMonotonicKernel(
    TestDeviceBundle bundle,
    int barrierIdx,
    int numPhases,
    int* result) {
  auto group = make_warp_group();

  for (int phase = 0; phase < numPhases; ++phase) {
    bundle.barrier.barrier(group, barrierIdx);
  }

  if (threadIdx.x == 0) {
    *result = 1;
  }
}

void testBarrierMonotonic(
    TestDeviceBundle& bundle,
    int barrierIdx,
    int numPhases,
    int* result) {
  barrierMonotonicKernel<<<1, 32>>>(bundle, barrierIdx, numPhases, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Multi-Block Stress Test
// =============================================================================

__global__ void barrierMultiBlockStressKernel(
    TestDeviceBundle bundle,
    int numSlots,
    int* results) {
  auto group = make_warp_group();

  uint32_t slotId = blockIdx.x % numSlots;

  bundle.barrier.barrier(group, slotId);

  if (threadIdx.x == 0) {
    results[blockIdx.x] = 1;
  }
}

void testBarrierMultiBlockStress(
    TestDeviceBundle& bundle,
    int numSlots,
    int* results,
    int numBlocks) {
  barrierMultiBlockStressKernel<<<numBlocks, 32>>>(bundle, numSlots, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Peer Test (Two-Sided Barrier)
// =============================================================================

__global__ void barrierPeerKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int barrierIdx,
    int* result) {
  auto group = make_warp_group();

  // DeviceWindowBarrier::barrier_peer takes peer_index, not global rank
  int peer_index = bundle.rank_to_peer_index(targetRank);
  bundle.barrier.barrier_peer(peer_index, group, barrierIdx);

  if (threadIdx.x == 0) {
    *result = 1;
  }
}

void testBarrierPeer(
    TestDeviceBundle& bundle,
    int targetRank,
    int barrierIdx,
    int* result) {
  barrierPeerKernel<<<1, 32>>>(bundle, targetRank, barrierIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal_from() Basic Test
// =============================================================================

__global__ void waitSignalFromPeerKernel(
    TestDeviceBundle bundle,
    int peerRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    bundle.signal.signal_peer(
        group, peerRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
  } else {
    bundle.signal.wait_signal_from(
        group, peerRank, signalIdx, CmpOp::CMP_GE, 1);
    uint64_t val = bundle.signal.read_signal_from(peerRank, signalIdx);
    if (val < 1) {
      *result = 0;
      return;
    }
  }

  *result = 1;
}

void testWaitSignalFromPeer(
    TestDeviceBundle& bundle,
    int peerRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  waitSignalFromPeerKernel<<<1, 32>>>(
      bundle, peerRank, signalIdx, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal_from() Multi-Peer Isolation Test
// =============================================================================

__global__ void waitSignalFromMultiPeerIsolationKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = bundle.rank();
  int nRanks = bundle.n_ranks();

  if (myRank == targetRank) {
    for (int r = 0; r < nRanks; ++r) {
      if (r == myRank) {
        continue;
      }
      uint64_t expectedValue = static_cast<uint64_t>(r + 1);
      bundle.signal.wait_signal_from(
          group, r, signalIdx, CmpOp::CMP_GE, expectedValue);
      uint64_t val = bundle.signal.read_signal_from(r, signalIdx);
      if (val != expectedValue) {
        *result = 0;
        return;
      }
    }
  } else {
    uint64_t signalValue = static_cast<uint64_t>(myRank + 1);
    bundle.signal.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_SET, signalValue);
  }

  *result = 1;
}

void testWaitSignalFromMultiPeerIsolation(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalFromMultiPeerIsolationKernel<<<1, 32>>>(
      bundle, targetRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal() and wait_signal_from() Both Work Test
// =============================================================================

__global__ void waitSignalAndWaitSignalFromBothWorkKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = bundle.rank();
  int nRanks = bundle.n_ranks();

  if (myRank == targetRank) {
    bundle.signal.wait_signal(group, signalIdx, CmpOp::CMP_GE, nRanks - 1);

    for (int r = 0; r < nRanks; ++r) {
      if (r == myRank) {
        continue;
      }
      bundle.signal.wait_signal_from(group, r, signalIdx, CmpOp::CMP_GE, 1);
    }
  } else {
    bundle.signal.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
  }

  *result = 1;
}

void testWaitSignalAndWaitSignalFromBothWork(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalAndWaitSignalFromBothWorkKernel<<<1, 32>>>(
      bundle, targetRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Signal/Wait Test (BLOCK Scope - exercises fallback path)
// =============================================================================

__global__ void signalWaitBlockScopeKernel(
    TestDeviceBundle bundle,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_block_group(); // Uses SyncScope::BLOCK

  if (isSignaler) {
    bundle.signal.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
    if (threadIdx.x == 0) {
      *result = 1;
    }
  } else {
    bundle.signal.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
    if (threadIdx.x == 0) {
      *result = 1;
    }
  }
}

void testSignalWaitBlockScope(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  // Launch with 1 block of 128 threads (BLOCK scope = all 128 threads in block)
  signalWaitBlockScopeKernel<<<1, 128>>>(
      bundle, targetRank, signalIdx, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

} // namespace comms::pipes::test
