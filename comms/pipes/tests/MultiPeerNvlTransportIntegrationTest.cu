// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultiPeerNvlTransportIntegrationTest.cuh"

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/window/DeviceWindow.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes::test {

// =============================================================================
// DeviceWindow Accessors Test
// =============================================================================

__global__ void multiPeerDeviceTransportAccessorsKernel(
    const DeviceWindow& dw,
    int* results) {
  results[0] = dw.rank();
  results[1] = dw.n_ranks();
  results[2] = dw.num_peers();
}

void testMultiPeerDeviceTransportAccessors(
    const DeviceWindow& dw,
    int* results) {
  multiPeerDeviceTransportAccessorsKernel<<<1, 1>>>(dw, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Signal/Wait Test
// =============================================================================

__global__ void signalWaitKernel(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    dw.signal_peer(group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
    *result = 1;
  } else {
    dw.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
    *result = 1;
  }
}

void testSignalWait(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  signalWaitKernel<<<1, 32>>>(dw, targetRank, signalIdx, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Test
// =============================================================================

__global__ void barrierKernel(DeviceWindow& dw, int barrierIdx, int* result) {
  auto group = make_warp_group();

  dw.barrier(group, barrierIdx);

  *result = 1;
}

void testBarrier(DeviceWindow& dw, int barrierIdx, int* result) {
  barrierKernel<<<1, 32>>>(dw, barrierIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Single-Peer Send/Recv Tests
// =============================================================================

__global__ void singlePeerSendKernel(
    DeviceWindow& dw,
    int peerRank,
    void* srcBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  dw.get_handle().get_nvl(peerRank).send(group, srcBuff, nbytes);
}

void testSinglePeerSend(
    DeviceWindow& dw,
    int peerRank,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerSendKernel<<<numBlocks, blockSize>>>(dw, peerRank, srcBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

__global__ void singlePeerRecvKernel(
    DeviceWindow& dw,
    int peerRank,
    void* dstBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  dw.get_handle().get_nvl(peerRank).recv(group, dstBuff, nbytes);
}

void testSinglePeerRecv(
    DeviceWindow& dw,
    int peerRank,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerRecvKernel<<<numBlocks, blockSize>>>(dw, peerRank, dstBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Multi-Peer Send/Recv Test (Parallel via Partition)
// =============================================================================

__global__ void multiPeerSendRecvAllPeersKernel(
    DeviceWindow& dw,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer) {
  auto group = make_warp_group();

  int myRank = dw.rank();
  int numPeers = dw.num_peers();

  // Partition into send and recv groups (interleaved for SM balance)
  auto [partition_id, send_recv_group] = group.partition_interleaved(2);

  // Further partition across peers
  auto [peer_idx, group_per_peer] =
      send_recv_group.partition_interleaved(numPeers);

  // Map peer_idx to actual rank (skip self)
  int peer_rank = peer_idx < myRank ? peer_idx : peer_idx + 1;

  if (partition_id == 0) {
    dw.get_handle().get_nvl(peer_rank).send(
        group_per_peer, srcBuffs[peer_idx], nbytesPerPeer);
  } else {
    dw.get_handle().get_nvl(peer_rank).recv(
        group_per_peer, dstBuffs[peer_idx], nbytesPerPeer);
  }
}

void testMultiPeerSendRecvAllPeers(
    DeviceWindow& dw,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize) {
  multiPeerSendRecvAllPeersKernel<<<numBlocks, blockSize>>>(
      dw, srcBuffs, dstBuffs, nbytesPerPeer);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Concurrent Signal Multi-Block Test
// =============================================================================

__global__ void concurrentSignalMultiBlockKernel(
    DeviceWindow& dw,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results) {
  auto group = make_warp_group();

  // Each block uses a different signal slot (blockIdx.x % numSlots)
  auto slotId = blockIdx.x % numSlots;

  if (isSignaler) {
    dw.signal_peer(group, targetRank, slotId, SignalOp::SIGNAL_ADD, 1);
  } else {
    dw.wait_signal(group, slotId, CmpOp::CMP_GE, 1);
  }

  // Record success for this block
  if (threadIdx.x == 0) {
    results[blockIdx.x] = 1;
  }
}

void testConcurrentSignalMultiBlock(
    DeviceWindow& dw,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int numBlocks) {
  concurrentSignalMultiBlockKernel<<<numBlocks, 32>>>(
      dw, targetRank, numSlots, isSignaler, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Put Operation Test
// =============================================================================

__global__ void putOperationKernel(
    DeviceWindow& dw,
    int targetRank,
    LocalBufferRegistration srcBuf,
    std::size_t nbytes,
    int signalId,
    bool isWriter,
    int* result) {
  auto group = make_warp_group();

  if (isWriter) {
    dw.put_signal(group, targetRank, 0, srcBuf, 0, nbytes, signalId, 1);
  } else {
    dw.wait_signal(group, signalId, CmpOp::CMP_GE, 1);
  }

  if (threadIdx.x == 0) {
    *result = 1;
  }
}

void testPutOperation(
    DeviceWindow& dw,
    int targetRank,
    const LocalBufferRegistration& srcBuf,
    std::size_t nbytes,
    int signalId,
    bool isWriter,
    int* result) {
  putOperationKernel<<<1, 32>>>(
      dw, targetRank, srcBuf, nbytes, signalId, isWriter, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Transport Types Test
// =============================================================================

__global__ void transportTypesKernel(const DeviceWindow& dw, int* results) {
  // Output numPeers in results[0]
  results[0] = dw.num_peers();

  // Self transport type
  int myRank = dw.rank();
  results[1 + myRank] =
      static_cast<int>(dw.get_handle().transports[myRank].type);

  // Peer transport types
  int nRanks = dw.n_ranks();
  for (int r = 0; r < nRanks; ++r) {
    if (r == myRank)
      continue;
    results[1 + r] = static_cast<int>(dw.get_handle().transports[r].type);
  }
}

void testTransportTypes(const DeviceWindow& dw, int* results) {
  transportTypesKernel<<<1, 1>>>(dw, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Concurrent Signal Multi-Warp Test
// =============================================================================

__global__ void concurrentSignalWaitMultiWarpKernel(
    DeviceWindow& dw,
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
    dw.signal_peer(group, targetRank, slotId, SignalOp::SIGNAL_ADD, 1);
  } else {
    dw.wait_signal(group, slotId, CmpOp::CMP_GE, 1);
  }

  // Record success for this warp
  if (laneIdx == 0) {
    results[warpIdx] = 1;
  }
}

void testConcurrentSignalWaitMultiWarp(
    DeviceWindow& dw,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int warpsPerBlock) {
  int blockSize = warpsPerBlock * 32; // 32 threads per warp
  concurrentSignalWaitMultiWarpKernel<<<1, blockSize>>>(
      dw, targetRank, numSlots, isSignaler, results, warpsPerBlock);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// signal_all() Test
// =============================================================================

__global__ void signalAllKernel(
    DeviceWindow& dw,
    int signalerRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = dw.rank();

  if (myRank == signalerRank) {
    dw.signal_all(group, signalIdx, SignalOp::SIGNAL_ADD, 1);
  } else {
    dw.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
  }

  *result = 1;
}

void testSignalAll(
    DeviceWindow& dw,
    int signalerRank,
    int signalIdx,
    int* result) {
  signalAllKernel<<<1, 32>>>(dw, signalerRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// signal_all() + read_signal() Aggregate Test
// =============================================================================

__global__ void signalAllAggregateDistributedKernel(
    DeviceWindow dw,
    int signalIdx,
    uint64_t* result) {
  auto group = make_warp_group();

  // Every rank signals all peers with value 1
  dw.signal_all(group, signalIdx, SignalOp::SIGNAL_ADD, 1);

  // Wait until aggregate reaches nRanks-1 (all peers have signaled us)
  dw.wait_signal(
      group, signalIdx, CmpOp::CMP_GE, dw.num_peers(), Timeout(10000000000ULL));

  // Read aggregate (thread-level API)
  if (group.is_leader()) {
    *result = dw.read_signal(signalIdx);
  }
}

void testSignalAllAggregateDistributed(
    DeviceWindow& dw,
    int signalIdx,
    uint64_t* result) {
  signalAllAggregateDistributedKernel<<<1, 32>>>(dw, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal_from_all() Test
// =============================================================================

__global__ void waitSignalFromAllKernel(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = dw.rank();

  if (myRank == targetRank) {
    int nRanks = dw.n_ranks();
    dw.wait_signal(group, signalIdx, CmpOp::CMP_GE, nRanks - 1);
  } else {
    dw.signal_peer(group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
  }

  *result = 1;
}

void testWaitSignalFromAll(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalFromAllKernel<<<1, 32>>>(dw, targetRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Wait with CMP_EQ Test
// =============================================================================

__global__ void waitWithCmpEqKernel(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    dw.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_SET, expectedValue);
  } else {
    dw.wait_signal(group, signalIdx, CmpOp::CMP_EQ, expectedValue);
  }

  *result = 1;
}

void testWaitWithCmpEq(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result) {
  waitWithCmpEqKernel<<<1, 32>>>(
      dw, targetRank, signalIdx, expectedValue, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Monotonic Wait Values Test
// =============================================================================

__global__ void monotonicWaitValuesKernel(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  for (int i = 0; i < numIterations; ++i) {
    if (isSignaler) {
      dw.signal_peer(group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
    } else {
      dw.wait_signal(group, signalIdx, CmpOp::CMP_GE, i + 1);
    }
    group.sync();
  }

  *result = 1;
}

void testMonotonicWaitValues(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result) {
  monotonicWaitValuesKernel<<<1, 32>>>(
      dw, targetRank, signalIdx, numIterations, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// SIGNAL_SET Integration Test
// =============================================================================

__global__ void signalWithSetKernel(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    dw.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_SET, setValue);
  } else {
    dw.wait_signal(group, signalIdx, CmpOp::CMP_GE, setValue);
  }

  *result = 1;
}

void testSignalWithSet(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result) {
  signalWithSetKernel<<<1, 32>>>(
      dw, targetRank, signalIdx, setValue, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Monotonic Counters Test
// =============================================================================

__global__ void barrierMonotonicKernel(
    DeviceWindow& dw,
    int barrierIdx,
    int numPhases,
    int* result) {
  auto group = make_warp_group();

  for (int phase = 0; phase < numPhases; ++phase) {
    dw.barrier(group, barrierIdx);
  }

  if (threadIdx.x == 0) {
    *result = 1;
  }
}

void testBarrierMonotonic(
    DeviceWindow& dw,
    int barrierIdx,
    int numPhases,
    int* result) {
  barrierMonotonicKernel<<<1, 32>>>(dw, barrierIdx, numPhases, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Multi-Block Stress Test
// =============================================================================

__global__ void
barrierMultiBlockStressKernel(DeviceWindow& dw, int numSlots, int* results) {
  auto group = make_warp_group();

  uint32_t slotId = blockIdx.x % numSlots;

  dw.barrier(group, slotId);

  if (threadIdx.x == 0) {
    results[blockIdx.x] = 1;
  }
}

void testBarrierMultiBlockStress(
    DeviceWindow& dw,
    int numSlots,
    int* results,
    int numBlocks) {
  barrierMultiBlockStressKernel<<<numBlocks, 32>>>(dw, numSlots, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Barrier Peer Test (Two-Sided Barrier)
// =============================================================================

__global__ void barrierPeerKernel(
    DeviceWindow& dw,
    int targetRank,
    int barrierIdx,
    int* result) {
  auto group = make_warp_group();

  dw.barrier_peer(targetRank, group, barrierIdx);

  if (threadIdx.x == 0) {
    *result = 1;
  }
}

void testBarrierPeer(
    DeviceWindow& dw,
    int targetRank,
    int barrierIdx,
    int* result) {
  barrierPeerKernel<<<1, 32>>>(dw, targetRank, barrierIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal_from() Basic Test
// =============================================================================

__global__ void waitSignalFromPeerKernel(
    DeviceWindow& dw,
    int peerRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_warp_group();

  if (isSignaler) {
    dw.signal_peer(group, peerRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
  } else {
    dw.wait_signal_from(group, peerRank, signalIdx, CmpOp::CMP_GE, 1);
    uint64_t val = dw.read_signal_from(peerRank, signalIdx);
    if (val < 1) {
      *result = 0;
      return;
    }
  }

  *result = 1;
}

void testWaitSignalFromPeer(
    DeviceWindow& dw,
    int peerRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  waitSignalFromPeerKernel<<<1, 32>>>(
      dw, peerRank, signalIdx, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal_from() Multi-Peer Isolation Test
// =============================================================================

__global__ void waitSignalFromMultiPeerIsolationKernel(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = dw.rank();
  int nRanks = dw.n_ranks();

  if (myRank == targetRank) {
    for (int r = 0; r < nRanks; ++r) {
      if (r == myRank) {
        continue;
      }
      uint64_t expectedValue = static_cast<uint64_t>(r + 1);
      dw.wait_signal_from(group, r, signalIdx, CmpOp::CMP_GE, expectedValue);
      uint64_t val = dw.read_signal_from(r, signalIdx);
      if (val != expectedValue) {
        *result = 0;
        return;
      }
    }
  } else {
    uint64_t signalValue = static_cast<uint64_t>(myRank + 1);
    dw.signal_peer(
        group, targetRank, signalIdx, SignalOp::SIGNAL_SET, signalValue);
  }

  *result = 1;
}

void testWaitSignalFromMultiPeerIsolation(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalFromMultiPeerIsolationKernel<<<1, 32>>>(
      dw, targetRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// wait_signal() and wait_signal_from() Both Work Test
// =============================================================================

__global__ void waitSignalAndWaitSignalFromBothWorkKernel(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int* result) {
  auto group = make_warp_group();
  int myRank = dw.rank();
  int nRanks = dw.n_ranks();

  if (myRank == targetRank) {
    dw.wait_signal(group, signalIdx, CmpOp::CMP_GE, nRanks - 1);

    for (int r = 0; r < nRanks; ++r) {
      if (r == myRank) {
        continue;
      }
      dw.wait_signal_from(group, r, signalIdx, CmpOp::CMP_GE, 1);
    }
  } else {
    dw.signal_peer(group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
  }

  *result = 1;
}

void testWaitSignalAndWaitSignalFromBothWork(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int* result) {
  waitSignalAndWaitSignalFromBothWorkKernel<<<1, 32>>>(
      dw, targetRank, signalIdx, result);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Signal/Wait Test (BLOCK Scope - exercises fallback path)
// =============================================================================

__global__ void signalWaitBlockScopeKernel(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  auto group = make_block_group(); // Uses SyncScope::BLOCK

  if (isSignaler) {
    dw.signal_peer(group, targetRank, signalIdx, SignalOp::SIGNAL_ADD, 1);
    if (threadIdx.x == 0) {
      *result = 1;
    }
  } else {
    dw.wait_signal(group, signalIdx, CmpOp::CMP_GE, 1);
    if (threadIdx.x == 0) {
      *result = 1;
    }
  }
}

void testSignalWaitBlockScope(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result) {
  // Launch with 1 block of 128 threads (BLOCK scope = all 128 threads in block)
  signalWaitBlockScopeKernel<<<1, 128>>>(
      dw, targetRank, signalIdx, isSignaler, result);
  CUDACHECK_TEST(cudaGetLastError());
}

} // namespace comms::pipes::test
