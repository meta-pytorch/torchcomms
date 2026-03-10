// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/window/DeviceWindowBarrier.cuh"
#include "comms/pipes/window/DeviceWindowMemory.cuh"
#include "comms/pipes/window/DeviceWindowSignal.cuh"

namespace comms::pipes::test {

/**
 * TestDeviceBundle - Temporary test-local device-side bundle
 *
 * Bundles MultiPeerDeviceHandle + DeviceWindowSignal + DeviceWindowBarrier
 * into a single struct for passing to test kernels. This replaces the
 * deleted MultiPeerDeviceTransport and will be replaced by DeviceWindow
 * in the next diff in the stack.
 *
 * All method signatures match the old MultiPeerDeviceTransport API so
 * kernel code changes are minimal.
 */
struct TestDeviceBundle {
  MultiPeerDeviceHandle handle;
  DeviceWindowSignal signal;
  DeviceWindowBarrier barrier;

#ifdef __CUDACC__
  __host__ __device__ int rank() const {
    return handle.myRank;
  }

  __host__ __device__ int n_ranks() const {
    return handle.nRanks;
  }

  __host__ __device__ int num_peers() const {
    return handle.nRanks - 1;
  }

  __host__ __device__ int rank_to_peer_index(int rank) const {
    return (rank < handle.myRank) ? rank : (rank - 1);
  }

  __device__ __forceinline__ MultiPeerDeviceHandle& get_handle() {
    return handle;
  }

  __device__ __forceinline__ const MultiPeerDeviceHandle& get_handle() const {
    return handle;
  }
#endif
};

/**
 * Test kernel: Verify TestDeviceBundle accessors on device
 *
 * @param bundle The TestDeviceBundle to test
 * @param results Output array: [0]=rank, [1]=nRanks, [2]=numPeers
 */
void testMultiPeerDeviceTransportAccessors(
    const TestDeviceBundle& bundle,
    int* results);

/**
 * Test kernel: Signal from one rank to another and wait for it
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param isSignaler If true, this rank signals; if false, this rank waits
 * @param result Output: 1 if successful, 0 if failed
 */
void testSignalWait(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result);

/**
 * Test kernel: Execute barrier across all ranks
 *
 * @param bundle The TestDeviceBundle to use
 * @param barrierIdx The barrier slot index to use
 * @param result Output: 1 if barrier completed successfully
 */
void testBarrier(TestDeviceBundle& bundle, int barrierIdx, int* result);

/**
 * Test kernel: Send data from this rank to a single peer
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The destination rank
 * @param srcBuff Source buffer containing data to send
 * @param nbytes Number of bytes to send
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerSend(
    TestDeviceBundle& bundle,
    int targetRank,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Receive data from a single peer to this rank
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The source rank
 * @param dstBuff Destination buffer for received data
 * @param nbytes Number of bytes to receive
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerRecv(
    TestDeviceBundle& bundle,
    int targetRank,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Parallel send/recv to all peers using partition
 *
 * @param bundle The TestDeviceBundle to use
 * @param srcBuffs Array of source buffers, one per peer
 * @param dstBuffs Array of destination buffers, one per peer
 * @param nbytesPerPeer Number of bytes to transfer per peer
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testMultiPeerSendRecvAllPeers(
    TestDeviceBundle& bundle,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Concurrent signal/barrier using multiple blocks
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The target rank to signal/wait from
 * @param numSlots Number of slots to test concurrently
 * @param isSignaler If true, this rank signals; if false, waits
 * @param results Output array: results[blockIdx] = 1 if successful
 */
void testConcurrentSignalMultiBlock(
    TestDeviceBundle& bundle,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int numBlocks);

/**
 * Test kernel: Verify transport type accessors
 *
 * @param bundle The TestDeviceBundle to use
 * @param results Output array: [0]=numPeers, [1..nRanks]=transport types
 */
void testTransportTypes(const TestDeviceBundle& bundle, int* results);

/**
 * Test kernel: Concurrent signal/wait from multiple warps within a block
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The target rank to signal/wait from
 * @param numSlots Number of signal slots (should be >= warps per block)
 * @param isSignaler If true, this rank signals; if false, waits
 * @param results Output array: results[warpIdx] = 1 if successful
 * @param warpsPerBlock Number of warps per block
 */
void testConcurrentSignalWaitMultiWarp(
    TestDeviceBundle& bundle,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int warpsPerBlock);

/**
 * Test kernel: signal_all() signals all peers at once
 *
 * @param bundle The TestDeviceBundle to use
 * @param signalerRank The rank that will call signal_all()
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testSignalAll(
    TestDeviceBundle& bundle,
    int signalerRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: signal_all() + read_signal() aggregate across all ranks
 *
 * @param bundle The TestDeviceBundle to use
 * @param signalIdx The signal slot index to use
 * @param result Output: aggregate signal value read by this rank
 */
void testSignalAllAggregateDistributed(
    TestDeviceBundle& bundle,
    int signalIdx,
    uint64_t* result);

/**
 * Test kernel: wait_signal_from_all() barrier-like synchronization
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The rank that will call wait_signal_from_all()
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testWaitSignalFromAll(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: Wait with CMP_EQ comparison operation
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param expectedValue The value to signal and wait for
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if successful
 */
void testWaitWithCmpEq(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result);

/**
 * Test kernel: Monotonically increasing wait values pattern
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param numIterations Number of iterations to perform
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if all iterations successful
 */
void testMonotonicWaitValues(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result);

/**
 * Test kernel: SIGNAL_SET operation in multi-GPU context
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param setValue The value to SET
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if successful
 */
void testSignalWithSet(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result);

/**
 * Test kernel: Barrier with monotonic counters across multiple phases
 *
 * @param bundle The TestDeviceBundle to use
 * @param barrierIdx The barrier slot index to use
 * @param numPhases Number of barrier phases to execute
 * @param result Output: 1 if successful
 */
void testBarrierMonotonic(
    TestDeviceBundle& bundle,
    int barrierIdx,
    int numPhases,
    int* result);

/**
 * Test kernel: Multi-block barrier stress test
 *
 * @param bundle The TestDeviceBundle to use
 * @param numSlots Number of barrier slots to use
 * @param results Output array: results[blockIdx] = 1 if successful
 * @param numBlocks Number of blocks to launch
 */
void testBarrierMultiBlockStress(
    TestDeviceBundle& bundle,
    int numSlots,
    int* results,
    int numBlocks);

/**
 * Test kernel: Two-sided barrier with a specific peer
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The target rank to synchronize with
 * @param barrierIdx The barrier slot index to use
 * @param result Output: 1 if successful
 */
void testBarrierPeer(
    TestDeviceBundle& bundle,
    int targetRank,
    int barrierIdx,
    int* result);

/**
 * Test kernel: Test the put() operation
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank Target rank
 * @param remoteDst Remote destination buffer
 * @param localSrc Local source buffer
 * @param nbytes Number of bytes to transfer
 * @param signalId Signal slot to use for completion notification
 * @param isWriter True if this rank writes, false if it waits
 * @param result Output: 1 if successful
 */
void testPutOperation(
    TestDeviceBundle& bundle,
    int targetRank,
    void* remoteDst,
    const void* localSrc,
    std::size_t nbytes,
    int signalId,
    bool isWriter,
    int* result);

/**
 * Test kernel: wait_signal_from() basic per-peer signal/wait
 *
 * @param bundle The TestDeviceBundle to use
 * @param peerRank The peer rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if successful
 */
void testWaitSignalFromPeer(
    TestDeviceBundle& bundle,
    int peerRank,
    int signalIdx,
    bool isSignaler,
    int* result);

/**
 * Test kernel: wait_signal_from() per-peer isolation
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The rank that all peers signal and that verifies isolation
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testWaitSignalFromMultiPeerIsolation(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: Both wait_signal() and wait_signal_from() work together
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The rank that waits for all signals
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testWaitSignalAndWaitSignalFromBothWork(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: Signal/Wait using BLOCK scope (exercises fallback path)
 *
 * @param bundle The TestDeviceBundle to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param isSignaler If true, this rank signals; if false, this rank waits
 * @param result Output: 1 if successful, 0 if failed
 */
void testSignalWaitBlockScope(
    TestDeviceBundle& bundle,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result);

} // namespace comms::pipes::test
