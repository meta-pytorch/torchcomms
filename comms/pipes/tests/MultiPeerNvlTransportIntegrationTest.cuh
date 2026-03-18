// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/window/DeviceWindow.cuh"

namespace comms::pipes::test {

/**
 * Test kernel: Verify DeviceWindow accessors on device
 *
 * @param dw The DeviceWindow to test
 * @param results Output array: [0]=rank, [1]=nRanks, [2]=numPeers
 */
void testMultiPeerDeviceTransportAccessors(
    const DeviceWindow& dw,
    int* results);

/**
 * Test kernel: Signal from one rank to another and wait for it
 *
 * Uses inbox-model signaling: rank signals to peer's inbox, peer waits on own
 * inbox.
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param isSignaler If true, this rank signals; if false, this rank waits
 * @param result Output: 1 if successful, 0 if failed
 */
void testSignalWait(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result);

/**
 * Test kernel: Execute barrier across all ranks
 *
 * This tests that the barrier correctly synchronizes all ranks.
 *
 * @param dw The DeviceWindow to use
 * @param barrierIdx The barrier slot index to use
 * @param result Output: 1 if barrier completed successfully
 */
void testBarrier(DeviceWindow& dw, int barrierIdx, int* result);

/**
 * Test kernel: Send data from this rank to a single peer
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The destination rank
 * @param srcBuff Source buffer containing data to send
 * @param nbytes Number of bytes to send
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerSend(
    DeviceWindow& dw,
    int targetRank,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Receive data from a single peer to this rank
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The source rank
 * @param dstBuff Destination buffer for received data
 * @param nbytes Number of bytes to receive
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerRecv(
    DeviceWindow& dw,
    int targetRank,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Parallel send/recv to all peers using partition
 *
 * Uses partition_interleaved to split warps between send and recv work,
 * then further partitions across peers. This avoids deadlocks that can occur
 * when send and recv are done sequentially.
 *
 * @param dw The DeviceWindow to use
 * @param srcBuffs Array of source buffers, one per peer
 * @param dstBuffs Array of destination buffers, one per peer
 * @param nbytesPerPeer Number of bytes to transfer per peer
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testMultiPeerSendRecvAllPeers(
    DeviceWindow& dw,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Concurrent signal/barrier using multiple blocks
 *
 * Each block uses different signal/barrier slots concurrently
 * to verify no races or deadlocks.
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The target rank to signal/wait from
 * @param numSlots Number of slots to test concurrently
 * @param isSignaler If true, this rank signals; if false, waits
 * @param results Output array: results[blockIdx] = 1 if successful
 */
void testConcurrentSignalMultiBlock(
    DeviceWindow& dw,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int numBlocks);

/**
 * Test kernel: Verify transport type accessors
 *
 * Checks that get_peer_transport() and get_self_transport() return correct
 * transport types for self vs peer.
 *
 * @param dw The DeviceWindow to use
 * @param results Output array: [0]=numPeers, [1..nRanks]=transport types
 */
void testTransportTypes(const DeviceWindow& dw, int* results);

/**
 * Test kernel: Concurrent signal/wait from multiple warps within a block
 *
 * Each warp uses a different signal slot to verify thread-safety of
 * signal operations when multiple warps operate concurrently.
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The target rank to signal/wait from
 * @param numSlots Number of signal slots (should be >= warps per block)
 * @param isSignaler If true, this rank signals; if false, waits
 * @param results Output array: results[warpIdx] = 1 if successful
 * @param warpsPerBlock Number of warps per block
 */
void testConcurrentSignalWaitMultiWarp(
    DeviceWindow& dw,
    int targetRank,
    int numSlots,
    bool isSignaler,
    int* results,
    int warpsPerBlock);

/**
 * Test kernel: signal_all() signals all peers at once
 *
 * Tests that signal_all() correctly signals all peers (excluding self).
 * One rank signals, all other ranks wait for the signal.
 *
 * @param dw The DeviceWindow to use
 * @param signalerRank The rank that will call signal_all()
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testSignalAll(
    DeviceWindow& dw,
    int signalerRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: signal_all() + read_signal() aggregate across all ranks
 *
 * All ranks call signal_all() to signal every peer, then each rank
 * waits for aggregate and reads it via read_signal().
 * Verifies the aggregate equals nRanks-1 (one signal from each peer).
 *
 * @param dw The DeviceWindow to use
 * @param signalIdx The signal slot index to use
 * @param result Output: aggregate signal value read by this rank
 */
void testSignalAllAggregateDistributed(
    DeviceWindow& dw,
    int signalIdx,
    uint64_t* result);

/**
 * Test kernel: wait_signal_from_all() barrier-like synchronization
 *
 * Tests that wait_signal_from_all() correctly waits for signals from
 * all peers. All peers signal one rank, that rank waits for all.
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The rank that will call wait_signal_from_all()
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testWaitSignalFromAll(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: Wait with CMP_EQ comparison operation
 *
 * Tests exact equality comparison (vs CMP_GE which is more commonly used).
 * Signals with exact value, waits with CMP_EQ.
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param expectedValue The value to signal and wait for
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if successful
 */
void testWaitWithCmpEq(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    uint64_t expectedValue,
    bool isSignaler,
    int* result);

/**
 * Test kernel: Monotonically increasing wait values pattern
 *
 * Tests the recommended pattern of using monotonically increasing wait
 * values (signal 1, wait for 1, signal 1, wait for 2, etc.) across
 * multiple iterations.
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param numIterations Number of iterations to perform
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if all iterations successful
 */
void testMonotonicWaitValues(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int numIterations,
    bool isSignaler,
    int* result);

/**
 * Test kernel: SIGNAL_SET operation in multi-GPU context
 *
 * Tests that SIGNAL_SET correctly overwrites values in multi-GPU signaling.
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param setValue The value to SET
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if successful
 */
void testSignalWithSet(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    uint64_t setValue,
    bool isSignaler,
    int* result);

/**
 * Test kernel: Barrier with monotonic counters across multiple phases
 *
 * Tests that a single barrier slot works across multiple phases via
 * counter accumulation (no reset needed).
 *
 * @param dw The DeviceWindow to use
 * @param barrierIdx The barrier slot index to use
 * @param numPhases Number of barrier phases to execute
 * @param result Output: 1 if successful
 */
void testBarrierMonotonic(
    DeviceWindow& dw,
    int barrierIdx,
    int numPhases,
    int* result);

/**
 * Test kernel: Multi-block barrier stress test
 *
 * Each block performs a barrier synchronization using a different slot.
 * Tests concurrent barrier operations from multiple blocks.
 *
 * @param dw The DeviceWindow to use
 * @param numSlots Number of barrier slots to use
 * @param results Output array: results[blockIdx] = 1 if successful
 * @param numBlocks Number of blocks to launch
 */
void testBarrierMultiBlockStress(
    DeviceWindow& dw,
    int numSlots,
    int* results,
    int numBlocks);

/**
 * Test kernel: Two-sided barrier with a specific peer
 *
 * Tests barrier_peer() which synchronizes with a single peer rather than
 * all ranks. Both ranks must call barrier_peer() with each other's rank.
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The target rank to synchronize with
 * @param barrierIdx The barrier slot index to use
 * @param result Output: 1 if successful
 */
void testBarrierPeer(
    DeviceWindow& dw,
    int targetRank,
    int barrierIdx,
    int* result);

/**
 * Test kernel: Test the put() operation (offset-based)
 *
 * @param dw The DeviceWindow to use
 * @param targetRank Target rank
 * @param srcBuf Registered source buffer
 * @param nbytes Number of bytes to transfer
 * @param signalId Signal slot to use for completion notification
 * @param isWriter True if this rank writes, false if it waits
 * @param result Output: 1 if successful
 */
void testPutOperation(
    DeviceWindow& dw,
    int targetRank,
    const LocalBufferRegistration& srcBuf,
    std::size_t nbytes,
    int signalId,
    bool isWriter,
    int* result);

/**
 * Test kernel: wait_signal_from() basic per-peer signal/wait
 *
 * Rank 0 signals rank 1, rank 1 uses wait_signal_from(0, ...) to wait
 * for the specific peer's signal. Verifies read_signal_from() as well.
 *
 * @param dw The DeviceWindow to use
 * @param peerRank The peer rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if successful
 */
void testWaitSignalFromPeer(
    DeviceWindow& dw,
    int peerRank,
    int signalIdx,
    bool isSignaler,
    int* result);

/**
 * Test kernel: wait_signal_from() per-peer isolation
 *
 * All peers signal one target rank with different values using SIGNAL_SET.
 * Target calls wait_signal_from() for each peer individually, verifying
 * each peer's sub-slot has the correct independent value.
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The rank that all peers signal and that verifies isolation
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testWaitSignalFromMultiPeerIsolation(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: Both wait_signal() and wait_signal_from() work together
 *
 * All peers signal rank 0 with SIGNAL_ADD, 1. Rank 0 verifies:
 * - wait_signal(signal_id, CMP_GE, nRanks-1) succeeds (accumulated sum)
 * - wait_signal_from(peer, signal_id, CMP_GE, 1) succeeds for each peer
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The rank that waits for all signals
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testWaitSignalAndWaitSignalFromBothWork(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: Signal/Wait using BLOCK scope (exercises fallback path)
 *
 * Same as testSignalWait but uses make_block_group() instead of
 * make_warp_group() to exercise the non-WARP fallback code path in
 * DeviceWindowSignal::wait_signal().
 *
 * @param dw The DeviceWindow to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param isSignaler If true, this rank signals; if false, this rank waits
 * @param result Output: 1 if successful, 0 if failed
 */
void testSignalWaitBlockScope(
    DeviceWindow& dw,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result);

} // namespace comms::pipes::test
