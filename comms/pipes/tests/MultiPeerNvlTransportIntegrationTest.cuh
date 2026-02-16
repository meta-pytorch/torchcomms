// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/MultiPeerDeviceTransport.cuh"

namespace comms::pipes::test {

/**
 * Test kernel: Verify MultiPeerDeviceTransport accessors on device
 *
 * @param transport The MultiPeerDeviceTransport to test
 * @param results Output array: [0]=rank, [1]=nRanks, [2]=numPeers
 */
void testMultiPeerDeviceTransportAccessors(
    const MultiPeerDeviceTransport& transport,
    int* results);

/**
 * Test kernel: Signal from one rank to another and wait for it
 *
 * Uses inbox-model signaling: rank signals to peer's inbox, peer waits on own
 * inbox.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param isSignaler If true, this rank signals; if false, this rank waits
 * @param result Output: 1 if successful, 0 if failed
 */
void testSignalWait(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    bool isSignaler,
    int* result);

/**
 * Test kernel: Execute barrier across all ranks
 *
 * This tests that the barrier correctly synchronizes all ranks.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param barrierIdx The barrier slot index to use
 * @param result Output: 1 if barrier completed successfully
 */
void testBarrier(
    MultiPeerDeviceTransport& transport,
    int barrierIdx,
    int* result);

/**
 * Test kernel: Send data from this rank to a single peer
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The destination rank
 * @param srcBuff Source buffer containing data to send
 * @param nbytes Number of bytes to send
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerSend(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Receive data from a single peer to this rank
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The source rank
 * @param dstBuff Destination buffer for received data
 * @param nbytes Number of bytes to receive
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerRecv(
    MultiPeerDeviceTransport& transport,
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
 * @param transport The MultiPeerDeviceTransport to use
 * @param srcBuffs Array of source buffers, one per peer
 * @param dstBuffs Array of destination buffers, one per peer
 * @param nbytesPerPeer Number of bytes to transfer per peer
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testMultiPeerSendRecvAllPeers(
    MultiPeerDeviceTransport& transport,
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
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The target rank to signal/wait from
 * @param numSlots Number of slots to test concurrently
 * @param isSignaler If true, this rank signals; if false, waits
 * @param results Output array: results[blockIdx] = 1 if successful
 */
void testConcurrentSignalMultiBlock(
    MultiPeerDeviceTransport& transport,
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
 * @param transport The MultiPeerDeviceTransport to use
 * @param results Output array: [0]=numPeers, [1..nRanks]=transport types
 */
void testTransportTypes(
    const MultiPeerDeviceTransport& transport,
    int* results);

/**
 * Test kernel: Concurrent signal/wait from multiple warps within a block
 *
 * Each warp uses a different signal slot to verify thread-safety of
 * signal operations when multiple warps operate concurrently.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The target rank to signal/wait from
 * @param numSlots Number of signal slots (should be >= warps per block)
 * @param isSignaler If true, this rank signals; if false, waits
 * @param results Output array: results[warpIdx] = 1 if successful
 * @param warpsPerBlock Number of warps per block
 */
void testConcurrentSignalWaitMultiWarp(
    MultiPeerDeviceTransport& transport,
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
 * @param transport The MultiPeerDeviceTransport to use
 * @param signalerRank The rank that will call signal_all()
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testSignalAll(
    MultiPeerDeviceTransport& transport,
    int signalerRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: wait_signal_from_all() barrier-like synchronization
 *
 * Tests that wait_signal_from_all() correctly waits for signals from
 * all peers. All peers signal one rank, that rank waits for all.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The rank that will call wait_signal_from_all()
 * @param signalIdx The signal slot index to use
 * @param result Output: 1 if successful
 */
void testWaitSignalFromAll(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int signalIdx,
    int* result);

/**
 * Test kernel: Wait with CMP_EQ comparison operation
 *
 * Tests exact equality comparison (vs CMP_GE which is more commonly used).
 * Signals with exact value, waits with CMP_EQ.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param expectedValue The value to signal and wait for
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if successful
 */
void testWaitWithCmpEq(
    MultiPeerDeviceTransport& transport,
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
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param numIterations Number of iterations to perform
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if all iterations successful
 */
void testMonotonicWaitValues(
    MultiPeerDeviceTransport& transport,
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
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The target rank to signal/wait from
 * @param signalIdx The signal slot index to use
 * @param setValue The value to SET
 * @param isSignaler If true, this rank signals; if false, waits
 * @param result Output: 1 if successful
 */
void testSignalWithSet(
    MultiPeerDeviceTransport& transport,
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
 * @param transport The MultiPeerDeviceTransport to use
 * @param barrierIdx The barrier slot index to use
 * @param numPhases Number of barrier phases to execute
 * @param result Output: 1 if successful
 */
void testBarrierMonotonic(
    MultiPeerDeviceTransport& transport,
    int barrierIdx,
    int numPhases,
    int* result);

/**
 * Test kernel: Multi-block barrier stress test
 *
 * Each block performs a barrier synchronization using a different slot.
 * Tests concurrent barrier operations from multiple blocks.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param numSlots Number of barrier slots to use
 * @param results Output array: results[blockIdx] = 1 if successful
 * @param numBlocks Number of blocks to launch
 */
void testBarrierMultiBlockStress(
    MultiPeerDeviceTransport& transport,
    int numSlots,
    int* results,
    int numBlocks);

/**
 * Test kernel: Two-sided barrier with a specific peer
 *
 * Tests barrier_peer() which synchronizes with a single peer rather than
 * all ranks. Both ranks must call barrier_peer() with each other's rank.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The target rank to synchronize with
 * @param barrierIdx The barrier slot index to use
 * @param result Output: 1 if successful
 */
void testBarrierPeer(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    int barrierIdx,
    int* result);

/**
 * Test kernel: Test the put() operation
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank Target rank
 * @param remoteDst Remote destination buffer
 * @param localSrc Local source buffer
 * @param nbytes Number of bytes to transfer
 * @param signalId Signal slot to use for completion notification
 * @param isWriter True if this rank writes, false if it waits
 * @param result Output: 1 if successful
 */
void testPutOperation(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    void* remoteDst,
    const void* localSrc,
    std::size_t nbytes,
    int signalId,
    bool isWriter,
    int* result);

} // namespace comms::pipes::test
