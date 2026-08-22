// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims::test {

using comms::prims::P2pNvlTransportDevice;

// Enum for specifying the thread group type
enum class GroupType {
  WARP, // 32-thread warp groups
  BLOCK // Full block groups (all threads in block)
};

void testTileSend(
    const P2pNvlTransportDevice& p2p,
    void* src_d,
    size_t nbytes,
    size_t maxSignalBytes,
    Timeout timeout,
    int numBlocks,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileRecv(
    const P2pNvlTransportDevice& p2p,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    Timeout timeout,
    int numBlocks,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileMultiCallSendRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallVariableSignalSendRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    bool waitForSecondCallSignal,
    int blockSize,
    Timeout timeout = Timeout(),
    cudaStream_t stream = nullptr);

void testTileTwoCallSendThenRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    Timeout timeout = Timeout(),
    cudaStream_t stream = nullptr);

void testTileMultiCallSendOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallSendOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileSendWaitsForWrappedSubstepAck(
    P2pNvlTransportDevice p2p,
    const char* sendData,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileForwardWaitsForWrappedSubstepAck(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    char* dst,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    int blockSize,
    cudaStream_t stream = nullptr);

void testPrepareTileStaging(
    P2pNvlTransportDevice p2p,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int sourceRank,
    int blockSize,
    cudaStream_t stream = nullptr);

void testPrepareTileTwoCallStaging(
    P2pNvlTransportDevice p2p,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int sourceRank,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileMultiCallRecvOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallRecvOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileMultiCallForward(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallForward(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallVariableSignalForward(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testCopyLocalStaging(
    P2pNvlTransportDevice p2p,
    void* dst,
    size_t nbytes,
    int blockSize,
    cudaStream_t stream = nullptr);

// Test put() - one-sided direct memory write to peer GPU and signal peer
// Unlike send()/recv(), put() writes directly to dst_d without staging buffers
void testPutWithSignal(
    P2pNvlTransportDevice* p2p,
    char* dst_d, // Destination on peer GPU (must be NVLink-accessible)
    const char* src_d, // Source on local GPU
    uint64_t signal_id,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

// Test wait() - one-sided wait for peer to write to dst_d and signal
void testWait(
    P2pNvlTransportDevice* p2p,
    CmpOp op,
    uint64_t signal_id,
    uint64_t expected,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

/*
 * Transition tallies for the resumable progress loop, written by the leader of
 * block 0. Lets a test assert on the state machine itself, not just on the
 * bytes that arrive.
 */
struct ProgressCounters {
  int sendWaiting;
  int sendProgressed;
  int recvWaiting;
  int recvProgressed;
  int sendCompleted;
  int recvCompleted;
};

/*
 * Drives one send and one recv through init_*_progress + progress_*_once,
 * cycling between them until both report Done. This is the shape the batched
 * MCCL send/recv kernel uses, so it exercises the API the way its real caller
 * does rather than in isolation.
 *
 * `counters` may be null; when set it receives block 0's transition tallies.
 */
void testProgressSendRecv(
    P2pNvlTransportDevice p2p,
    void* src_d,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    Timeout abort,
    ProgressCounters* counters,
    int numBlocks,
    int blockSize,
    cudaStream_t stream = nullptr);

/*
 * One-sided send with no peer draining, so the pipeline fills and the sender is
 * forced into Waiting. Runs at most `maxIterations` progress calls and is
 * expected NOT to complete; `counters` reports what was observed.
 */
void testProgressSendBackpressure(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    size_t maxSignalBytes,
    int maxIterations,
    Timeout abort,
    ProgressCounters* counters,
    int numBlocks,
    int blockSize,
    cudaStream_t stream = nullptr);

/*
 * Aborts a stalled send, then immediately re-inits the same channel in the same
 * kernel. Pins the invariant that justifies clearing `stage` on the abort path:
 * the cleanup must be visible to every thread in the group before the next
 * init reads it, with no kernel boundary in between. `counters->sendCompleted`
 * reports whether the abort concluded; reaching the end without trapping or
 * hanging is the rest of the assertion.
 */
void testProgressAbortThenReinit(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    int maxIterations,
    Timeout abort,
    ProgressCounters* counters,
    int numBlocks,
    int blockSize,
    cudaStream_t stream = nullptr);

/*
 * Two sequential progress operations on the same channel, to cover the cursor
 * reservation and the Active -> Idle reset between them. The second call must
 * resume from where the first left the cursor, not from zero.
 */
void testProgressTwoCallSendRecv(
    P2pNvlTransportDevice p2p,
    void* src_d,
    void* dst_d,
    size_t firstBytes,
    size_t secondBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    Timeout abort,
    int numBlocks,
    int blockSize,
    cudaStream_t stream = nullptr);

/*
 * Receiver-side counterpart of testProgressSendBackpressure: a recv with no
 * sender. DATA_READY never advances, so the poll must keep reporting Waiting
 * and must not signal SLOT_FREE credit for a chunk it never received.
 */
void testProgressRecvBackpressure(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    int maxIterations,
    Timeout abort,
    ProgressCounters* counters,
    int numBlocks,
    int blockSize,
    cudaStream_t stream = nullptr);

/*
 * Reads this rank's SLOT_FREE counter for `channel` into `out`. Lets a test
 * assert the peer never credited a chunk it did not consume, which is the
 * invariant the recv abort path protects.
 */
void testReadSlotFreeCounter(
    P2pNvlTransportDevice p2p,
    int channel,
    unsigned long long* out,
    cudaStream_t stream = nullptr);

} // namespace comms::prims::test
