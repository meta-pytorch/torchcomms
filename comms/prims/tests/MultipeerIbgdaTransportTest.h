// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {

// Forward declaration - full definition in P2pIbTransportDevice.cuh
struct P2pIbTransportDevice;
// Forward declaration - full definition in P2pIbgdaTransportDevice.cuh
class P2pIbgdaTransportDevice;

} // namespace comms::prims

namespace comms::prims::test {

/**
 * Test kernel: Put data + signal remote (thread-scope, slot-index)
 *
 * Uses thread-scope put() with slot-index signal to write data and signal
 * completion via the transport's owned signal buffer.
 */
void testPutAndSignal(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Explicit cooperative put + signal (warp group, slot-index)
 */
void testPutAndSignalGroup(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multi-warp presharded group put + signal (slot-index)
 */
void testPutAndSignalGroupMultiWarp(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Block-scope presharded group put + signal (slot-index)
 */
void testPutAndSignalGroupBlock(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for signal via slot-index on transport's local inbox
 */
void testWaitSignal(
    P2pIbTransportDevice transport,
    int signalId,
    uint64_t expectedSignal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multiple put + signal operations in sequence (slot-index)
 */
void testMultiplePutAndSignal(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    int signalId,
    int numPuts,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: post a burst larger than the SQ, then flush once. Slot reuse
 * must therefore make progress through reserve_wq_slots' internal CQ poll.
 */
void testBurstPutAndFlush(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    int numPuts,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Send signal only (no data, slot-index)
 */
void testSignalOnly(
    P2pIbTransportDevice deviceTransportPtr,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Put data without signal
 */
void testPutOnly(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Return whether resumable send/recv progress kernels are available.
 */
bool supportsProgressSendRecv();

struct RegisteredSendObservation {
  uint64_t waitingCount{0};
  uint64_t progressedCount{0};
  uint64_t postedCount{0};
  uint64_t drainedCount{0};
  uint64_t abortedCount{0};
  /// Iterations the drain loop actually took. Only written by the
  /// drain-abort test; zero elsewhere.
  uint64_t drainIterations{0};

  template <typename Status>
  IBGDA_HOST_DEVICE void record(Status status) {
    switch (status) {
      case Status::Waiting:
        ++waitingCount;
        break;
      case Status::Progressed:
        ++progressedCount;
        break;
      case Status::Posted:
        ++postedCount;
        break;
      case Status::Drained:
        ++drainedCount;
        break;
      case Status::Aborted:
        ++abortedCount;
        break;
    }
  }
};

/**
 * Test kernel: snapshot send/recv pipeline geometry through the unified IB
 * transport wrapper. Output: pipelineDepth, pipelineWindow, pipelineChunk.
 */
void testPipelineGeometry(
    P2pIbTransportDevice transport,
    uint64_t* output,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Blocking pipelined send or recv.
 */
void testSendRecv(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Blocking pipelined send or recv driving the variable-size
 * `AnsCompress` CopyOp — the compressed transport path added in D111967119.
 * `maxSignalBytes == 0` exercises the transport's 0-sentinel (which derives a
 * trap-safe chunk size via CopyOp::max_safe_chunk_size_for_slot()); a non-zero
 * value (e.g. 256 KiB) exercises the explicit signaled-chunk-size path.
 * `blockSize` must be 128, 256, or 512 (NumWarps 4/8/16). Defined in the
 * separately device-linked MultipeerIbgdaTransportAnsTest.cu.
 */
void testSendRecvAns(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: two sequential bidirectional blocking send/recv calls.
 */
void testTwoCallSendThenRecv(
    P2pIbTransportDevice transport,
    const void* sendBuffer,
    void* recvBuffer,
    std::size_t firstBytes,
    std::size_t secondBytes,
    std::size_t maxSignalBytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Warp-proxy send or recv with queue-full observation.
 */
// The device budget comes from `testAbortDevice()` inside the launcher, as with
// the other kernels here; there is no caller-supplied cycle count since the
// standalone Prims `Timeout` was replaced by the communicator abort handle.
void testWarpProxySendRecv(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    uint32_t queueDepth,
    uint64_t* queueFullCount);

/**
 * Test kernel: warp-proxy send against a peer that never runs its own proxy.
 *
 * Launches asynchronously and does NOT synchronize -- the caller is expected to
 * abort the supplied handle while the kernel is parked, then synchronize. The
 * abort is caller-owned rather than `testAbortDevice()` on purpose: that helper
 * is a `TRAP`-mode watchdog, so it can only end a stuck proxy by taking the
 * CUDA context down, which cannot distinguish "the service loop honoured the
 * abort" from "the watchdog fired".
 */
void launchWarpProxyStalledSend(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    uint32_t queueDepth,
    comms::fault_tolerance::AbortDevice abort);

/**
 * Test kernel: Resumable pipelined send or recv progress loop.
 */
void testProgressSendRecv(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize,
    uint64_t* waitingCount = nullptr);

/**
 * Test kernel: initialize transport-owned send/recv progress state and report
 * the reserved byte cursors.
 */
void testProgressReservations(
    P2pIbgdaTransportDevice* transport,
    int64_t* output,
    std::size_t sendBytes,
    std::size_t recvBytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: registered-source send progress or the matching staged recv.
 */
void testRegisteredSendRecv(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& source,
    void* recvBuffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize,
    RegisteredSendObservation* observation = nullptr,
    bool blocking = false,
    bool overwriteAfterDrain = false,
    uint8_t overwriteValue = 0,
    bool zeroByteAfterPosted = false);

/**
 * Test kernel: registered A, staged B, registered C on one send cursor.
 */
void testMixedRegisteredAndStagedSendRecv(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& sendBuffer,
    void* recvBuffer,
    std::size_t firstBytes,
    std::size_t secondBytes,
    std::size_t thirdBytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize);

/** Fill or verify a byte range in this channel's transport staging. */
void testFillTransportStaging(
    P2pIbgdaTransportDevice* transport,
    bool sendStaging,
    std::size_t offset,
    std::size_t nbytes,
    uint8_t value,
    int numBlocks,
    int blockSize);

void testVerifyTransportStaging(
    P2pIbgdaTransportDevice* transport,
    bool sendStaging,
    std::size_t offset,
    std::size_t nbytes,
    uint8_t expected,
    int* errorCount,
    int numBlocks,
    int blockSize);

/**
 * Fill a device buffer with a pattern based on index
 */
void fillBufferWithPattern(
    void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int numBlocks,
    int blockSize);

/**
 * Verify a device buffer matches expected pattern
 */
void verifyBufferPattern(
    const void* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for ready signal, then put data with signal (slot-index)
 */
void testWaitReadyThenPutAndSignal(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int readySignalId,
    uint64_t readySignalVal,
    int dataSignalId,
    uint64_t dataSignalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Bidirectional put and wait in single kernel (slot-index)
 */
void testBidirectionalPutAndWait(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int sendSignalId,
    uint64_t sendSignalVal,
    int recvSignalId,
    uint64_t recvSignalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: All-to-all send phase (slot-index)
 */
void testAllToAll(
    P2pIbTransportDevice* peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: All-to-all wait phase (slot-index)
 */
void testAllToAllWait(
    P2pIbTransportDevice* peerTransports,
    int numPeers,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Put data + signal remote + counter (slot-index)
 */
void testPutSignalCounter(
    P2pIbTransportDevice deviceTransportPtr,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal,
    int numBlocks,
    int blockSize,
    int numIterations = 1);

/**
 * Test kernel: Wait for local counter to reach expected value (slot-index)
 */
void testWaitCounter(
    P2pIbTransportDevice transport,
    int counterId,
    uint64_t expectedVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multi-QP put + signal with per-block QP selection
 *
 * Each block selects its QP via blockIdx.x % numQps, puts its chunk
 * of totalBytes, then signals. Tests that independent QPs work correctly
 * when blocks use different QPs.
 *
 * @param transport Peer transport handle
 * @param numQps Number of QPs configured on the transport
 * @param localBuf Local source buffer
 * @param remoteBuf Remote destination buffer
 * @param totalBytes Total bytes (split across blocks)
 * @param signalId Signal slot index
 * @param signalVal Signal value per block
 * @param numBlocks Grid dimension
 * @param blockSize Block dimension
 */
void testMultiQpPutAndSignal(
    P2pIbTransportDevice transport,
    int numQps,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t totalBytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: put + flush against a caller-supplied abort handle.
 *
 * The fault injector for the completion-error path: give it a `remoteBuf` whose
 * rkey the peer rejects and the NIC produces an error CQE, which `flush()`
 * observes inside `wait_local_on_qp`. Unlike a dead peer -- where the op simply
 * never completes until IB retry exhaustion, roughly a minute out -- a remote
 * access error is terminal and reported immediately.
 *
 * `abort` is by value: `AbortDevice` is a handle over shared state, so the copy
 * the kernel mutates and the host's `Abort` observe the same latch.
 */
/**
 * Test kernel: a real registered send whose completions never land, followed by
 * the production-shaped drain loop.
 *
 * `poisonedRemote` is an exchanged peer buffer with its rkey corrupted. The
 * kernel puts to it several times -- deliberately without flushing -- to drive
 * the channel's QP lanes into error state, so the registered send that follows
 * has completions the NIC will never report successfully. The drain then loops
 * `while (status != Drained)` exactly as `ReduceScatterDirectIbV2.cu` does.
 *
 * The channel layout's own staging rkeys are NOT usable for this: they are not
 * populated at the point the kernel runs, and indexing them traps.
 *
 * `drainIterationCap` bounds that loop so a regression reports a failure
 * instead of hanging until the harness timeout; `observation->drainIterations`
 * records what it actually took, which is the assertion that matters.
 */
void testRegisteredSendDrainWithAbort(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& source,
    const IbgdaRemoteBuffer& poisonedRemote,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    RegisteredSendObservation* observation,
    uint64_t drainIterationCap,
    comms::fault_tolerance::AbortDevice abort,
    int numBlocks,
    int blockSize);

void testPutAndFlushWithAbort(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& poisonedRemoteBuf,
    const IbgdaRemoteBuffer& validRemoteBuf,
    std::size_t nbytes,
    comms::fault_tolerance::AbortDevice abort,
    int numBlocks,
    int blockSize);

} // namespace comms::prims::test
