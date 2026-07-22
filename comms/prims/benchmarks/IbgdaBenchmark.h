// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims::benchmark {

inline constexpr int kIbgdaCounterWarmupIters = 10;

// All launchers take the backend-agnostic P2pIbTransportDevice handle (by value
// for single-peer, as a contiguous device array for multi-peer) so the same
// benchmark drives either IBGDA (GPU-initiated) or IBRC (CPU-proxy). The handle
// dispatches each device call on its embedded backend tag.

/**
 * Single-shot launchers for correctness verification.
 * Each launches exactly one put + signal + counter, GPU spins on the local
 * counter slot. No warmup, no loop.
 */
void launchIbgdaPutSignalSingle(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    int counterId,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple put + counter iterations
 *
 * Counter-only put (no peer signal): the put completion increments the local
 * counter (IBGDA companion-QP loopback; IBRC CPU proxy). GPU spins on the local
 * counter slot.
 *
 * Avoids per-operation kernel launch overhead and uses GPU cycle counters
 * for accurate timing of raw RDMA operations.
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutWaitCounterBatch(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/** Launch batched raw puts, waiting on each returned local-completion ticket.
 */
void launchIbgdaPutWaitLocalBatch(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple put + flush iterations
 *
 * Measures raw RDMA write completion through flush(), matching the NCCL GIN
 * put latency benchmark shape.
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutFlushBatch(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: one thread per block uses the thread-scope put +
 * flush API on a block-private buffer slice.
 *
 * This exercises the no-ThreadGroup API from multiple physical blocks, so it
 * catches regressions where thread-scope wrappers accidentally route every
 * block through block 0's transport state.
 *
 * @param blockCycles Output: one cycle count per launched block
 */
void launchIbgdaThreadScopeMultiBlockPutFlushBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytesPerBlock,
    int numBlocks,
    int numIters,
    unsigned long long* blockCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple put + signal + counter iterations
 *
 * The put+signal completion increments the local counter (IBGDA companion-QP
 * loopback; IBRC CPU proxy). GPU spins on the local counter slot.
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutSignalWaitCounterBatch(
    P2pIbTransportDevice transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple signal-only iterations
 *
 * Signal-only path uses flush() for completion (no counter primitive applies
 * to signal-only ops).
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaSignalOnlyBatch(
    P2pIbTransportDevice transport,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

// =========================================================================
// Multi-peer kernels for counter fan-out validation
// =========================================================================

/**
 * Launch multi-peer serial counter fan-out: put+signal+counter to each peer
 * with a per-peer counter slot, then wait_counter on each slot serially.
 *
 * O(N) wait_counter calls (one per peer, each peer's completion increments its
 * own counter slot). This is the per-peer baseline for comparison against the
 * shared-counter fan-out path (launchMultiPeerCounterFanOutBatch), which
 * collapses the N waits into a single wait on a shared slot.
 *
 * @param transports Device array of per-peer P2pIbTransportDevice handles
 *                   (index p == peer p)
 * @param numPeers Number of peers
 * @param localBuf Source data buffer (same for all peers)
 * @param remoteDataBufs Device array of per-peer remote data buffers
 * @param nbytes Data size per peer
 * @param remoteSignalBufs Device array of per-peer remote signal buffers
 * @param signalId Signal slot index
 * @param localCounterBuf Local counter buffer with at least numPeers slots;
 *                        slot p is used by peer p's completion
 * @param numIters Batch iterations
 * @param totalCycles Output: total GPU cycles
 */
void launchMultiPeerSerialCounterFanOutBatch(
    const P2pIbTransportDevice* transports,
    int numPeers,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    const IbgdaLocalBuffer& localCounterBuf,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch multi-peer counter fan-out kernel: put+signal+counter to all peers,
 * single counter poll
 *
 * GPU thread fires put() with signal+counter to all peers (each peer's
 * completion increments the SAME counter slot), then polls one counter value
 * until it reaches numPeers. Total wait ≈ max(peer latency) + loopback.
 *
 * @param transports Device array of per-peer P2pIbTransportDevice handles
 *                   (index p == peer p)
 * @param numPeers Number of peers
 * @param localBuf Source data buffer (same for all peers)
 * @param remoteDataBufs Device array of per-peer remote data buffers
 * @param nbytes Data size per peer
 * @param remoteSignalBufs Device array of per-peer remote signal buffers
 * @param signalId Signal slot index
 * @param localCounterBuf Local counter buffer (shared by all peers)
 * @param counterId Counter slot index
 * @param numIters Batch iterations
 * @param totalCycles Output: total GPU cycles
 */
void launchMultiPeerCounterFanOutBatch(
    const P2pIbTransportDevice* transports,
    int numPeers,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

} // namespace comms::prims::benchmark
