// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"

namespace comms::pipes {
// Forward declaration
class P2pIbgdaTransportDevice;
} // namespace comms::pipes

namespace comms::pipes::benchmark {

/**
 * Single-shot launchers for correctness verification.
 * Each launches exactly one put + signal + wait_local, no warmup, no loop.
 */
void launchIbgdaPutSignalSingle(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple put+wait_local iterations in a single kernel
 *
 * This avoids per-operation kernel launch overhead and uses GPU cycle counters
 * for accurate timing of raw RDMA operations.
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple put+signal+wait_local iterations
 *
 * Uses separate put + signal_remote_with_fence, which is safe for adaptive
 * routing.
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutSignalWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple signal-only iterations
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaSignalOnlyBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: put + wait_local (CQ poll)
 *
 * GPU thread polls NIC CQ at the work handle's WQE index for completion.
 * This is the CQ-based completion path for comparison against counter-based.
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutCqPollWaitBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: put + signal + counter via companion QP
 *
 * NIC does data write + remote signal atomic + companion QP WAIT +
 * local counter atomic. GPU thread spins on volatile local counter.
 * More NIC work but GPU polls local GPU memory (L2 cache).
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutSignalCounterBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

// =========================================================================
// Multi-peer kernels for counter fan-out validation
// =========================================================================

/**
 * Launch multi-peer CQ-poll kernel: put to each peer + wait_local on each QP
 *
 * GPU thread iterates over all peers, doing put + wait_local() on each QP
 * serially. Total wait = sum of all individual wait_local latencies (O(N) CQ
 * polls).
 *
 * @param transportsBase Base pointer to P2pIbgdaTransportDevice array (GPU mem)
 * @param transportStride Byte stride between consecutive transports
 * @param numPeers Number of peers
 * @param localBuf Source data buffer (same for all peers)
 * @param remoteDataBufs Device array of per-peer remote data buffers
 * @param nbytes Data size per peer
 * @param numIters Batch iterations
 * @param totalCycles Output: total GPU cycles
 */
void launchMultiPeerCqPollFanOutBatch(
    P2pIbgdaTransportDevice* transportsBase,
    std::size_t transportStride,
    int numPeers,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch multi-peer counter fan-out kernel: put+signal+counter to all peers,
 * single counter poll
 *
 * GPU thread fires put_signal_counter_remote() to all peers (each companion
 * QP atomically increments the SAME counter slot), then polls one counter
 * value until it reaches numPeers. Total wait ≈ max(peer latency) + loopback.
 *
 * @param transportsBase Base pointer to P2pIbgdaTransportDevice array (GPU mem)
 * @param transportStride Byte stride between consecutive transports
 * @param numPeers Number of peers
 * @param localBuf Source data buffer (same for all peers)
 * @param remoteDataBufs Device array of per-peer remote data buffers
 * @param nbytes Data size per peer
 * @param remoteSignalBufs Device array of per-peer remote signal buffers
 * @param signalId Signal slot index
 * @param localCounterBuf Local counter buffer (shared by all companion QPs)
 * @param counterId Counter slot index
 * @param numIters Batch iterations
 * @param totalCycles Output: total GPU cycles
 */
void launchMultiPeerCounterFanOutBatch(
    P2pIbgdaTransportDevice* transportsBase,
    std::size_t transportStride,
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

} // namespace comms::pipes::benchmark
