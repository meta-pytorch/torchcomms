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
 * Each launches exactly one put + signal + counter, GPU spins on the local
 * counter slot. No warmup, no loop.
 */
void launchIbgdaPutSignalSingle(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple put + counter iterations
 *
 * Counter-only put (no peer signal): companion-QP loopback atomically
 * increments the local counter when the put completes at the NIC. GPU spins
 * on the local counter slot.
 *
 * Avoids per-operation kernel launch overhead and uses GPU cycle counters
 * for accurate timing of raw RDMA operations.
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple put + signal + counter iterations
 *
 * Companion-QP loopback atomically increments the local counter when the
 * put+signal completes at the NIC. GPU spins on the local counter slot.
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
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

/**
 * Launch batched kernel: Multiple signal-only iterations
 *
 * Signal-only path uses fence() for completion (no counter primitive applies
 * to signal-only ops).
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

// =========================================================================
// Multi-peer kernels for counter fan-out validation
// =========================================================================

/**
 * Launch multi-peer serial counter fan-out: put+signal+counter to each peer
 * with a per-peer counter slot, then wait_counter on each slot serially.
 *
 * O(N) wait_counter calls (one per peer, each peer's companion QP increments
 * its own counter slot). This is the per-peer baseline for comparison against
 * the shared-counter fan-out path (launchMultiPeerCounterFanOutBatch), which
 * collapses the N waits into a single wait on a shared slot.
 *
 * @param transportsBase Base pointer to P2pIbgdaTransportDevice array (GPU mem)
 * @param transportStride Byte stride between consecutive transports
 * @param numPeers Number of peers
 * @param localBuf Source data buffer (same for all peers)
 * @param remoteDataBufs Device array of per-peer remote data buffers
 * @param nbytes Data size per peer
 * @param remoteSignalBufs Device array of per-peer remote signal buffers
 * @param signalId Signal slot index
 * @param localCounterBuf Local counter buffer with at least numPeers slots;
 *                        slot p is used by peer p's companion QP
 * @param numIters Batch iterations
 * @param totalCycles Output: total GPU cycles
 */
void launchMultiPeerSerialCounterFanOutBatch(
    P2pIbgdaTransportDevice* transportsBase,
    std::size_t transportStride,
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
 * GPU thread fires put() with signal+counter to all peers (each companion
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
