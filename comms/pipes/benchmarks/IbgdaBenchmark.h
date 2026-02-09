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
 * Launch kernel: Put with signal, then wait for local completion
 */
void launchIbgdaPutSignalWaitLocal(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream);

/**
 * Launch kernel: Wait for signal from remote
 */
void launchIbgdaWaitSignal(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    IbgdaCmpOp cmpOp,
    uint64_t expectedSignal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream);

/**
 * Launch kernel: Signal only (no data transfer)
 */
void launchIbgdaSignalOnly(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream);

/**
 * Launch kernel: Reset signal buffer
 */
void launchIbgdaResetSignal(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    cudaStream_t stream);

/**
 * Launch kernel: Put only (no signal), then wait for local completion
 */
void launchIbgdaPutWaitLocal(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numBlocks,
    int numThreads,
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
 * Launch batched kernel: Multiple put_signal+wait_local iterations
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutSignalWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
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
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream);

} // namespace comms::pipes::benchmark
