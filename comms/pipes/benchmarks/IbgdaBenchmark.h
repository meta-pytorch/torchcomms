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
 * Launch kernel: Put with signal (non-adaptive routing version)
 * Uses fused put_signal operation - faster but unsafe with adaptive routing.
 */
void launchIbgdaPutSignalNonAdaptiveWaitLocal(
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

} // namespace comms::pipes::benchmark
