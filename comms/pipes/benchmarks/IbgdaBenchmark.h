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
 * Uses separate put + signal operations, which is safe for adaptive routing.
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
 * Launch batched kernel: Multiple put_signal_non_adaptive+wait_local iterations
 *
 * Uses fused put_signal operation - faster but unsafe with adaptive routing.
 *
 * @param totalCycles Output: total GPU cycles for numIters operations
 */
void launchIbgdaPutSignalNonAdaptiveWaitLocalBatch(
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
