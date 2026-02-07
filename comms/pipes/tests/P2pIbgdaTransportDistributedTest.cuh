// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"

namespace comms::pipes::tests {

// =============================================================================
// CUDA Kernels for Distributed P2pIbgdaTransport Tests
// =============================================================================

/**
 * Kernel to execute put_signal operation.
 * Single-threaded kernel for simplicity.
 */
__global__ void putSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal);

/**
 * Kernel to execute put_signal_non_adaptive operation.
 * Single-threaded kernel for simplicity.
 */
__global__ void putSignalNonAdaptiveKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal);

/**
 * Kernel to wait for signal and set success flag.
 */
__global__ void waitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t expectedSignal,
    bool* success);

/**
 * Kernel to send signal only (no data transfer).
 */
__global__ void signalOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t signalVal);

/**
 * Kernel to verify data buffer contents match expected pattern.
 */
__global__ void verifyDataKernel(
    void* data,
    std::size_t nbytes,
    uint8_t expectedPattern,
    bool* success);

/**
 * Kernel to fill data buffer with a pattern.
 */
__global__ void fillDataKernel(void* data, std::size_t nbytes, uint8_t pattern);

} // namespace comms::pipes::tests
