// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/common/DeviceConstants.cuh"
#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::benchmark {

// Use SyncScope from ThreadGroup.cuh for thread group type selection

// =============================================================================
// Barrier Benchmark Kernel
// =============================================================================

/**
 * N-way barrier benchmark kernel.
 *
 * Each thread group executes barrier() in a loop.
 * Uses unique slot per group to avoid contention.
 *
 * Half-duplex measurement: all ranks synchronize together.
 */
template <SyncScope G>
__global__ void multiPeerBarrierKernel(
    MultiPeerDeviceTransport transport,
    int nSteps);

// =============================================================================
// Signal Ping-Pong Benchmark Kernel
// =============================================================================

/**
 * Signal ping-pong benchmark kernel (2 ranks only).
 *
 * Rank 0 and Rank 1 alternate signaling each other.
 * Uses SIGNAL_ADD with cumulative wait values for reusability.
 *
 * Half-duplex measurement: one signal in flight at a time.
 */
template <SyncScope S>
__global__ void multiPeerSignalPingPongKernel(
    MultiPeerDeviceTransport transport,
    int targetRank,
    int nSteps);

// =============================================================================
// Signal-All Benchmark Kernel
// =============================================================================

/**
 * Signal-all benchmark kernel.
 *
 * Each rank signals all peers, then waits for all arrivals.
 * Uses SIGNAL_ADD with cumulative wait values.
 */
template <SyncScope S>
__global__ void multiPeerSignalAllKernel(
    MultiPeerDeviceTransport transport,
    int nSteps);

// =============================================================================
// Put Ping-Pong Benchmark Kernel
// =============================================================================

/**
 * Put ping-pong benchmark kernel (2 ranks only).
 *
 * Rank 0 and Rank 1 alternate put() + signal_peer() operations.
 * Measures NVLink write bandwidth for small messages combined with
 * signal latency.
 *
 * Half-duplex measurement: one put in flight at a time.
 *
 * @param transport MultiPeerDeviceTransport for NVLink operations
 * @param targetRank The target rank to communicate with
 * @param localSrc Local source buffer to read from
 * @param remoteDst Remote destination buffer to write to (IPC-mapped)
 * @param nbytes Number of bytes to transfer per put
 * @param nSteps Number of ping-pong iterations
 */
template <SyncScope S>
__global__ void multiPeerPutPingPongKernel(
    MultiPeerDeviceTransport transport,
    int targetRank,
    const void* localSrc,
    void* remoteDst,
    std::size_t nbytes,
    int nSteps);

// =============================================================================
// Put+Signal Ping-Pong Benchmark Kernel
// =============================================================================

/**
 * Put+Signal ping-pong benchmark kernel (2 ranks only).
 *
 * Rank 0 and Rank 1 alternate put_signal() operations.
 * Measures combined NVLink write + signal latency using the convenience API.
 *
 * Half-duplex measurement: one put_signal in flight at a time.
 *
 * @param transport MultiPeerDeviceTransport for NVLink operations
 * @param targetRank The target rank to communicate with
 * @param localSrc Local source buffer to read from
 * @param remoteDst Remote destination buffer to write to (IPC-mapped)
 * @param nbytes Number of bytes to transfer per put
 * @param nSteps Number of ping-pong iterations
 */
template <SyncScope S>
__global__ void multiPeerPutSignalPingPongKernel(
    MultiPeerDeviceTransport transport,
    int targetRank,
    const void* localSrc,
    void* remoteDst,
    std::size_t nbytes,
    int nSteps);

} // namespace comms::pipes::benchmark
