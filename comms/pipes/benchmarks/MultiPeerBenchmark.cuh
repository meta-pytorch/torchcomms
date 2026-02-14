// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/common/DeviceConstants.cuh"
#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::benchmark {

// Use SyncScope from ThreadGroup.cuh for thread group type selection

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
 *
 * Note: For 2-rank case, there's exactly one peer at index 0.
 */
template <SyncScope S>
__global__ void multiPeerSignalPingPongKernel(
    MultiPeerDeviceTransport transport,
    int peerIndex,
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
// Counter Benchmark Kernel
// =============================================================================

/**
 * Counter benchmark kernel.
 *
 * Measures incrementCounter() / waitCounter() latency.
 * Local operation only (no NVLink).
 */
template <SyncScope G>
__global__ void multiPeerCounterKernel(
    MultiPeerDeviceTransport transport,
    int nSteps);

} // namespace comms::pipes::benchmark
