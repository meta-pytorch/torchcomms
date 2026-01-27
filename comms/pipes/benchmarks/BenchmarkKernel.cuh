// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include "comms/pipes/AllToAllv.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/collectives/Broadcast.cuh"
#include "comms/pipes/collectives/BroadcastBinomialTree.cuh"
#include "comms/pipes/collectives/BroadcastRing.cuh"

namespace comms::pipes::benchmark {

/**
 * TimingStats - GPU-side timing statistics using clock64()
 *
 * Records cycle counts inside the kernel for accurate timing without
 * host-side overhead. Use with cudaDevAttrClockRate to convert to time.
 */
struct TimingStats {
  unsigned long long startCycle;
  unsigned long long endCycle;
  unsigned long long totalCycles;
};

/**
 * BroadcastTimingStats - Detailed timing breakdown for broadcast profiling
 *
 * Captures fine-grained timing information for broadcast operations,
 * allowing identification of bottlenecks in the chunked pipelined protocol.
 *
 * To convert cycles to microseconds:
 *   int clockRateKHz;
 *   cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, deviceId);
 *   double microseconds = (double)cycles / (double)clockRateKHz * 1000.0;
 */
struct BroadcastTimingStats {
  unsigned long long waitForReadyCycles; // Time waiting for ChunkState
  unsigned long long dataCopyCycles; // Time in memcpy_vectorized
  unsigned long long signalCycles; // Time signaling readyToRecv/readyToSend
  unsigned long long totalCycles; // End-to-end kernel time
  uint32_t numChunksProcessed; // Chunks handled by this warp
  uint32_t warpId; // Global warp ID for identification
};

// =============================================================================
// Benchmark kernels with configurable parallelism
// =============================================================================

// Send kernel - groupScope selects warp vs block level parallelism
__global__ void p2pSend(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP);

// Recv kernel
__global__ void p2pRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP);

// Timed versions that export GPU-side clock64() timing stats
__global__ void p2pSendTimed(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    TimingStats* stats,
    SyncScope groupScope = SyncScope::WARP);

__global__ void p2pRecvTimed(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    TimingStats* stats,
    SyncScope groupScope = SyncScope::WARP);

// Bidirectional kernel - half groups send, half groups receive
__global__ void p2pBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP);

// Signal benchmark kernel - ping-pong signaling between two peers
__global__ void p2pSignalBenchKernel(
    P2pNvlTransportDevice p2p,
    int nSteps,
    SyncScope groupScope = SyncScope::WARP);

/**
 * AllToAllv benchmark kernel.
 * All ranks participate in all-to-all communication with variable chunk sizes.
 */
__global__ void allToAllvKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos);

/**
 * Broadcast benchmark kernel.
 * Root rank broadcasts data to all non-root ranks using flat-tree algorithm.
 */
__global__ void broadcastKernel(
    void* buff_d,
    int myRank,
    int rootRank,
    DeviceSpan<Transport> transports,
    std::size_t nbytes);

/**
 * Broadcast benchmark kernel using binomial tree algorithm.
 * Root rank broadcasts data to all non-root ranks using O(log N) rounds.
 * More bandwidth-efficient than flat-tree for large messages.
 */
__global__ void broadcastBinomialTreeKernel(
    void* buff_d,
    int myRank,
    int rootRank,
    DeviceSpan<Transport> transports,
    std::size_t nbytes);

/**
 * Adaptive broadcast kernel that selects algorithm based on message size.
 * Uses flat-tree for small messages (< 8MB), ring for larger (>= 8MB).
 * This achieves best performance across all message sizes.
 */
__global__ void broadcastAdaptiveKernel(
    void* buff_d,
    int myRank,
    int rootRank,
    DeviceSpan<Transport> transports,
    std::size_t nbytes);

/**
 * Ring broadcast kernel using ring algorithm.
 * Achieves near-optimal bandwidth utilization by overlapping send/recv.
 * Best performance for large messages (>= 1MB).
 */
__global__ void broadcastRingKernel(
    void* buff_d,
    int myRank,
    int rootRank,
    DeviceSpan<Transport> transports,
    std::size_t nbytes);

} // namespace comms::pipes::benchmark
