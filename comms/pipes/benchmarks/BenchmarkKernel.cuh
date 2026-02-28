// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/TimeoutUtils.h"

#include "comms/pipes/collectives/AllToAllv.cuh"
#include "comms/pipes/collectives/broadcast/BroadcastTopologies.cuh"

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

// =============================================================================
// Benchmark kernels with configurable parallelism
// =============================================================================

// Send kernel - groupScope selects warp vs block level parallelism
__global__ void p2pSend(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Recv kernel
__global__ void p2pRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

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
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Signal benchmark kernel - ping-pong signaling between two peers
__global__ void p2pSignalBenchKernel(
    P2pNvlTransportDevice p2p,
    int nSteps,
    SyncScope groupScope = SyncScope::WARP);

// Send one kernel - single chunk transfer with metadata
__global__ void p2pSendOne(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP);

// Recv one kernel - single chunk receive with metadata
__global__ void p2pRecvOne(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    SyncScope groupScope = SyncScope::WARP);

// Send multiple kernel - transfer multiple chunks with metadata
__global__ void p2pSendMultiple(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    DeviceSpan<const std::size_t> chunkSizes,
    DeviceSpan<const std::size_t> chunkIndices,
    SyncScope groupScope = SyncScope::WARP);

// Recv multiple kernel - receive multiple chunks with metadata
__global__ void p2pRecvMultiple(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    DeviceSpan<std::size_t> chunkSizes,
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
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout);

/**
 * Broadcast benchmark kernel using flat-tree algorithm.
 *
 * The root rank broadcasts data to all non-root ranks in a single step.
 * Each non-root rank receives the complete data directly from root.
 *
 * @param buff_d     Device buffer for both send (root) and receive (non-root).
 *                   Must have at least nbytes allocated.
 * @param myRank     This rank's ID in the communicator (0 to numRanks-1).
 * @param rootRank   The rank that originates the broadcast data.
 * @param transports Array of Transport handles for peer communication.
 *                   Index i corresponds to peer rank i.
 * @param nbytes     Number of bytes to broadcast from root to all peers.
 *
 * Thread/block configuration: Recommended minimum 8 blocks x 256 threads
 * for good overlap of computation and communication.
 */
__global__ void broadcastFlatKernel(
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
 * Uses flat-tree for small messages (< 4MB), ring for larger (>= 4MB).
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
