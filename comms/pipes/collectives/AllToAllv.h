// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <optional>

#include "comms/pipes/collectives/AllToAllv.cuh"

namespace comms::pipes {

/**
 * Host wrapper for AllToAllv collective communication.
 *
 * Performs variable-sized all-to-all data exchange among multiple ranks.
 * Each rank sends a potentially different amount of data to every other rank,
 * and receives a potentially different amount of data from every other rank.
 *
 * This is a host function that launches the AllToAllv kernel. All device
 * pointers and DeviceSpans must already be allocated and populated on the GPU.
 *
 * @param recvbuff_d Device pointer to receive buffer
 * @param sendbuff_d Device pointer to send buffer (const)
 * @param my_rank_id Current rank ID
 * @param transports_per_rank DeviceSpan of Transport objects (self for my_rank,
 *                            P2P for others)
 * @param send_chunk_infos DeviceSpan of ChunkInfo for send operations
 * @param recv_chunk_infos DeviceSpan of ChunkInfo for receive operations
 * @param timeout Timeout duration (0ms = no timeout, default)
 * @param stream CUDA stream for kernel execution
 * @param num_blocks Number of thread blocks to launch (default: 4)
 * @param num_threads Number of threads per block (default: 256)
 * @param cluster_dim Cluster dimensions for spread cluster launch.
 *                    Default: dim3{4, 1, 1} for better load balancing.
 *                    Set to std::nullopt to use standard kernel launch.
 */
void all_to_allv(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{0},
    cudaStream_t stream = nullptr,
    int num_blocks = 4,
    int num_threads = 256,
    std::optional<dim3> cluster_dim = dim3{4, 1, 1});

} // namespace comms::pipes
