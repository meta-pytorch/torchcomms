// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <optional>

#include "comms/pipes/collectives/AllGather.cuh"

namespace comms::pipes {

/**
 * Host wrapper for AllGather collective communication.
 *
 * Gathers data from all ranks, each rank contributes sendcount bytes, and after
 * the operation, each rank has nranks * sendcount bytes containing all ranks'
 * data.
 *
 * This is a host function that launches the AllGather kernel. All device
 * pointers and DeviceSpans must already be allocated and populated on the GPU.
 *
 * @param recvbuff_d Device pointer to receive buffer (nranks * sendcount bytes)
 * @param sendbuff_d Device pointer to send buffer (sendcount bytes)
 * @param sendcount Number of bytes each rank contributes
 * @param my_rank_id Current rank ID
 * @param transports_per_rank DeviceSpan of Transport objects (self for my_rank,
 *                            P2P for others)
 * @param timeout Timeout duration (0ms = no timeout, default)
 * @param stream CUDA stream for kernel execution
 * @param num_blocks Number of thread blocks to launch (default: 4)
 * @param num_threads Number of threads per block (default: 256)
 * @param cluster_dim Cluster dimensions for spread cluster launch.
 *                    Default: dim3{4, 1, 1} for better load balancing.
 *                    Set to std::nullopt to use standard kernel launch.
 */
void all_gather(
    void* recvbuff_d,
    const void* sendbuff_d,
    std::size_t sendcount,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{0},
    cudaStream_t stream = nullptr,
    int num_blocks = 4,
    int num_threads = 256,
    std::optional<dim3> cluster_dim = dim3{4, 1, 1});

} // namespace comms::pipes
