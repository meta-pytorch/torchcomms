// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <chrono>

#include "comms/pipes/collectives/AllToAllvLl128.cuh"

namespace comms::pipes {

/**
 * Host wrapper for AllToAllv LL128 collective communication.
 *
 * Uses the LL128 protocol for fine-grained (128B packet) pipelining with
 * inline flag signaling, optimized for small/medium messages (<= 256KB).
 *
 * Requires LL128 buffers to be allocated in the transport config
 * (MultiPeerNvlTransportConfig::ll128BufferSize > 0).
 *
 * All user buffers and ChunkInfo sizes must be 16-byte aligned.
 *
 * This overload creates a Timeout internally per call. For pipelined usage
 * (multiple back-to-back calls), prefer the Timeout overload below to avoid
 * per-call cudaGetDevice/cudaDeviceGetAttribute overhead.
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
 * @param num_blocks Number of thread blocks to launch (default: 16).
 *                   Must satisfy: num_blocks * (num_threads / 32) >= 2 *
 * nranks. Default 16 supports up to 72 NVLink ranks (GB200)
 *                   (16 blocks * 16 warps = 256 >= 2*71 = 142).
 * @param num_threads Number of threads per block (default: 512)
 */
void all_to_allv_ll128(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{0},
    cudaStream_t stream = nullptr,
    int num_blocks = 16,
    int num_threads = 512);

/**
 * Host wrapper for AllToAllv LL128 with pre-built Timeout.
 *
 * Use this overload for pipelined/multi-call usage (e.g., benchmarks) where
 * Timeout is created once outside the loop (avoids per-call CUDA API
 * queries from makeTimeout).
 *
 * Flag management is handled internally by the LL128 protocol layer.
 *
 * @param recvbuff_d Device pointer to receive buffer
 * @param sendbuff_d Device pointer to send buffer (const)
 * @param my_rank_id Current rank ID
 * @param transports_per_rank DeviceSpan of Transport objects
 * @param send_chunk_infos DeviceSpan of ChunkInfo for send operations
 * @param recv_chunk_infos DeviceSpan of ChunkInfo for receive operations
 * @param timeout_config Pre-built Timeout (create once with makeTimeout())
 * @param stream CUDA stream for kernel execution
 * @param num_blocks Number of thread blocks to launch (default: 16)
 * @param num_threads Number of threads per block (default: 512)
 */
void all_to_allv_ll128(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout_config,
    cudaStream_t stream = nullptr,
    int num_blocks = 16,
    int num_threads = 512);

} // namespace comms::pipes
