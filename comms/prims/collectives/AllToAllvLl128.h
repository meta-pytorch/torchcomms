// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <chrono>

#include "comms/prims/collectives/AllToAllvLl128.cuh"
#include "comms/prims/transport/ll128/Ll128AutoTune.cuh"

namespace comms::prims {

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
 * This compatibility overload accepts only a zero timeout. New callers that
 * need abort or timeout behavior should pass an externally owned `AbortDevice`
 * with the overload below.
 *
 * @param recvbuff_d Device pointer to receive buffer
 * @param sendbuff_d Device pointer to send buffer (const)
 * @param my_rank_id Current rank ID
 * @param transports_per_rank DeviceSpan of Transport objects (self for my_rank,
 *                            P2P for others)
 * @param send_chunk_infos DeviceSpan of ChunkInfo for send operations
 * @param recv_chunk_infos DeviceSpan of ChunkInfo for receive operations
 * @param timeout Compatibility timeout; non-zero values are rejected.
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
    int num_threads = kLl128ThreadsPerBlock);

/**
 * Host wrapper for AllToAllv LL128 with an externally owned device abort
 * handle.
 *
 * Flag management is handled internally by the LL128 protocol layer.
 *
 * @param recvbuff_d Device pointer to receive buffer
 * @param sendbuff_d Device pointer to send buffer (const)
 * @param my_rank_id Current rank ID
 * @param transports_per_rank DeviceSpan of Transport objects
 * @param send_chunk_infos DeviceSpan of ChunkInfo for send operations
 * @param recv_chunk_infos DeviceSpan of ChunkInfo for receive operations
 * @param abort Device abort handle owned by the caller.
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
    AbortDevice abort,
    cudaStream_t stream = nullptr,
    int num_blocks = 16,
    int num_threads = kLl128ThreadsPerBlock);

} // namespace comms::prims
