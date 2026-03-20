// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <cstddef>
#include <optional>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/AllToAllv.cuh" // ChunkInfo

namespace comms::pipes {

/**
 * Configuration for the hybrid AllToAllv auto-selector.
 *
 * Automatically chooses LL128 for small messages and Simple for large ones,
 * giving users best-of-both-worlds performance without manual protocol
 * selection.
 */
struct AllToAllvAutoConfig {
  /// Threshold in bytes per peer. Messages <= this use LL128, > this use
  /// Simple. Sweep benchmarks show LL128 beats NCCL by 1.36x at 256KB
  /// and converges at ~1MB, so 256KB captures the full LL128 advantage.
  /// Callers should size ll128BufferSize for their max expected per-peer
  /// message size for best performance (chunking handles undersized buffers
  /// but adds synchronization overhead).
  std::size_t ll128Threshold{256 * 1024};

  /// Simple protocol settings.
  int simpleNumBlocks{4};
  int simpleNumThreads{256};
  std::optional<dim3> simpleClusterDim{dim3{4, 1, 1}};

  /// LL128 protocol settings. 0 = use auto-tune based on message size and
  /// nranks.
  int ll128NumBlocks{0};
  int ll128NumThreads{512};
};

/**
 * Host wrapper for hybrid AllToAllv that auto-selects between LL128 and Simple.
 *
 * Dispatches to all_to_allv_ll128() for small messages (<= threshold) and
 * all_to_allv() for large messages (> threshold). The decision is based on
 * max_bytes_per_peer, which the caller must provide (since chunk infos live
 * in device memory).
 *
 * @param recvbuff_d Device pointer to receive buffer
 * @param sendbuff_d Device pointer to send buffer (const)
 * @param my_rank_id Current rank ID
 * @param nranks Total number of ranks
 * @param transports_per_rank DeviceSpan of Transport objects
 * @param send_chunk_infos DeviceSpan of ChunkInfo for send operations
 * @param recv_chunk_infos DeviceSpan of ChunkInfo for receive operations
 * @param max_bytes_per_peer Maximum bytes sent to any single peer (caller
 *                           provides this since chunk infos are in device
 *                           memory)
 * @param config Auto-selector configuration
 * @param timeout Timeout duration (0ms = no timeout, default)
 * @param stream CUDA stream for kernel execution
 */
void all_to_allv_auto(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    std::size_t max_bytes_per_peer,
    const AllToAllvAutoConfig& config = {},
    std::chrono::milliseconds timeout = std::chrono::milliseconds{0},
    cudaStream_t stream = nullptr);

} // namespace comms::pipes
