// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// AllToAllv using tile-based pipelined send/recv with NVLink and IB support.
//
// Uses MultiPeerDeviceHandle for per-peer NVLink/IB transport dispatch.
// Configurable num_blocks_nvl / num_blocks_ib for independent concurrency.
// ThreadGroup::split() partitions blocks between NVL and IB groups.
// NVL 1D: sequential peer iteration with paired send/recv ordering.
// NVL 2D: partition_interleaved across peers for parallel processing.
// IB: partition across IB peers with round-robin for overflow.

#pragma once

#include <cstddef>

#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/TiledBuffer.cuh"

namespace comms::pipes {

/**
 * AllToAllvTileArgs — Kernel arguments for tile-based alltoallv.
 *
 * All arrays are indexed by peer rank (0 to nRanks-1).
 * The entry at myRank is ignored (no self-send).
 */
struct AllToAllvTileArgs {
  MultiPeerDeviceHandle handle;

  // Send side: per-peer source pointers and byte counts
  char** send_ptrs;
  std::size_t* send_counts;

  // Recv side: per-peer destination pointers and byte counts
  char** recv_ptrs;
  std::size_t* recv_counts;

  // Blocks allocated to NVLink peers (portion of total grid).
  // For NVL-only: set to total grid size, num_blocks_ib = 0.
  int num_blocks_nvl;
  // Blocks allocated to IB peers (portion of total grid).
  // For IB-only: set to total grid size, num_blocks_nvl = 0.
  int num_blocks_ib;
  // Sub-chunk signaling hint (bytes). 0 = one signal per slot fill.
  std::size_t max_signal_bytes;
};

#ifdef __CUDACC__
/**
 * NVL 1D sequential + IB 2D parallel.
 * Best for large NVL messages. NVL blocks iterate peers sequentially.
 * Grid: 2 * (num_blocks_nvl + num_blocks_ib).
 */
__global__ __launch_bounds__(512, 1) void alltoallv_tile_1d_kernel(
    const __grid_constant__ AllToAllvTileArgs args,
    Timeout timeout = Timeout());

/**
 * NVL 2D interleaved + IB 2D parallel.
 * Best for small NVL messages. NVL blocks partitioned across peers.
 * Grid: 2 * (num_blocks_nvl + num_blocks_ib).
 */
__global__ __launch_bounds__(512, 1) void alltoallv_tile_2d_kernel(
    const __grid_constant__ AllToAllvTileArgs args,
    Timeout timeout = Timeout());
#else
void alltoallv_tile_1d_kernel(AllToAllvTileArgs args, Timeout timeout);
void alltoallv_tile_2d_kernel(AllToAllvTileArgs args, Timeout timeout);
#endif

} // namespace comms::pipes
