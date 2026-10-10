// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/HierarchicalReduceScatterDirect.cuh"

#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/pipes/collectives/DirectCollectiveUtils.cuh"

namespace comms::pipes {

// Runs the NVLink reduction phase of hierarchical reduce-scatter. For each
// `ib_dst` slot the local rank's slice is reduced with all other NVL peers'
// slices. The reduction destination is `workspace[ib_dst]` for every slot
// EXCEPT `ib_dst == ib_rank`, which is written directly into `output`. The IB
// ring's final reduce-into-output step consumes that slot as `local_input`, so
// diverting it here removes one full workspace round-trip (one write here +
// one read in the IB ring) for the `ib_rank` slot.
template <typename T, typename ReduceOp, typename Group>
__device__ __forceinline__ void hierarchical_reduce_scatter_nvl_reduce_phase(
    Group& group,
    int ib_size,
    int ib_rank,
    int nvl_rank,
    int nvl_size,
    std::size_t chunk_elements,
    std::size_t max_sig,
    const P2pNvlTransportDevice* peers,
    const T* input,
    T* workspace,
    T* output,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  const std::size_t chunk_bytes = chunk_elements * sizeof(T);
  const char* input_base = reinterpret_cast<const char*>(input);
  char* workspace_base = reinterpret_cast<char*>(workspace);
  char* output_base = reinterpret_cast<char*>(output);

  TiledBuffer<char> tile(nullptr, chunk_bytes, group);
  const std::size_t tile_offset = group.group_id * tile.tile_elements;
  const std::size_t tile_bytes = tile.bytes();

  // Peer set and group are loop-invariant, so compute pipeline_window once.
  const std::size_t pipeline_window = (nvl_size > 1)
      ? direct_pipeline_window(peers, nvl_rank, nvl_size, group.total_groups)
      : 0;
  if (nvl_size > 1) {
    PIPES_DEVICE_CHECK_MSG(
        pipeline_window != 0,
        "hierarchical reduce-scatter NVLink reduce pipeline window is zero");
  }

  for (int ib_dst = 0; ib_dst < ib_size; ++ib_dst) {
    const std::size_t input_chunk =
        (static_cast<std::size_t>(ib_dst) * nvl_size + nvl_rank) * chunk_bytes;
    // Divert the `ib_rank` slot directly into output; everything else lands in
    // workspace as before.
    char* dst = (ib_dst == ib_rank)
        ? (output_base + tile_offset)
        : (workspace_base + static_cast<std::size_t>(ib_dst) * chunk_bytes +
           tile_offset);

    memcpy_vectorized(
        dst, input_base + input_chunk + tile_offset, tile_bytes, group);
    group.sync();

    if (nvl_size <= 1 || tile_bytes == 0) {
      continue;
    }

    for (std::size_t off = 0; off < tile_bytes; off += pipeline_window) {
      const std::size_t remaining = tile_bytes - off;
      const std::size_t window =
          remaining < pipeline_window ? remaining : pipeline_window;

      for (int peer_rank = 0; peer_rank < nvl_size; ++peer_rank) {
        if (peer_rank == nvl_rank) {
          continue;
        }
        const std::size_t peer_input_chunk =
            (static_cast<std::size_t>(ib_dst) * nvl_size + peer_rank) *
            chunk_bytes;
        auto peer = peers[peer_rank];
        peer.send(
            group,
            input_base + peer_input_chunk + tile_offset + off,
            window,
            group.total_groups,
            max_sig,
            timeout);
      }

      for (int peer_rank = 0; peer_rank < nvl_size; ++peer_rank) {
        if (peer_rank == nvl_rank) {
          continue;
        }
        char* window_dst = dst + off;
        auto peer = peers[peer_rank];
        peer.template recv<ReduceOp>(
            group,
            window_dst,
            window,
            group.total_groups,
            max_sig,
            timeout,
            window_dst);
      }
    }
  }
#endif
}

template <typename T, typename ReduceOp, typename Group>
__device__ __forceinline__ void hierarchical_reduce_scatter_ib_ring(
    Group& group,
    int ib_rank,
    int ib_size,
    std::size_t chunk_elements,
    std::size_t max_sig,
    const HierarchicalReduceScatterIbgdaRing& topo,
    const T* workspace,
    T* output,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  // For ib_size <= 1 the NVL reduction phase has already written the final
  // result for `ib_dst == ib_rank` directly into `output`, so the IB ring has
  // nothing to do.
  if (ib_size <= 1) {
    return;
  }

  const std::size_t chunk_bytes = chunk_elements * sizeof(T);

  TiledBuffer<char> ring_tile(nullptr, chunk_bytes, group);
  const std::size_t io_tile_offset = group.group_id * ring_tile.tile_elements;
  const std::size_t io_tile_bytes = ring_tile.bytes();

  const char* input_base = reinterpret_cast<const char*>(workspace);
  char* output_base = reinterpret_cast<char*>(output);

  auto& prev = *topo.prev;
  auto& next = *topo.next;
  const std::size_t pipeline_window = next.pipeline_window(group.total_groups);
  PIPES_DEVICE_CHECK_MSG(
      pipeline_window != 0,
      "hierarchical reduce-scatter IB pipeline window is zero");

  const int stride = (ib_rank - topo.prev_rank + ib_size) % ib_size;
  PIPES_DEVICE_CHECK_MSG(
      stride != 0 && (ib_rank + stride) % ib_size == topo.next_rank,
      "hierarchical reduce-scatter IB ring topology is inconsistent");

  for (std::size_t off = 0; off < io_tile_bytes; off += pipeline_window) {
    const std::size_t remaining = io_tile_bytes - off;
    const std::size_t window =
        (remaining < pipeline_window) ? remaining : pipeline_window;

    int current_rank = (ib_rank + ib_size - stride) % ib_size;
    const char* send_src = input_base +
        static_cast<std::size_t>(current_rank) * chunk_bytes + io_tile_offset +
        off;
    next.send(group, send_src, window, group.total_groups, max_sig, timeout);

    for (int step = 0; step < ib_size - 1; step++) {
      current_rank = (current_rank + ib_size - stride) % ib_size;

      if (step < ib_size - 2) {
        const char* local_input = input_base +
            static_cast<std::size_t>(current_rank) * chunk_bytes +
            io_tile_offset + off;
        prev.template forward<ReduceOp>(
            group,
            nullptr,
            next,
            window,
            group.total_groups,
            max_sig,
            timeout,
            local_input);
      } else {
        // Final step: current_rank == ib_rank by construction. The NVL phase
        // has already written this rank's NVL-reduced slice into `output`, so
        // we use `output` itself as local_input for the reduce instead of
        // reading workspace[ib_rank]. This eliminates the workspace round-trip
        // for the `ib_rank` slot.
        char* dst = output_base + io_tile_offset + off;
        prev.template recv<ReduceOp>(
            group, dst, window, group.total_groups, max_sig, timeout, dst);
      }
    }
  }
#endif
}

__global__
__launch_bounds__(512, 1) void hierarchical_reduce_scatter_fused_float_sum_kernel(
    const __grid_constant__ HierarchicalReduceScatterFusedArgs<float> args,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  timeout.start();

  auto group = make_block_group();
  using ReduceOp = TileReduceStaged<float, SumOp, 16384, 512>;

  hierarchical_reduce_scatter_nvl_reduce_phase<float, ReduceOp>(
      group,
      args.ib_size,
      args.ib_rank,
      args.nvl_rank,
      args.nvl_size,
      args.chunk_elements,
      args.nvl_signaling_data_size,
      args.nvl_peers,
      args.input,
      args.workspace,
      args.output,
      timeout);
  group.sync();

  hierarchical_reduce_scatter_ib_ring<float, ReduceOp>(
      group,
      args.ib_rank,
      args.ib_size,
      args.chunk_elements,
      args.ib_signaling_data_size,
      args.ib_ring,
      args.workspace,
      args.output,
      timeout);
#endif
}

} // namespace comms::pipes
