// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/ReduceScatterDirect.cuh"

#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/pipes/collectives/DirectCollectiveUtils.cuh"

namespace comms::pipes {

template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
__global__
__launch_bounds__(kBlockSize, 1) void direct_reduce_scatter_nvl_kernel(
    const __grid_constant__ DirectReduceScatterNvlArgs<T> args,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  timeout.start();

  auto group = make_block_group();
  const int my_rank = args.my_rank;
  const int W = args.num_ranks;
  const std::size_t chunk_elems = args.chunk_elements;
  const std::size_t chunk_bytes = chunk_elems * sizeof(T);
  const std::size_t max_sig = args.signaling_data_size;

  TiledBuffer<char> tile(nullptr, chunk_bytes, group);
  const std::size_t tile_offset = group.group_id * tile.tile_elements;
  const std::size_t tile_bytes = tile.bytes();

  const char* own_src = reinterpret_cast<const char*>(args.input) +
      my_rank * chunk_bytes + tile_offset;
  char* own_dst = reinterpret_cast<char*>(args.output) + tile_offset;
  memcpy_vectorized(own_dst, own_src, tile_bytes, group);
  group.sync();

  if (W <= 1 || tile_bytes == 0) {
    return;
  }

  const std::size_t pipeline_window =
      direct_pipeline_window(args.peers, my_rank, W, group.total_groups);
  PIPES_DEVICE_CHECK_MSG(
      pipeline_window != 0,
      "direct NVLink reduce-scatter pipeline window is zero");

  using ReduceOp = TileReduceStaged<T, AccumOp, kTileElems, kBlockSize>;
  const char* input_base = reinterpret_cast<const char*>(args.input);
  char* output_base = reinterpret_cast<char*>(args.output);

  for (std::size_t off = 0; off < tile_bytes; off += pipeline_window) {
    const std::size_t remaining = tile_bytes - off;
    const std::size_t window =
        remaining < pipeline_window ? remaining : pipeline_window;

    for (int peer_rank = 0; peer_rank < W; ++peer_rank) {
      if (peer_rank == my_rank) {
        continue;
      }
      const char* send_src =
          input_base + peer_rank * chunk_bytes + tile_offset + off;
      auto peer = args.peers[peer_rank];
      peer.send(group, send_src, window, group.total_groups, max_sig, timeout);
    }

    for (int peer_rank = 0; peer_rank < W; ++peer_rank) {
      if (peer_rank == my_rank) {
        continue;
      }
      char* dst = output_base + tile_offset + off;
      auto peer = args.peers[peer_rank];
      peer.template recv<ReduceOp>(
          group, dst, window, group.total_groups, max_sig, timeout, dst);
    }
  }
#endif
}

template __global__ void
direct_reduce_scatter_nvl_kernel<float, SumOp, 16384, 512>(
    const __grid_constant__ DirectReduceScatterNvlArgs<float>,
    Timeout);

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
  const std::size_t chunk_bytes = chunk_elements * sizeof(T);

  TiledBuffer<char> ring_tile(nullptr, chunk_bytes, group);
  const std::size_t io_tile_offset = group.group_id * ring_tile.tile_elements;
  const std::size_t io_tile_bytes = ring_tile.bytes();

  const char* input_base = reinterpret_cast<const char*>(workspace);
  char* output_base = reinterpret_cast<char*>(output);

  if (ib_size <= 1) {
    memcpy_vectorized(
        output_base + io_tile_offset,
        input_base + io_tile_offset,
        io_tile_bytes,
        group);
    return;
  }

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
      const char* local_input = input_base +
          static_cast<std::size_t>(current_rank) * chunk_bytes +
          io_tile_offset + off;

      if (step < ib_size - 2) {
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
        char* dst = output_base + io_tile_offset + off;
        prev.template recv<ReduceOp>(
            group,
            dst,
            window,
            group.total_groups,
            max_sig,
            timeout,
            local_input);
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

  hierarchical_reduce_scatter_nvl_reduce_to_workspace<float, ReduceOp>(
      group,
      args.ib_size,
      args.nvl_rank,
      args.nvl_size,
      args.chunk_elements,
      args.nvl_signaling_data_size,
      args.nvl_peers,
      args.input,
      args.workspace,
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
