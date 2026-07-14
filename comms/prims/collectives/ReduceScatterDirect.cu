// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/ReduceScatterDirect.cuh"

#include "comms/prims/collectives/DirectCollectiveUtils.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"

namespace comms::prims {

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
      direct_pipeline_window(args.peers, my_rank, W);
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
      peer.send(group, send_src, window, max_sig, timeout);
    }

    for (int peer_rank = 0; peer_rank < W; ++peer_rank) {
      if (peer_rank == my_rank) {
        continue;
      }
      char* dst = output_base + tile_offset + off;
      auto peer = args.peers[peer_rank];
      peer.template recv<ReduceOp>(group, dst, window, max_sig, timeout, dst);
    }
  }
#endif
}

template __global__ void
direct_reduce_scatter_nvl_kernel<float, SumOp, 16384, 512>(
    const __grid_constant__ DirectReduceScatterNvlArgs<float>,
    Timeout);

} // namespace comms::prims
