// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/DirectNvl.cuh"

#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/TiledBuffer.cuh"

namespace comms::pipes {

namespace {

struct MemcpyAndSelfCopy {
  template <typename... Args>
  __device__ __forceinline__ static void send(
      char* staging,
      const char* src,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      char* self_dst,
      Args...) {
    memcpy_vectorized(staging, self_dst + byte_offset, src, nbytes, group);
  }
};

__device__ __forceinline__ std::size_t direct_pipeline_window(
    const P2pNvlTransportDevice* peers,
    int my_rank,
    int num_ranks,
    int total_groups) {
  std::size_t window = 0;
  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) {
      continue;
    }
    const std::size_t peer_window = peers[peer].pipeline_window(total_groups);
    window = window == 0 || peer_window < window ? peer_window : window;
  }
  return window;
}

} // namespace

template <int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void direct_allgather_nvl_kernel(
    const __grid_constant__ DirectAllgatherNvlArgs args,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  timeout.start();

  auto group = make_block_group();
  const int my_rank = args.my_rank;
  const int W = args.num_ranks;
  const std::size_t sendcount = args.sendcount;
  const std::size_t max_sig = args.signaling_data_size;

  TiledBuffer<char> tile(nullptr, sendcount, group);
  const std::size_t tile_offset = group.group_id * tile.tile_elements;
  const std::size_t tile_bytes = tile.bytes();

  const char* own_src = args.sendbuf + tile_offset;
  char* own_dst = args.recvbuf + my_rank * sendcount + tile_offset;
  memcpy_vectorized(own_dst, own_src, tile_bytes, group);
  group.sync();

  if (W <= 1 || tile_bytes == 0) {
    return;
  }

  const std::size_t pipeline_window =
      direct_pipeline_window(args.peers, my_rank, W, group.total_groups);
  PIPES_DEVICE_CHECK_MSG(
      pipeline_window != 0, "direct NVLink allgather pipeline window is zero");

  for (std::size_t off = 0; off < tile_bytes; off += pipeline_window) {
    const std::size_t remaining = tile_bytes - off;
    const std::size_t window =
        remaining < pipeline_window ? remaining : pipeline_window;

    const char* send_src = args.sendbuf + tile_offset + off;
    for (int peer_rank = 0; peer_rank < W; ++peer_rank) {
      if (peer_rank == my_rank) {
        continue;
      }
      auto peer = args.peers[peer_rank];
      peer.send(group, send_src, window, group.total_groups, max_sig, timeout);
    }

    for (int peer_rank = 0; peer_rank < W; ++peer_rank) {
      if (peer_rank == my_rank) {
        continue;
      }
      char* dst = args.recvbuf + peer_rank * sendcount + tile_offset + off;
      auto peer = args.peers[peer_rank];
      peer.recv(group, dst, window, group.total_groups, max_sig, timeout);
    }
  }
#endif
}

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

template <typename Group>
__device__ __forceinline__ void
hierarchical_allgather_nvl_broadcast_from_recvbuf(
    Group& group,
    int ib_size,
    int nvl_rank,
    int nvl_size,
    std::size_t sendcount,
    std::size_t max_sig,
    const P2pNvlTransportDevice* peers,
    char* recvbuf,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  if (nvl_size <= 1) {
    return;
  }

  const std::size_t pipeline_window =
      direct_pipeline_window(peers, nvl_rank, nvl_size, group.total_groups);
  PIPES_DEVICE_CHECK_MSG(
      pipeline_window != 0,
      "hierarchical allgather NVLink broadcast pipeline window is zero");

  for (int ib_src = 0; ib_src < ib_size; ++ib_src) {
    TiledBuffer<char> tile(nullptr, sendcount, group);
    const std::size_t tile_offset = group.group_id * tile.tile_elements;
    const std::size_t tile_bytes = tile.bytes();

    for (std::size_t off = 0; off < tile_bytes; off += pipeline_window) {
      const std::size_t remaining = tile_bytes - off;
      const std::size_t window =
          remaining < pipeline_window ? remaining : pipeline_window;
      const char* send_src = recvbuf +
          (static_cast<std::size_t>(ib_src) * nvl_size + nvl_rank) * sendcount +
          tile_offset + off;

      for (int peer_rank = 0; peer_rank < nvl_size; ++peer_rank) {
        if (peer_rank == nvl_rank) {
          continue;
        }
        auto peer = peers[peer_rank];
        peer.send(
            group, send_src, window, group.total_groups, max_sig, timeout);
      }

      for (int peer_rank = 0; peer_rank < nvl_size; ++peer_rank) {
        if (peer_rank == nvl_rank) {
          continue;
        }
        char* dst = recvbuf +
            (static_cast<std::size_t>(ib_src) * nvl_size + peer_rank) *
                sendcount +
            tile_offset + off;
        auto peer = peers[peer_rank];
        peer.recv(group, dst, window, group.total_groups, max_sig, timeout);
      }
    }
  }
#endif
}

template <int kBlockSize>
__global__
__launch_bounds__(kBlockSize, 1) void hierarchical_allgather_fused_kernel(
    const __grid_constant__ HierarchicalAllgatherFusedArgs args,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  timeout.start();

  auto group = make_block_group();
  const int W = args.ib_size;
  const std::size_t chunk_bytes = args.sendcount;
  const std::size_t max_sig = args.ib_signaling_data_size;

  TiledBuffer<char> ring_tile(nullptr, chunk_bytes, group);
  const std::size_t io_tile_offset = group.group_id * ring_tile.tile_elements;
  const std::size_t io_tile_bytes = ring_tile.bytes();

  const char* own_src = args.sendbuf + io_tile_offset;
  char* own_dst = args.recvbuf +
      (static_cast<std::size_t>(args.ib_rank) * args.nvl_size + args.nvl_rank) *
          chunk_bytes +
      io_tile_offset;

  // Single IB node (W==1): self-copy then NVL broadcast within the local
  // NVL group. The broadcast is still needed because each NVL peer must
  // receive every other peer's chunk even when there is only one IB node.
  if (W <= 1) {
    memcpy_vectorized(own_dst, own_src, io_tile_bytes, group);
    group.sync();
    hierarchical_allgather_nvl_broadcast_from_recvbuf(
        group,
        args.ib_size,
        args.nvl_rank,
        args.nvl_size,
        args.sendcount,
        args.nvl_signaling_data_size,
        args.nvl_peers,
        args.recvbuf,
        timeout);
    return;
  }

  const auto& topo = args.ib_ring;
  auto& prev = *topo.prev;
  auto& next = *topo.next;
  const std::size_t pipeline_window = next.pipeline_window(group.total_groups);
  PIPES_DEVICE_CHECK_MSG(
      pipeline_window != 0,
      "hierarchical allgather IB pipeline window is zero");

  const int stride = (args.ib_rank - topo.prev_rank + W) % W;

  for (std::size_t off = 0; off < io_tile_bytes; off += pipeline_window) {
    const std::size_t remaining = io_tile_bytes - off;
    const std::size_t window =
        (remaining < pipeline_window) ? remaining : pipeline_window;

    const char* send_src = args.sendbuf + io_tile_offset + off;
    next.template send<MemcpyAndSelfCopy>(
        group,
        send_src,
        window,
        group.total_groups,
        max_sig,
        timeout,
        own_dst + off);

    int fwd_current_rank = args.ib_rank;
    for (int step = 0; step < W - 1; step++) {
      fwd_current_rank = (fwd_current_rank + W - stride) % W;
      char* dst = args.recvbuf +
          (static_cast<std::size_t>(fwd_current_rank) * args.nvl_size +
           args.nvl_rank) *
              chunk_bytes +
          io_tile_offset + off;

      if (step < W - 2) {
        prev.forward(
            group, dst, next, window, group.total_groups, max_sig, timeout);
      } else {
        prev.recv(group, dst, window, group.total_groups, max_sig, timeout);
      }
    }
  }

  group.sync();

  hierarchical_allgather_nvl_broadcast_from_recvbuf(
      group,
      args.ib_size,
      args.nvl_rank,
      args.nvl_size,
      args.sendcount,
      args.nvl_signaling_data_size,
      args.nvl_peers,
      args.recvbuf,
      timeout);
#endif
}

template __global__ void direct_allgather_nvl_kernel<512>(
    const __grid_constant__ DirectAllgatherNvlArgs,
    Timeout);

template __global__ void
direct_reduce_scatter_nvl_kernel<float, SumOp, 16384, 512>(
    const __grid_constant__ DirectReduceScatterNvlArgs<float>,
    Timeout);

template __global__ void hierarchical_allgather_fused_kernel<512>(
    const __grid_constant__ HierarchicalAllgatherFusedArgs,
    Timeout);

} // namespace comms::pipes
