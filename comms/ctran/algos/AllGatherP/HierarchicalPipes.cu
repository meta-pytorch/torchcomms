// Copyright (c) Meta Platforms, Inc. and affiliates.

#if defined(ENABLE_PIPES)

#include "comms/ctran/algos/AllGatherP/HierarchicalPipes.cuh"

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/TiledBuffer.cuh"

namespace {

struct MemcpyAndSelfCopy {
  template <typename... Args>
  __device__ __forceinline__ static void send(
      char* staging,
      const char* src,
      std::size_t nbytes,
      comms::pipes::ThreadGroup& group,
      std::size_t byte_offset,
      char* self_dst,
      Args...) {
    comms::pipes::memcpy_vectorized(
        staging, self_dst + byte_offset, src, nbytes, group);
  }
};

__device__ __forceinline__ std::size_t direct_pipeline_window(
    const comms::pipes::P2pNvlTransportDevice* peers,
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

template <typename Group>
__device__ __forceinline__ void
hierarchical_allgather_nvl_broadcast_from_recvbuf(
    Group& group,
    int ib_size,
    int nvl_rank,
    int nvl_size,
    std::size_t sendcount,
    std::size_t max_sig,
    const comms::pipes::P2pNvlTransportDevice* peers,
    char* recvbuf,
    comms::pipes::Timeout timeout) {
  if (nvl_size <= 1) {
    return;
  }

  const std::size_t pipeline_window =
      direct_pipeline_window(peers, nvl_rank, nvl_size, group.total_groups);
  PIPES_DEVICE_CHECK_MSG(
      pipeline_window != 0,
      "ctran hierarchical allgather NVLink broadcast pipeline window is zero");

  for (int ib_src = 0; ib_src < ib_size; ++ib_src) {
    comms::pipes::TiledBuffer<char> tile(nullptr, sendcount, group);
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
}

} // namespace

__global__
__launch_bounds__(ctran::allgatherp::hierarchical_pipes::kBlockSize, 1) void ncclKernelAllGatherPHierarchicalPipes(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    ctran::allgatherp::hierarchical_pipes::KernArgs kernArgs) {
#ifdef __CUDA_ARCH__
  const auto& args = kernArgs.args;
  auto timeout = kernArgs.timeout;
  timeout.start();

  auto group = comms::pipes::make_block_group();
  const int W = args.ib_size;
  const std::size_t chunk_bytes = args.sendcount;
  const std::size_t max_sig = args.ib_signaling_data_size;

  comms::pipes::TiledBuffer<char> ring_tile(nullptr, chunk_bytes, group);
  const std::size_t io_tile_offset = group.group_id * ring_tile.tile_elements;
  const std::size_t io_tile_bytes = ring_tile.bytes();

  const char* own_src = args.sendbuf + io_tile_offset;
  char* own_dst = args.recvbuf +
      (static_cast<std::size_t>(args.ib_rank) * args.nvl_size + args.nvl_rank) *
          chunk_bytes +
      io_tile_offset;

  if (W <= 1) {
    comms::pipes::memcpy_vectorized(own_dst, own_src, io_tile_bytes, group);
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
      "ctran hierarchical allgather IB pipeline window is zero");

  const int stride = (args.ib_rank - topo.prev_rank + W) % W;

  for (std::size_t off = 0; off < io_tile_bytes; off += pipeline_window) {
    const std::size_t remaining = io_tile_bytes - off;
    const std::size_t window =
        remaining < pipeline_window ? remaining : pipeline_window;

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

#endif // ENABLE_PIPES
