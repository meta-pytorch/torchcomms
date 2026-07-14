// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims {

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
    int num_ranks) {
  std::size_t window = 0;
  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) {
      continue;
    }
    const std::size_t peer_window = peers[peer].pipeline_window();
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
    const P2pNvlTransportDevice* peers,
    char* recvbuf,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  if (nvl_size <= 1) {
    return;
  }

  const std::size_t pipeline_window =
      direct_pipeline_window(peers, nvl_rank, nvl_size);
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
        peer.send(group, send_src, window, max_sig, timeout);
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
        peer.recv(group, dst, window, max_sig, timeout);
      }
    }
  }
#endif
}

} // namespace comms::prims
