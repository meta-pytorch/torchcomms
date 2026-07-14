// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/AllGatherDirect.cuh"

#include "comms/common/AtomicUtils.cuh"
#include "comms/prims/collectives/DirectCollectiveUtils.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims {

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

  const char* tile_src = args.sendbuf + tile_offset;
  char* own_dst = args.recvbuf + static_cast<std::size_t>(my_rank) * sendcount +
      tile_offset;
  memcpy_vectorized(own_dst, tile_src, tile_bytes, group);
  group.sync();

  if (W <= 1 || tile_bytes == 0) {
    return;
  }

  const std::size_t pipeline_window =
      direct_pipeline_window(args.peers, my_rank, W);
  PIPES_DEVICE_CHECK_MSG(
      pipeline_window != 0, "direct NVLink allgather pipeline window is zero");

  for (std::size_t off = 0; off < tile_bytes; off += pipeline_window) {
    const std::size_t remaining = tile_bytes - off;
    const std::size_t window =
        remaining < pipeline_window ? remaining : pipeline_window;

    const char* send_src = tile_src + off;
    for (int peer_rank = 0; peer_rank < W; ++peer_rank) {
      if (peer_rank == my_rank) {
        continue;
      }
      auto peer = args.peers[peer_rank];
      peer.send(group, send_src, window, max_sig, timeout);
    }

    for (int peer_rank = 0; peer_rank < W; ++peer_rank) {
      if (peer_rank == my_rank) {
        continue;
      }
      char* dst = args.recvbuf + peer_rank * sendcount + tile_offset + off;
      auto peer = args.peers[peer_rank];
      peer.recv(group, dst, window, max_sig, timeout);
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

  char* own_dst = args.recvbuf +
      (static_cast<std::size_t>(args.ib_rank) * args.nvl_size + args.nvl_rank) *
          chunk_bytes +
      io_tile_offset;
  const char* tile_src = args.sendbuf + io_tile_offset;

  // Single IB node (W==1): self-copy then NVL broadcast within the local
  // NVL group. The broadcast is still needed because each NVL peer must
  // receive every other peer's chunk even when there is only one IB node.
  if (W <= 1) {
    memcpy_vectorized(own_dst, tile_src, io_tile_bytes, group);
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

    const char* send_src = tile_src + off;
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

namespace {

__device__ __forceinline__ std::size_t ceil_div(
    std::size_t numerator,
    std::size_t denominator) {
  return denominator == 0 ? 0 : (numerator + denominator - 1) / denominator;
}

__device__ __forceinline__ ThreadGroup make_sub_block_group(
    const ThreadGroup& group,
    uint32_t group_id,
    uint32_t total_groups) {
  return ThreadGroup{
      .thread_id_in_group = group.thread_id_in_group,
      .group_size = group.group_size,
      .group_id = group_id,
      .block_id = group.block_id,
      .total_groups = total_groups,
      .scope = group.scope};
}

__device__ __forceinline__ uint64_t load_ready_counter(const uint64_t* ptr) {
  return comms::device::ld_acquire_sys_global(ptr);
}

__device__ __forceinline__ void publish_ready(
    ThreadGroup& group,
    uint64_t* ready_counters,
    std::size_t idx,
    uint64_t sequence) {
  __threadfence();
  group.sync();
  if (group.is_leader()) {
    comms::device::st_release_sys_global(ready_counters + idx, sequence);
  }
  group.sync();
}

__device__ __forceinline__ void wait_ready(
    ThreadGroup& group,
    const uint64_t* ready_counters,
    std::size_t idx,
    uint64_t sequence,
    const Timeout& timeout) {
  while (load_ready_counter(ready_counters + idx) != sequence) {
    TIMEOUT_TRAP_IF_EXPIRED(
        timeout,
        group,
        "hierarchical allgather waiting for ready counter idx=%llu sequence=%llu",
        static_cast<unsigned long long>(idx),
        static_cast<unsigned long long>(sequence));
  }
  group.sync();
}

__device__ __forceinline__ void trace_hierarchical_allgather(
    PipesTraceHandle trace,
    const ThreadGroup& group,
    PipesTraceEventType type,
    std::size_t chunk,
    int rank) {
  if (!group.is_leader()) {
    return;
  }
  write_pipes_trace(
      trace,
      type,
      static_cast<uint32_t>(chunk),
      static_cast<uint16_t>(group.group_id),
      static_cast<uint8_t>(rank));
}

} // namespace

template <int kBlockSize>
__global__
__launch_bounds__(kBlockSize, 1) void hierarchical_allgather_overlap_kernel(
    const __grid_constant__ HierarchicalAllgatherOverlapArgs args,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  timeout.start();

  auto base_group = make_block_group();
  const std::size_t chunk_bytes = args.chunk_bytes;
  const std::size_t total_chunks = ceil_div(args.sendcount, chunk_bytes);
  if (chunk_bytes == 0 || total_chunks == 0) {
    return;
  }

  const bool is_ib_block = base_group.group_id < args.ib_num_blocks;
  if (is_ib_block) {
    auto group = make_sub_block_group(
        base_group, base_group.group_id, args.ib_num_blocks);
    const int W = args.ib_size;

    for (std::size_t chunk = group.group_id; chunk < total_chunks;
         chunk += group.total_groups) {
      trace_hierarchical_allgather(
          args.trace,
          group,
          PipesTraceEventType::kHierAgIbChunkBegin,
          chunk,
          args.ib_rank);
      const std::size_t off = chunk * chunk_bytes;
      const std::size_t bytes = (off + chunk_bytes <= args.sendcount)
          ? chunk_bytes
          : (args.sendcount - off);
      char* own_dst = args.recvbuf +
          (static_cast<std::size_t>(args.ib_rank) * args.nvl_size +
           args.nvl_rank) *
              args.sendcount +
          off;
      const char* send_src = args.sendbuf + off;

      if (W <= 1) {
        memcpy_vectorized(own_dst, send_src, bytes, group);
        publish_ready(
            group,
            args.ready_counters,
            static_cast<std::size_t>(args.ib_rank) * total_chunks + chunk,
            args.ready_sequence);
        trace_hierarchical_allgather(
            args.trace,
            group,
            PipesTraceEventType::kHierAgIbChunkReady,
            chunk,
            args.ib_rank);
        continue;
      }

      const auto& topo = args.ib_ring;
      auto& prev = *topo.prev;
      auto& next = *topo.next;
      const int stride = (args.ib_rank - topo.prev_rank + W) % W;
      const std::size_t ib_window = next.pipeline_window(args.ib_num_blocks);
      PIPES_DEVICE_CHECK_MSG(
          ib_window != 0,
          "hierarchical allgather overlap IB pipeline window is zero");

      for (std::size_t chunk_off = 0; chunk_off < bytes;
           chunk_off += ib_window) {
        const std::size_t remaining = bytes - chunk_off;
        const std::size_t window =
            remaining < ib_window ? remaining : ib_window;

        next.template sendWithTrace<MemcpyAndSelfCopy>(
            group,
            send_src + chunk_off,
            window,
            args.ib_num_blocks,
            args.ib_signaling_data_size,
            timeout,
            args.trace,
            static_cast<uint8_t>(args.ib_rank),
            own_dst + chunk_off);

        int fwd_current_rank = args.ib_rank;
        for (int step = 0; step < W - 1; step++) {
          fwd_current_rank = (fwd_current_rank + W - stride) % W;
          char* dst = args.recvbuf +
              (static_cast<std::size_t>(fwd_current_rank) * args.nvl_size +
               args.nvl_rank) *
                  args.sendcount +
              off + chunk_off;

          if (step < W - 2) {
            prev.forwardWithTrace(
                group,
                dst,
                next,
                window,
                args.ib_num_blocks,
                args.ib_signaling_data_size,
                timeout,
                args.trace,
                static_cast<uint8_t>(args.ib_rank));
          } else {
            prev.recvWithTrace(
                group,
                dst,
                window,
                args.ib_num_blocks,
                args.ib_signaling_data_size,
                timeout,
                args.trace,
                static_cast<uint8_t>(args.ib_rank));
          }
        }
      }

      publish_ready(
          group,
          args.ready_counters,
          static_cast<std::size_t>(args.ib_rank) * total_chunks + chunk,
          args.ready_sequence);

      int fwd_ready_rank = args.ib_rank;
      for (int step = 0; step < W - 1; step++) {
        fwd_ready_rank = (fwd_ready_rank + W - stride) % W;
        publish_ready(
            group,
            args.ready_counters,
            static_cast<std::size_t>(fwd_ready_rank) * total_chunks + chunk,
            args.ready_sequence);
      }
      trace_hierarchical_allgather(
          args.trace,
          group,
          PipesTraceEventType::kHierAgIbChunkReady,
          chunk,
          args.ib_rank);
    }
    return;
  }

  if (args.nvl_size <= 1) {
    return;
  }

  const uint32_t nvl_group_id = base_group.group_id - args.ib_num_blocks;
  auto group =
      make_sub_block_group(base_group, nvl_group_id, args.nvl_num_blocks);
  const std::size_t total_tasks =
      static_cast<std::size_t>(args.ib_size) * total_chunks;
  const std::size_t nvl_window =
      direct_pipeline_window(args.nvl_peers, args.nvl_rank, args.nvl_size);
  PIPES_DEVICE_CHECK_MSG(
      nvl_window != 0,
      "hierarchical allgather overlap NVLink pipeline window is zero");

  for (std::size_t task = group.group_id; task < total_tasks;
       task += group.total_groups) {
    const std::size_t chunk = task / static_cast<std::size_t>(args.ib_size);
    const int ib_src =
        static_cast<int>(task % static_cast<std::size_t>(args.ib_size));
    const std::size_t off = chunk * chunk_bytes;
    const std::size_t bytes = (off + chunk_bytes <= args.sendcount)
        ? chunk_bytes
        : (args.sendcount - off);

    trace_hierarchical_allgather(
        args.trace,
        group,
        PipesTraceEventType::kHierAgNvlWaitBegin,
        chunk,
        ib_src);
    wait_ready(
        group,
        args.ready_counters,
        static_cast<std::size_t>(ib_src) * total_chunks + chunk,
        args.ready_sequence,
        timeout);
    trace_hierarchical_allgather(
        args.trace,
        group,
        PipesTraceEventType::kHierAgNvlChunkReady,
        chunk,
        ib_src);

    const char* send_src = args.recvbuf +
        (static_cast<std::size_t>(ib_src) * args.nvl_size + args.nvl_rank) *
            args.sendcount +
        off;
    for (std::size_t chunk_off = 0; chunk_off < bytes;
         chunk_off += nvl_window) {
      const std::size_t remaining = bytes - chunk_off;
      const std::size_t window =
          remaining < nvl_window ? remaining : nvl_window;

      for (int peer_rank = 0; peer_rank < args.nvl_size; ++peer_rank) {
        if (peer_rank == args.nvl_rank) {
          continue;
        }
        auto peer = args.nvl_peers[peer_rank];
        peer.send(
            group,
            send_src + chunk_off,
            window,
            args.nvl_signaling_data_size,
            timeout);
      }

      for (int peer_rank = 0; peer_rank < args.nvl_size; ++peer_rank) {
        if (peer_rank == args.nvl_rank) {
          continue;
        }
        char* dst = args.recvbuf +
            (static_cast<std::size_t>(ib_src) * args.nvl_size + peer_rank) *
                args.sendcount +
            off + chunk_off;
        auto peer = args.nvl_peers[peer_rank];
        peer.recv(group, dst, window, args.nvl_signaling_data_size, timeout);
      }
    }
    trace_hierarchical_allgather(
        args.trace,
        group,
        PipesTraceEventType::kHierAgNvlTaskDone,
        chunk,
        ib_src);
  }
#endif
}

template __global__ void direct_allgather_nvl_kernel<512>(
    const __grid_constant__ DirectAllgatherNvlArgs,
    Timeout);

template __global__ void hierarchical_allgather_fused_kernel<512>(
    const __grid_constant__ HierarchicalAllgatherFusedArgs,
    Timeout);

template __global__ void hierarchical_allgather_overlap_kernel<512>(
    const __grid_constant__ HierarchicalAllgatherOverlapArgs,
    Timeout);

} // namespace comms::prims
