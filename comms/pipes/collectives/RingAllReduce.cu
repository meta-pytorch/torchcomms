// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/pipes/collectives/RingAllReduce.cuh"

namespace comms::pipes {

template <
    int NumRings,
    typename T,
    typename AccumOp,
    int kTileElems,
    int kBlockSize,
    bool EnableBidirAg>
__global__ __launch_bounds__(kBlockSize, 1) void ring_allreduce_kernel(
    const __grid_constant__ RingAllReduceArgs<NumRings, T> args,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  timeout.start();

  auto group = make_block_group();
  auto [ring_id, ring_group] = group.partition(NumRings);
  const auto& topo = args.rings[ring_id];
  auto& prev = *topo.prev;
  auto& next = *topo.next;

  const int W = args.num_ranks;
  const std::size_t chunk_elems = args.chunk_elements;
  const std::size_t chunk_bytes = chunk_elems * sizeof(T);
  const std::size_t max_sig = args.signaling_data_size;

  const std::size_t base_ring_elems = chunk_elems / NumRings;
  const std::size_t ring_elems = (ring_id < NumRings - 1)
      ? base_ring_elems
      : (chunk_elems - (NumRings - 1) * base_ring_elems);
  const std::size_t ring_bytes = ring_elems * sizeof(T);
  const std::size_t ring_offset = ring_id * base_ring_elems * sizeof(T);

  TiledBuffer<char> ring_tile(nullptr, ring_bytes, ring_group);
  const std::size_t io_tile_offset =
      ring_group.group_id * ring_tile.tile_elements;
  const std::size_t io_tile_bytes = ring_tile.bytes();

  const std::size_t pipeline_window = next.pipeline_window(group.total_groups);

  const int my_rank = args.my_rank;
  const int stride = (my_rank - topo.prev_rank + W) % W;

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 1: REDUCE-SCATTER (TileReduceStaged CopyOp)
  // ═══════════════════════════════════════════════════════════════════
  {
    const char* input_base = reinterpret_cast<const char*>(args.input);
    using ReduceOp = TileReduceStaged<T, AccumOp, kTileElems, kBlockSize>;

    for (std::size_t off = 0; off < io_tile_bytes; off += pipeline_window) {
      const std::size_t remaining = io_tile_bytes - off;
      const std::size_t window =
          (remaining < pipeline_window) ? remaining : pipeline_window;

      int current_rank = (my_rank + W - stride) % W;

      const char* send_src = input_base + current_rank * chunk_bytes +
          ring_offset + io_tile_offset + off;
      next.send(group, send_src, window, group.total_groups, max_sig, timeout);

      for (int step = 0; step < W - 1; step++) {
        current_rank = (current_rank + W - stride) % W;
        const char* local_input = input_base + current_rank * chunk_bytes +
            ring_offset + io_tile_offset + off;

        if (step < W - 2) {
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
          char* dst = reinterpret_cast<char*>(args.output) + ring_offset +
              io_tile_offset + off;
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
  }

  // ═══════════════════════════════════════════════════════════════════
  // PHASE 2: ALL-GATHER (Memcpy CopyOp)
  // ═══════════════════════════════════════════════════════════════════
  {
    char* output_base = reinterpret_cast<char*>(args.output);

    // Local copy: reduced shard → correct position in output.
    const char* own_src = output_base + ring_offset + io_tile_offset;
    char* own_dst =
        output_base + my_rank * chunk_bytes + ring_offset + io_tile_offset;
    if (own_src != own_dst) {
      memcpy_vectorized(own_dst, own_src, io_tile_bytes, ring_group);
    }
    ring_group.sync();

    const int num_fwd_recvs = EnableBidirAg ? (W - 1 + 1) / 2 : (W - 1);
    const int num_rev_recvs = EnableBidirAg ? (W - 1) / 2 : 0;

    for (std::size_t off = 0; off < io_tile_bytes; off += pipeline_window) {
      const std::size_t remaining = io_tile_bytes - off;
      const std::size_t window =
          (remaining < pipeline_window) ? remaining : pipeline_window;

      char* send_src = output_base + my_rank * chunk_bytes + ring_offset +
          io_tile_offset + off;
      next.send(group, send_src, window, group.total_groups, max_sig, timeout);
      if constexpr (EnableBidirAg) {
        if (num_rev_recvs > 0) {
          prev.send(
              group, send_src, window, group.total_groups, max_sig, timeout);
        }
      }

      int fwd_current = my_rank;
      int rev_current = my_rank;

      for (int step = 0; step < num_fwd_recvs; step++) {
        fwd_current = (fwd_current + W - stride) % W;
        char* fwd_dst = output_base + fwd_current * chunk_bytes + ring_offset +
            io_tile_offset + off;
        if (step < num_fwd_recvs - 1) {
          prev.forward(
              group,
              fwd_dst,
              next,
              window,
              group.total_groups,
              max_sig,
              timeout);
        } else {
          prev.recv(
              group, fwd_dst, window, group.total_groups, max_sig, timeout);
        }

        if constexpr (EnableBidirAg) {
          if (step < num_rev_recvs) {
            rev_current = (rev_current + stride) % W;
            char* rev_dst = output_base + rev_current * chunk_bytes +
                ring_offset + io_tile_offset + off;
            if (step < num_rev_recvs - 1) {
              next.forward(
                  group,
                  rev_dst,
                  prev,
                  window,
                  group.total_groups,
                  max_sig,
                  timeout);
            } else {
              next.recv(
                  group, rev_dst, window, group.total_groups, max_sig, timeout);
            }
          }
        }
      }
    }
  }
#endif
}

// Template instantiations: EnableBidirAg = false (default)
template __global__ void ring_allreduce_kernel<1, float, SumOp, 16384, 512>(
    const __grid_constant__ RingAllReduceArgs<1, float>,
    Timeout);
template __global__ void ring_allreduce_kernel<2, float, SumOp, 16384, 512>(
    const __grid_constant__ RingAllReduceArgs<2, float>,
    Timeout);
template __global__ void ring_allreduce_kernel<4, float, SumOp, 16384, 512>(
    const __grid_constant__ RingAllReduceArgs<4, float>,
    Timeout);

// Template instantiations: EnableBidirAg = true
template __global__ void
ring_allreduce_kernel<1, float, SumOp, 16384, 512, true>(
    const __grid_constant__ RingAllReduceArgs<1, float>,
    Timeout);
template __global__ void
ring_allreduce_kernel<2, float, SumOp, 16384, 512, true>(
    const __grid_constant__ RingAllReduceArgs<2, float>,
    Timeout);
template __global__ void
ring_allreduce_kernel<4, float, SumOp, 16384, 512, true>(
    const __grid_constant__ RingAllReduceArgs<4, float>,
    Timeout);

} // namespace comms::pipes
