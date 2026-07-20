// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Definition of ring_reduce_scatter_kernel, factored out of a single .cu so the
// explicit template instantiations can be split across one translation unit per
// datatype (RingReduceScatterInst_*.cu). This keeps each nvcc compile action
// small and parallel; compiling all datatype x op x ring-count instantiations
// in one TU is a major build-speed regression. Include this ONLY from the
// per-datatype instantiation .cu files -- not from the launcher, which needs
// only the declaration in RingReduceScatter.cuh.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "comms/prims/collectives/RingReduceScatter.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"

namespace comms::prims {

template <
    int NumRings,
    typename T,
    typename AccumOp,
    int kTileElems,
    int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void ring_reduce_scatter_kernel(
    const __grid_constant__ RingReduceScatterArgs<NumRings, T> args,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  timeout.start();

  auto group = make_block_group();
  auto [ring_id, ring_group] = group.partition(NumRings);
  const auto& topo = args.rings[ring_id];
  auto prev = topo.prev;
  auto next = topo.next;

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

  const std::size_t pipeline_window = next.pipeline_window();

  const int my_rank = args.my_rank;
  const int stride = (my_rank - topo.prev_rank + W) % W;
  const char* input_base = reinterpret_cast<const char*>(args.input);

  using ReduceOp = TileReduceStaged<T, AccumOp, kTileElems, kBlockSize>;

  for (std::size_t off = 0; off < io_tile_bytes; off += pipeline_window) {
    const std::size_t remaining = io_tile_bytes - off;
    const std::size_t window =
        (remaining < pipeline_window) ? remaining : pipeline_window;

    int current_rank = (my_rank + W - stride) % W;

    // Step 0: Send raw input chunk to next.
    const char* send_src = input_base + current_rank * chunk_bytes +
        ring_offset + io_tile_offset + off;
    next.send(group, send_src, window, max_sig, timeout);

    // W-1 receive steps: forward for intermediate, recv for final.
    for (int step = 0; step < W - 1; step++) {
      current_rank = (current_rank + W - stride) % W;
      const char* local_input = input_base + current_rank * chunk_bytes +
          ring_offset + io_tile_offset + off;

      if (step < W - 2) {
        prev.template forward<ReduceOp>(
            group, nullptr, next, window, max_sig, timeout, local_input);
      } else {
        char* dst = reinterpret_cast<char*>(args.output) + ring_offset +
            io_tile_offset + off;
        prev.template recv<ReduceOp>(
            group, dst, window, max_sig, timeout, local_input);
      }
    }
  }
#endif
}

} // namespace comms::prims

// Explicit-instantiation macros. Invoke
// INSTANTIATE_RING_REDUCE_SCATTER_FOR_TYPE inside `namespace comms::prims { ...
// }` from a per-datatype .cu. The instantiated set (NumRings in {1,2,4} x
// {SumOp,MaxOp,MinOp}, tile 16384, block 512) is identical to the previous
// single-file instantiation block.
#define INSTANTIATE_RING_REDUCE_SCATTER(num_rings, type, op)   \
  template __global__ void                                     \
  ring_reduce_scatter_kernel<num_rings, type, op, 16384, 512>( \
      const __grid_constant__ RingReduceScatterArgs<num_rings, type>, Timeout)

#define INSTANTIATE_RING_REDUCE_SCATTER_OPS(num_rings, type) \
  INSTANTIATE_RING_REDUCE_SCATTER(num_rings, type, SumOp);   \
  INSTANTIATE_RING_REDUCE_SCATTER(num_rings, type, MaxOp);   \
  INSTANTIATE_RING_REDUCE_SCATTER(num_rings, type, MinOp)

#define INSTANTIATE_RING_REDUCE_SCATTER_FOR_TYPE(type) \
  INSTANTIATE_RING_REDUCE_SCATTER_OPS(1, type);        \
  INSTANTIATE_RING_REDUCE_SCATTER_OPS(2, type);        \
  INSTANTIATE_RING_REDUCE_SCATTER_OPS(4, type)
