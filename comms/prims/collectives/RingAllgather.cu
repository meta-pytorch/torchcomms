// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/RingAllgather.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"

namespace comms::prims {

template <int NumRings, int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void ring_allgather_kernel(
    const __grid_constant__ RingAllgatherArgs<NumRings> args,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  timeout.start();

  auto group = make_block_group();
  auto [ring_id, ring_group] = group.partition(NumRings);
  const auto& topo = args.rings[ring_id];
  auto prev = topo.prev;
  auto next = topo.next;

  const int W = args.num_ranks;
  const std::size_t chunk_bytes = args.sendcount;
  const std::size_t max_sig = args.signaling_data_size;

  const std::size_t base_ring_bytes = chunk_bytes / NumRings;
  const std::size_t ring_bytes = (ring_id < NumRings - 1)
      ? base_ring_bytes
      : (chunk_bytes - (NumRings - 1) * base_ring_bytes);
  const std::size_t ring_offset = ring_id * base_ring_bytes;

  TiledBuffer<char> ring_tile(nullptr, ring_bytes, ring_group);
  const std::size_t io_tile_offset =
      ring_group.group_id * ring_tile.tile_elements;
  const std::size_t io_tile_bytes = ring_tile.bytes();

  const std::size_t pipeline_window = next.pipeline_window();

  const int my_rank = args.my_rank;
  const int stride = (my_rank - topo.prev_rank + W) % W;

  // Local copy: own chunk from sendbuf to recvbuf.
  const char* own_src = args.sendbuf + ring_offset + io_tile_offset;
  char* own_dst =
      args.recvbuf + my_rank * chunk_bytes + ring_offset + io_tile_offset;
  memcpy_vectorized(own_dst, own_src, io_tile_bytes, ring_group);
  ring_group.sync();

  for (std::size_t off = 0; off < io_tile_bytes; off += pipeline_window) {
    const std::size_t remaining = io_tile_bytes - off;
    const std::size_t window =
        (remaining < pipeline_window) ? remaining : pipeline_window;

    // Step 0: Send own chunk to next.
    char* send_src = args.recvbuf + my_rank * chunk_bytes + ring_offset +
        io_tile_offset + off;
    next.send(group, send_src, window, max_sig, timeout);

    // Steps 1..W-1: receive and forward (or just receive on last step).
    int current_rank = my_rank;
    for (int step = 0; step < W - 1; step++) {
      current_rank = (current_rank + W - stride) % W;
      char* dst = args.recvbuf + current_rank * chunk_bytes + ring_offset +
          io_tile_offset + off;

      if (step < W - 2) {
        prev.forward(group, dst, next, window, max_sig, timeout);
      } else {
        prev.recv(group, dst, window, max_sig, timeout);
      }
    }
  }
#endif
}

// Template instantiations
template __global__ void ring_allgather_kernel<1, 512>(
    const __grid_constant__ RingAllgatherArgs<1>,
    Timeout);
template __global__ void ring_allgather_kernel<2, 512>(
    const __grid_constant__ RingAllgatherArgs<2>,
    Timeout);
template __global__ void ring_allgather_kernel<4, 512>(
    const __grid_constant__ RingAllgatherArgs<4>,
    Timeout);

} // namespace comms::prims
