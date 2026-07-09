// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/ReduceScatterDirectIb.cuh"

#include "comms/prims/core/Checks.h"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/transport/P2pIbTransportDevice.cuh"

namespace comms::prims {

template <
    bool kStaggerChannels,
    typename T,
    typename AccumOp,
    int kSendThreads,
    int kRecvThreads,
    int kBlockSize,
    typename ReduceOp>
__global__
__launch_bounds__(kBlockSize, 1) void direct_reduce_scatter_ib_kernel(
    const __grid_constant__ DirectReduceScatterIbArgs<T> args,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  timeout.start();

  static_assert(kSendThreads % comms::device::kWarpSize == 0);
  static_assert(kRecvThreads % comms::device::kWarpSize == 0);
  static_assert(kSendThreads + kRecvThreads == kBlockSize);

  const ThreadGroup block = make_block_group();
  const bool is_recv = block.thread_id_in_group < kRecvThreads;
  ThreadGroup group = is_recv
      ? ThreadGroup{
            .thread_id_in_group = block.thread_id_in_group,
            .group_size = kRecvThreads,
            .group_id = block.group_id,
            .block_id = block.block_id,
            .total_groups = block.total_groups,
            .scope = SyncScope::MULTIWARP}
      : ThreadGroup{
            .thread_id_in_group = block.thread_id_in_group - kRecvThreads,
            .group_size = kSendThreads,
            .group_id = block.group_id,
            .block_id = block.block_id,
            .total_groups = block.total_groups,
            .scope = SyncScope::MULTIWARP};

  const int channels = static_cast<int>(group.total_groups);
  const int channel = static_cast<int>(group.group_id);
  const int my_rank = args.my_rank;
  const int W = args.num_ranks;
  const std::size_t max_sig = args.signaling_data_size;

  T* output_base = args.output;
  const T* input_base = args.input;

  TiledBuffer<T> output_tile(output_base, args.chunk_elements, group);
  T* output = output_tile.data();
  const std::size_t tile_bytes = output_tile.bytes();
  if (tile_bytes == 0) {
    return;
  }
  char* output_bytes = reinterpret_cast<char*>(output);

  if (is_recv) {
    const T* own_src = input_base +
        static_cast<std::size_t>(my_rank) * args.chunk_elements +
        static_cast<std::size_t>(channel) * output_tile.tile_elements;

    if (W <= 1) {
      if (!args.in_place) {
        memcpy_vectorized(
            output_bytes,
            reinterpret_cast<const char*>(own_src),
            tile_bytes,
            group);
      }
      return;
    }

    for (int step = 0; step < W - 1; ++step) {
      const int peer_offset =
          kStaggerChannels ? (step + channel) % (W - 1) : step;
      const int peer = (my_rank + 1 + peer_offset) % W;
      const char* local_input = !args.in_place && step == 0
          ? reinterpret_cast<const char*>(own_src)
          : output_bytes;
      auto transport = args.peers[peer];
      transport.template recv<ReduceOp>(
          group,
          output_bytes,
          tile_bytes,
          channels,
          max_sig,
          timeout,
          local_input);
    }
  } else {
    if (W <= 1) {
      return;
    }

    for (int step = 0; step < W - 1; ++step) {
      const int peer_offset =
          kStaggerChannels ? (step + channel) % (W - 1) : step;
      const int peer = (my_rank + W - 1 - peer_offset) % W;
      TiledBuffer<const T> send_tile(
          input_base + static_cast<std::size_t>(peer) * args.chunk_elements,
          args.chunk_elements,
          group);
      auto transport = args.peers[peer];
      transport.send(
          group,
          reinterpret_cast<const char*>(send_tile.data()),
          send_tile.bytes(),
          channels,
          max_sig,
          timeout);
    }
  }
#endif
}

template __global__ void direct_reduce_scatter_ib_kernel<
    true,
    float,
    SumOp,
    128,
    384,
    512,
    TileReduceStaged<float, SumOp, 24576, 384>>(
    const __grid_constant__ DirectReduceScatterIbArgs<float>,
    Timeout);

void launch_direct_reduce_scatter_ib_impl(
    const DirectReduceScatterIbArgs<float>& args,
    int num_blocks,
    cudaStream_t stream,
    Timeout timeout) {
  auto* kernel = direct_reduce_scatter_ib_kernel<
      true,
      float,
      SumOp,
      128,
      384,
      512,
      TileReduceStaged<float, SumOp, 24576, 384>>;
  using ReduceOp = TileReduceStaged<float, SumOp, 24576, 384>;
  constexpr std::size_t dynamic_smem = ReduceOp::smem_bytes();
  if constexpr (dynamic_smem > 0) {
    PIPES_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem)));
  }
  kernel<<<num_blocks, 512, dynamic_smem, stream>>>(args, timeout);
  PIPES_CUDA_CHECK(cudaGetLastError());
}

} // namespace comms::prims
