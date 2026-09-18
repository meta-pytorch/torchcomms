// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <atomic>

#include "comms/prims/collectives/ReduceScatterDirectIb.cuh"
#include "comms/prims/collectives/ReduceScatterDirectIbCore.cuh"

#include "comms/prims/core/Checks.h"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/QuantCopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/transport/P2pIbTransportDevice.cuh"

namespace comms::prims {

template <
    bool kQuantized,
    bool kTmaRecv,
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
    AbortDevice abortDevice) {
#ifdef __CUDA_ARCH__
  abortDevice.start();

  static_assert(kSendThreads % comms::device::kWarpSize == 0);
  static_assert(kRecvThreads % comms::device::kWarpSize == 0);
  static_assert(kSendThreads + kRecvThreads == kBlockSize);

  using QuantOp = QuantizedReduceScatterCopyOpT<kTmaRecv>;

  const ThreadGroup block = make_block_group();
  const bool is_recv = block.thread_id_in_group < kRecvThreads;
  ThreadGroup group = is_recv
      ? ThreadGroup{
            .thread_id_in_group = block.thread_id_in_group,
            .group_size = kRecvThreads,
            .group_id = block.group_id,
            .block_id = block.block_id,
            .total_groups = block.total_groups,
            .scope = SyncScope::MULTIWARP,
            .barrier_id = kQuantized ? 1 : ThreadGroup::kAutoBarrierId}
      : ThreadGroup{
            .thread_id_in_group = block.thread_id_in_group - kRecvThreads,
            .group_size = kSendThreads,
            .group_id = block.group_id,
            .block_id = block.block_id,
            .total_groups = block.total_groups,
            .scope = SyncScope::MULTIWARP,
            .barrier_id = kQuantized ? 2 : ThreadGroup::kAutoBarrierId};

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
      const int peer = direct_ib_reduce_scatter_peer_for_step(
          my_rank,
          W,
          channel,
          step,
          DirectIbReduceScatterRole::RECEIVE,
          kStaggerChannels);
      const char* local_input = !args.in_place && step == 0
          ? reinterpret_cast<const char*>(own_src)
          : output_bytes;
      auto transport = args.peers[peer];
      if constexpr (kQuantized) {
        const std::size_t wire_bytes =
            output_tile.tile_size(channel) * sizeof(__nv_bfloat16);
        typename QuantOp::Args copy_args{
            .sender_input_base = nullptr,
            .receiver_input_base = reinterpret_cast<const T*>(local_input),
            .receiver_output_base = output,
            .seed = *args.seed_ptr,
            .logical_element_base = 0,
        };
        transport.template recv<QuantOp>(
            group, output_bytes, wire_bytes, max_sig, abortDevice, copy_args);
      } else {
        transport.template recv<ReduceOp>(
            group, output_bytes, tile_bytes, max_sig, abortDevice, local_input);
      }
    }
  } else {
    if (W <= 1) {
      return;
    }

    for (int step = 0; step < W - 1; ++step) {
      const int peer = direct_ib_reduce_scatter_peer_for_step(
          my_rank,
          W,
          channel,
          step,
          DirectIbReduceScatterRole::SEND,
          kStaggerChannels);
      TiledBuffer<const T> send_tile(
          input_base + static_cast<std::size_t>(peer) * args.chunk_elements,
          args.chunk_elements,
          group);
      auto transport = args.peers[peer];
      if constexpr (kQuantized) {
        const std::size_t wire_bytes =
            send_tile.tile_size(channel) * sizeof(__nv_bfloat16);
        const std::uint64_t total_elements =
            static_cast<std::uint64_t>(args.chunk_elements) *
            static_cast<std::uint64_t>(W);
        const std::uint64_t logical_element_base =
            static_cast<std::uint64_t>(my_rank) * total_elements +
            static_cast<std::uint64_t>(peer) * args.chunk_elements +
            static_cast<std::uint64_t>(channel) * send_tile.tile_elements;
        typename QuantOp::Args copy_args{
            .sender_input_base = send_tile.data(),
            .receiver_input_base = nullptr,
            .receiver_output_base = nullptr,
            .seed = *args.seed_ptr,
            .logical_element_base = logical_element_base,
        };
        transport.template send<QuantOp>(
            group,
            reinterpret_cast<const char*>(send_tile.data()),
            wire_bytes,
            max_sig,
            abortDevice,
            copy_args);
      } else {
        transport.send(
            group,
            reinterpret_cast<const char*>(send_tile.data()),
            send_tile.bytes(),
            max_sig,
            abortDevice);
      }
    }
  }
#endif
}

template __global__ void direct_reduce_scatter_ib_kernel<
    false,
    false,
    true,
    float,
    SumOp,
    128,
    384,
    512,
    CpAsyncSmemReduce<float, SumOp, 8192, 384, 2>>(
    const __grid_constant__ DirectReduceScatterIbArgs<float>,
    AbortDevice);

void launch_direct_reduce_scatter_ib_impl(
    const DirectReduceScatterIbArgs<float>& args,
    int num_blocks,
    cudaStream_t stream,
    AbortDevice abortDevice) {
  auto* kernel = direct_reduce_scatter_ib_kernel<
      false,
      false,
      true,
      float,
      SumOp,
      128,
      384,
      512,
      CpAsyncSmemReduce<float, SumOp, 8192, 384, 2>>;
  using ReduceOp = CpAsyncSmemReduce<float, SumOp, 8192, 384, 2>;
  constexpr std::size_t dynamic_smem = ReduceOp::smem_bytes();
  if constexpr (dynamic_smem > 0) {
    int device = 0;
    PIPES_CUDA_CHECK(cudaGetDevice(&device));

    int compute_capability_major = 0;
    PIPES_CUDA_CHECK(cudaDeviceGetAttribute(
        &compute_capability_major, cudaDevAttrComputeCapabilityMajor, device));
    if (compute_capability_major < 8) {
      throw std::runtime_error(
          "CpAsyncSmemReduce requires a GPU with compute capability 8.0 or "
          "newer");
    }

    int max_dynamic_smem = 0;
    PIPES_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (dynamic_smem > static_cast<std::size_t>(max_dynamic_smem)) {
      throw std::runtime_error(
          "CpAsyncSmemReduce requires " + std::to_string(dynamic_smem) +
          " bytes of dynamic shared memory, but the GPU supports only " +
          std::to_string(max_dynamic_smem));
    }

    PIPES_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem)));
  }
  kernel<<<num_blocks, 512, dynamic_smem, stream>>>(args, abortDevice);
  PIPES_CUDA_CHECK(cudaGetLastError());
}

namespace {

// Instantiated once per (TMA, geometry) pair. The TMA receive path stages both
// operands through shared memory, which frees the receive group from issuing
// per-thread loads, so it needs far fewer threads than the register path.
template <bool kTmaRecv, int kSend, int kRecv, int kBlock>
void launch_quantized(
    const DirectReduceScatterIbArgs<float>& args,
    int num_blocks,
    cudaStream_t stream,
    AbortDevice abortDevice) {
  auto* kernel = direct_reduce_scatter_ib_kernel<
      true,
      kTmaRecv,
      true,
      float,
      SumOp,
      kSend,
      kRecv,
      kBlock,
      CpAsyncSmemReduce<float, SumOp, 8192, 384, 2>>;
  constexpr std::size_t dynamic_smem =
      QuantizedReduceScatterCopyOpT<kTmaRecv>::smem_bytes();
  if constexpr (dynamic_smem > 0) {
    // The device's opt-in limit was already checked by
    // tma_supported_on_device() before this specialization was selected.
    PIPES_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem)));
  }
  kernel<<<num_blocks, kBlock, dynamic_smem, stream>>>(args, abortDevice);
  PIPES_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace {

// The TMA receive path is only validated on Blackwell (sm_100+) and its shared
// memory footprint exceeds what earlier architectures can opt into, so it is
// gated on the device rather than on the compile target. Anything else silently
// takes the register path, which is functionally identical.
//
// Cached per device ordinal, not once per process: a process can bind several
// GPUs, and a single cached answer would route a device of a different
// capability onto the wrong geometry. Racing writers compute the same value.
bool tma_supported_on_device() {
  int device = 0;
  PIPES_CUDA_CHECK(cudaGetDevice(&device));

  constexpr int kMaxCachedDevices = 16;
  constexpr signed char kUnknown = 0;
  constexpr signed char kSupported = 1;
  constexpr signed char kUnsupported = 2;
  static std::atomic<signed char> cache[kMaxCachedDevices];

  const bool cacheable = device >= 0 && device < kMaxCachedDevices;
  if (cacheable) {
    const signed char cached = cache[device].load(std::memory_order_relaxed);
    if (cached != kUnknown) {
      return cached == kSupported;
    }
  }

  int cc_major = 0;
  PIPES_CUDA_CHECK(cudaDeviceGetAttribute(
      &cc_major, cudaDevAttrComputeCapabilityMajor, device));
  bool supported = cc_major >= 10;
  if (supported) {
    int max_dynamic_smem = 0;
    PIPES_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    supported = QuantizedReduceScatterCopyOpT<true>::smem_bytes() <=
        static_cast<std::size_t>(max_dynamic_smem);
  }
  if (cacheable) {
    cache[device].store(
        supported ? kSupported : kUnsupported, std::memory_order_relaxed);
  }
  return supported;
}

} // namespace

void launch_direct_reduce_scatter_ib_quantized_impl(
    const DirectReduceScatterIbArgs<float>& args,
    int num_blocks,
    bool use_tma,
    cudaStream_t stream,
    AbortDevice abortDevice) {
  if (use_tma && tma_supported_on_device()) {
    launch_quantized<true, 640, 128, 768>(
        args, num_blocks, stream, abortDevice);
  } else {
    launch_quantized<false, 480, 160, 640>(
        args, num_blocks, stream, abortDevice);
  }
}

} // namespace comms::prims
