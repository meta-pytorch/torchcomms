// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Standalone microbenchmark kernels for `AnsCompress::send()` /
// `::recv()`. See the matching header (`AnsCopyOpBench.cuh`) for the
// "why this lives in its own TU + dlink target" rationale.
//
// Built with `--device-c` + `PIPES_ENABLE_ANS_COMPRESSION` so that
// `AnsCompress<...>` instantiates against nvcompdx, and device-linked
// against `:nvcompdx_fatbin` via `gen_alltoallv_tile_dlink_cmd` in
// the sibling `:ans_copy_op_bench` BUCK target.

#include <cstddef>

#include "comms/prims/collectives/benchmarks/AnsCopyOpBench.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims {

namespace {

// Pin the per-piece chunking constant to the same value the
// production IBGDA dispatcher uses. Sourced from CopyOp.cuh's
// `PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES` macro (256 KiB) so the two
// stay in lockstep without re-declaring the literal.
constexpr std::size_t kBenchAnsMaxUncompBytes =
    PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES;

} // namespace

template <int NumWarps>
__global__
__launch_bounds__(NumWarps * 32, 1) void ans_copy_op_compress_bench_kernel(
    char* d_input,
    char* d_staging,
    std::size_t input_bytes,
    std::size_t staging_stride,
    int iters) {
  auto group = make_block_group();
  char* my_input = d_input + static_cast<std::size_t>(blockIdx.x) * input_bytes;
  char* my_staging =
      d_staging + static_cast<std::size_t>(blockIdx.x) * staging_stride;

  // `kSrcAligned=false` (template default) so this matches the
  // production IBGDA `ibgda_send_compressed` instantiation for the
  // same NumWarps; sharing the instantiation across TUs keeps
  // `s_compress_shmem` from doubling when LTO link-time pools static
  // `__shared__`. The runtime realign branch inside `send()` is dead
  // here because `my_input = base + blockIdx.x * input_bytes` is
  // always 16-byte aligned, so `nullptr` for `alignedAuxBuf` is safe.
  using Comp = AnsCompress<NumWarps, kBenchAnsMaxUncompBytes>;

  for (int i = 0; i < iters; ++i) {
    (void)Comp::send(
        my_staging,
        my_input,
        input_bytes,
        group,
        /*byte_offset=*/0,
        /*alignedAuxBuf=*/static_cast<char*>(nullptr));
    group.sync();
  }
}

template <int NumWarps>
__global__
__launch_bounds__(NumWarps * 32, 1) void ans_copy_op_decompress_bench_kernel(
    char* d_output,
    const char* d_staging,
    std::size_t input_bytes,
    std::size_t staging_stride,
    int iters) {
  auto group = make_block_group();
  char* my_output =
      d_output + static_cast<std::size_t>(blockIdx.x) * input_bytes;
  const char* my_staging =
      d_staging + static_cast<std::size_t>(blockIdx.x) * staging_stride;

  using Comp = AnsCompress<NumWarps, kBenchAnsMaxUncompBytes>;

  for (int i = 0; i < iters; ++i) {
    (void)Comp::recv(
        my_output,
        my_staging,
        input_bytes,
        group,
        /*byte_offset=*/0);
    group.sync();
  }
}

// Explicit instantiations for every NumWarps accepted by
// `pick_ans_copy_op_bench_kernels`. Extend this macro block (and the
// switch below) if a wider blockDim.x range is needed.
#define PIPES_INSTANTIATE_BENCH_KERNELS(NUM_WARPS)              \
  template __global__ __launch_bounds__(NUM_WARPS * 32, 1) void \
  ans_copy_op_compress_bench_kernel<NUM_WARPS>(                 \
      char* d_input,                                            \
      char* d_staging,                                          \
      std::size_t input_bytes,                                  \
      std::size_t staging_stride,                               \
      int iters);                                               \
  template __global__ __launch_bounds__(NUM_WARPS * 32, 1) void \
  ans_copy_op_decompress_bench_kernel<NUM_WARPS>(               \
      char* d_output,                                           \
      const char* d_staging,                                    \
      std::size_t input_bytes,                                  \
      std::size_t staging_stride,                               \
      int iters);

PIPES_INSTANTIATE_BENCH_KERNELS(1)
PIPES_INSTANTIATE_BENCH_KERNELS(2)
PIPES_INSTANTIATE_BENCH_KERNELS(4)
PIPES_INSTANTIATE_BENCH_KERNELS(8)
PIPES_INSTANTIATE_BENCH_KERNELS(16)

#undef PIPES_INSTANTIATE_BENCH_KERNELS

AnsCopyOpBenchKernels pick_ans_copy_op_bench_kernels(int threads_per_block) {
  switch (threads_per_block) {
    case 32:
      return {
          reinterpret_cast<void*>(&ans_copy_op_compress_bench_kernel<1>),
          reinterpret_cast<void*>(&ans_copy_op_decompress_bench_kernel<1>)};
    case 64:
      return {
          reinterpret_cast<void*>(&ans_copy_op_compress_bench_kernel<2>),
          reinterpret_cast<void*>(&ans_copy_op_decompress_bench_kernel<2>)};
    case 128:
      return {
          reinterpret_cast<void*>(&ans_copy_op_compress_bench_kernel<4>),
          reinterpret_cast<void*>(&ans_copy_op_decompress_bench_kernel<4>)};
    case 256:
      return {
          reinterpret_cast<void*>(&ans_copy_op_compress_bench_kernel<8>),
          reinterpret_cast<void*>(&ans_copy_op_decompress_bench_kernel<8>)};
    case 512:
      return {
          reinterpret_cast<void*>(&ans_copy_op_compress_bench_kernel<16>),
          reinterpret_cast<void*>(&ans_copy_op_decompress_bench_kernel<16>)};
    default:
      return {nullptr, nullptr};
  }
}

std::size_t ans_copy_op_bench_staging_stride(std::size_t input_bytes) {
  // Closed-form upper bound on
  //   AnsCompress<N, kBenchAnsMaxUncompBytes>::worst_case_chunk_stride(
  //       input_bytes)
  // for any supported NumWarps N. Doesn't require nvcompdx's
  // `max_comp_chunk_size()` to be host-callable. nvcompdx's
  // documented ANS worst-case output is approximately
  // `1.3 * MaxUncompChunkSize + 1576` bytes per piece; we use
  // `1.4 * MaxUncompChunkSize + 4096` for headroom, then pad per
  // piece to 256 bytes (nvCOMPDx decompress input alignment) and
  // pad the total to the 512-byte NIC-burst boundary.
  if (input_bytes == 0) {
    return 0;
  }
  constexpr std::size_t kMaxUncomp = kBenchAnsMaxUncompBytes;
  constexpr std::size_t kInputAlign = 256ULL;
  constexpr std::size_t kBurstAlign = 512ULL;
  const std::size_t numChunks = (input_bytes + kMaxUncomp - 1ULL) / kMaxUncomp;
  constexpr std::size_t kPerPieceUnaligned =
      (kMaxUncomp * 14ULL) / 10ULL + 4096ULL;
  constexpr std::size_t kPerPiecePadded =
      (kPerPieceUnaligned + (kInputAlign - 1ULL)) & ~(kInputAlign - 1ULL);
  const std::size_t headerBytes = numChunks * sizeof(std::size_t);
  const std::size_t headerPadded =
      (headerBytes + (kInputAlign - 1ULL)) & ~(kInputAlign - 1ULL);
  const std::size_t total = headerPadded + numChunks * kPerPiecePadded;
  const std::size_t aligned =
      (total + (kBurstAlign - 1ULL)) & ~(kBurstAlign - 1ULL);
  return input_bytes > aligned ? input_bytes : aligned;
}

} // namespace comms::prims
