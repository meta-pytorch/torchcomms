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

template <int NumWarps, int MinBlocksPerSM>
__global__
__launch_bounds__(NumWarps * 32, MinBlocksPerSM) void ans_copy_op_compress_bench_kernel(
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

template <int NumWarps, int MinBlocksPerSM>
__global__
__launch_bounds__(NumWarps * 32, MinBlocksPerSM) void ans_copy_op_decompress_bench_kernel(
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

// Explicit instantiations for every (NumWarps, MinBlocksPerSM) pair
// accepted by `pick_ans_copy_op_bench_kernels`. NumWarps controls
// blockDim.x (= NumWarps * 32); MinBlocksPerSM is the second
// `__launch_bounds__` argument, which tells ptxas the minimum blocks
// it must keep resident per SM and therefore caps per-thread register
// usage at ~`65536 / (NumWarps * 32 * MinBlocksPerSM)`. Extend this
// macro block (and the switch below) if a wider range is needed.
#define PIPES_INSTANTIATE_BENCH_KERNELS(NUM_WARPS, MIN_BLOCKS)           \
  template __global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS) void \
  ans_copy_op_compress_bench_kernel<NUM_WARPS, MIN_BLOCKS>(              \
      char* d_input,                                                     \
      char* d_staging,                                                   \
      std::size_t input_bytes,                                           \
      std::size_t staging_stride,                                        \
      int iters);                                                        \
  template __global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS) void \
  ans_copy_op_decompress_bench_kernel<NUM_WARPS, MIN_BLOCKS>(            \
      char* d_output,                                                    \
      const char* d_staging,                                             \
      std::size_t input_bytes,                                           \
      std::size_t staging_stride,                                        \
      int iters);

// NumWarps ∈ {1, 2, 4, 8, 16} (blockDim.x ∈ {32, 64, 128, 256, 512})
// × MinBlocksPerSM ∈ {1, 2, 3, 4}.
#define PIPES_INSTANTIATE_BENCH_KERNELS_FOR_WARPS(NUM_WARPS) \
  PIPES_INSTANTIATE_BENCH_KERNELS(NUM_WARPS, 1)              \
  PIPES_INSTANTIATE_BENCH_KERNELS(NUM_WARPS, 2)              \
  PIPES_INSTANTIATE_BENCH_KERNELS(NUM_WARPS, 3)              \
  PIPES_INSTANTIATE_BENCH_KERNELS(NUM_WARPS, 4)

PIPES_INSTANTIATE_BENCH_KERNELS_FOR_WARPS(1)
PIPES_INSTANTIATE_BENCH_KERNELS_FOR_WARPS(2)
PIPES_INSTANTIATE_BENCH_KERNELS_FOR_WARPS(4)
PIPES_INSTANTIATE_BENCH_KERNELS_FOR_WARPS(8)
PIPES_INSTANTIATE_BENCH_KERNELS_FOR_WARPS(16)

// Extra point: 128 threads (NumWarps=4) with MinBlocksPerSM=16 caps registers
// at 65536/(128*16)=32/thread, the value needed to fit 16 blocks/SM
// (= 64 warps = 100% occupancy) at 128 threads/block. Used to measure whether
// forcing 100% occupancy (at the cost of heavy spill) helps the 128-thread
// kernel.
PIPES_INSTANTIATE_BENCH_KERNELS(4, 16)

// 256 threads (NumWarps=8) with MinBlocksPerSM=8 caps registers at
// 65536/(256*8)=32/thread, fitting 8 blocks/SM (= 64 warps = 100% occupancy).
PIPES_INSTANTIATE_BENCH_KERNELS(8, 8)

#undef PIPES_INSTANTIATE_BENCH_KERNELS_FOR_WARPS
#undef PIPES_INSTANTIATE_BENCH_KERNELS

namespace {
// {compress, decompress} kernel-address pair for one explicit
// (NumWarps, MinBlocksPerSM) instantiation.
#define PIPES_BENCH_KERNEL_PAIR(NUM_WARPS, MIN_BLOCKS)                   \
  AnsCopyOpBenchKernels {                                                \
    reinterpret_cast<void*>(                                             \
        &ans_copy_op_compress_bench_kernel<NUM_WARPS, MIN_BLOCKS>),      \
        reinterpret_cast<void*>(                                         \
            &ans_copy_op_decompress_bench_kernel<NUM_WARPS, MIN_BLOCKS>) \
  }

// Inner dispatch on MinBlocksPerSM for a fixed NumWarps. Every arm
// (including `default`) returns, so it is safe to drop into from the
// outer threads_per_block switch.
#define PIPES_BENCH_PICK_MIN_BLOCKS(NUM_WARPS)      \
  switch (min_blocks_per_sm) {                      \
    case 1:                                         \
      return PIPES_BENCH_KERNEL_PAIR(NUM_WARPS, 1); \
    case 2:                                         \
      return PIPES_BENCH_KERNEL_PAIR(NUM_WARPS, 2); \
    case 3:                                         \
      return PIPES_BENCH_KERNEL_PAIR(NUM_WARPS, 3); \
    case 4:                                         \
      return PIPES_BENCH_KERNEL_PAIR(NUM_WARPS, 4); \
    default:                                        \
      return {nullptr, nullptr};                    \
  }
} // namespace

AnsCopyOpBenchKernels pick_ans_copy_op_bench_kernels(
    int threads_per_block,
    int min_blocks_per_sm) {
  switch (threads_per_block) {
    case 32:
      PIPES_BENCH_PICK_MIN_BLOCKS(1);
    case 64:
      PIPES_BENCH_PICK_MIN_BLOCKS(2);
    case 128:
      // m=16 (100% occupancy, 32-reg cap) is instantiated only for
      // NumWarps=4; handle it before the generic 1..4 dispatch.
      if (min_blocks_per_sm == 16) {
        return PIPES_BENCH_KERNEL_PAIR(4, 16);
      }
      PIPES_BENCH_PICK_MIN_BLOCKS(4);
    case 256:
      // m=8 (100% occupancy, 32-reg cap) is instantiated only for
      // NumWarps=8; handle it before the generic 1..4 dispatch.
      if (min_blocks_per_sm == 8) {
        return PIPES_BENCH_KERNEL_PAIR(8, 8);
      }
      PIPES_BENCH_PICK_MIN_BLOCKS(8);
    case 512:
      PIPES_BENCH_PICK_MIN_BLOCKS(16);
    default:
      return {nullptr, nullptr};
  }
}

#undef PIPES_BENCH_PICK_MIN_BLOCKS
#undef PIPES_BENCH_KERNEL_PAIR

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
