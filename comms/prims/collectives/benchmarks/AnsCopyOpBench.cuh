// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Single-GPU standalone microbenchmark for the raw ANS `CopyOp` APIs
// in `comms/prims/core/CopyOp.cuh` (`AnsCompress<NumWarps, ...>::send()` /
// `::recv()`). Lives in its own translation unit / device-link target
// (`:ans_copy_op_bench` in `comms/prims/collectives/benchmarks/BUCK`)
// instead of being bolted onto `:alltoallv_tile_compressed` so that:
//
//   1. The existing `alltoallv_tile_{1d,2d}_compressed_kernel` symbols
//      stay in a `--device-c` TU that does NOT carry any additional
//      static `__shared__` allocations from these bench kernels.
//   2. Host-only callers that don't want the bench symbols (and the
//      extra nvcompdx fatbin device-link they imply) keep a clean
//      `:alltoallv_tile_compressed` dependency.
//
// The bench kernels are templated on `NumWarps = blockDim.x / 32`
// so the bench .cc can pick the matching kernel symbol at runtime
// (via `pick_ans_copy_op_bench_kernels()`) without dragging every
// NumWarps' static `__shared__` scratch into a single kernel's
// footprint.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace comms::prims {

#ifdef __CUDACC__
/**
 * Single-GPU microbenchmark kernel for measuring the raw
 * `AnsCompress::send()` (compression) throughput in isolation — no
 * transport, no inter-block communication, no NVL/IB path. Each
 * threadblock owns its own per-block input region (`input_bytes`
 * bytes at offset `blockIdx.x * input_bytes`) and per-block staging
 * region (`staging_stride` bytes at offset
 * `blockIdx.x * staging_stride`) and loops `iters` times calling
 * `AnsCompress<NumWarps, kAnsMaxUncompBytes>::send()` over those
 * regions.
 *
 * Templated on `NumWarps = blockDim.x / 32` and `MinBlocksPerSM`
 * (the second `__launch_bounds__` argument — the minimum blocks ptxas
 * must keep resident per SM, which caps per-thread register usage).
 * Explicit instantiations are emitted in `AnsCopyOpBench.cu` for
 * `NumWarps ∈ {1, 2, 4, 8, 16}` (blockDim.x ∈ {32, 64, 128, 256, 512})
 * × `MinBlocksPerSM ∈ {1, 2, 3, 4}`; extend the macro block there if a
 * wider range is needed.
 *
 * Launch requirements:
 *   - blockDim.x == NumWarps * 32
 *   - d_input / d_staging must be 16-byte aligned (cudaMalloc gives
 *     >=256 so this is satisfied for any base pointer)
 *   - `staging_stride` must be >=
 *     `ans_copy_op_bench_staging_stride(input_bytes)`
 */
template <int NumWarps, int MinBlocksPerSM>
__global__
    __launch_bounds__(NumWarps * 32, MinBlocksPerSM) void ans_copy_op_compress_bench_kernel(
        char* d_input,
        char* d_staging,
        std::size_t input_bytes,
        std::size_t staging_stride,
        int iters);

/**
 * Single-GPU microbenchmark companion to
 * `ans_copy_op_compress_bench_kernel` — measures raw
 * `AnsCompress::recv()` (decompression) throughput in isolation.
 * The per-block staging region must already be populated with the
 * ANS-compressed payload produced by the matching compress kernel
 * over the same (input_bytes, staging_stride, NumWarps) parameters.
 */
template <int NumWarps, int MinBlocksPerSM>
__global__
    __launch_bounds__(NumWarps * 32, MinBlocksPerSM) void ans_copy_op_decompress_bench_kernel(
        char* d_output,
        const char* d_staging,
        std::size_t input_bytes,
        std::size_t staging_stride,
        int iters);
#else
template <int NumWarps, int MinBlocksPerSM>
void ans_copy_op_compress_bench_kernel(
    char* d_input,
    char* d_staging,
    std::size_t input_bytes,
    std::size_t staging_stride,
    int iters);
template <int NumWarps, int MinBlocksPerSM>
void ans_copy_op_decompress_bench_kernel(
    char* d_output,
    const char* d_staging,
    std::size_t input_bytes,
    std::size_t staging_stride,
    int iters);
#endif

/**
 * Addresses of the compress + decompress bench kernels matching the
 * caller's runtime-chosen `threads_per_block`. Both `nullptr` when
 * the caller picks a value that no explicit instantiation in
 * `AnsCopyOpBench.cu` covers — the bench should fail loudly
 * host-side rather than launching an uninstantiated symbol.
 */
struct AnsCopyOpBenchKernels {
  void* compress_kernel;
  void* decompress_kernel;
};

/**
 * Map a runtime-chosen `threads_per_block` (= `NumWarps * 32`) and
 * `min_blocks_per_sm` (the `__launch_bounds__` minBlocksPerSM) to the
 * matching explicit instantiation of the bench kernels in
 * `AnsCopyOpBench.cu`. Supported `threads_per_block`: 32, 64, 128,
 * 256, 512 (NumWarps ∈ {1, 2, 4, 8, 16}); supported
 * `min_blocks_per_sm`: 1, 2, 3, 4. Returns `{nullptr, nullptr}` for
 * any unsupported combination.
 */
AnsCopyOpBenchKernels pick_ans_copy_op_bench_kernels(
    int threads_per_block,
    int min_blocks_per_sm);

/**
 * Host-side helper that returns the per-block staging stride (in
 * bytes) the `ans_copy_op_{compress,decompress}_bench_kernel` pair
 * requires to safely hold the worst-case ANS-compressed output of an
 * `input_bytes`-byte uncompressed input. Closed-form upper bound on
 * `AnsCompress<...>::worst_case_chunk_stride(input_bytes)` —
 * conservative enough to avoid requiring nvcompdx's
 * `max_comp_chunk_size()` to be host-callable, while staying within
 * ~1.5x of the uncompressed input size for the bench's range
 * (256KiB..4MiB). NumWarps does not affect this size at the bench's
 * MaxUncomp=256KiB, so the same value works for every supported
 * NumWarps.
 */
std::size_t ans_copy_op_bench_staging_stride(std::size_t input_bytes);

} // namespace comms::prims
