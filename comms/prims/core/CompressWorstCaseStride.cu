// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Host-side evaluation of a compressor's worst-case chunk stride. ANS is the
// only codec wired up today; a second one adds its own entry point here rather
// than a new translation unit.
//
// This is a thin TU rather than an inline helper in `core/CompressorTags.cuh`
// because the computation needs `AnsCompress` from `core/CopyOp.cuh`, and that
// header is only parseable when both PIPES_ENABLE_ANS_COMPRESSION and
// __CUDACC__ are defined (its `AnsCompress` body uses device intrinsics). The
// callers that need this value -- benchmarks and tests -- are host `.cc` TUs
// compiled by clang, so they cannot include it. This file is compiled by nvcc
// and re-exports the one number they need behind a plain host signature.
//
// No kernel and no device link: `nvcompdx`'s `max_comp_chunk_size()` is
// `constexpr __host__ __device__` and resolves to
// `MaxCompChunkSize<algorithm>::execute(max_uncomp_chunk_size)` -- a function
// of the algorithm and the uncompressed chunk size only. The `SM<>` arch in the
// descriptor type does not participate, so the host answer equals the device
// answer and there is nothing to ask the GPU.

#include "comms/prims/core/CompressorTags.cuh"

namespace comms::prims {

namespace {

// The compressor the collectives drive, named directly off `AnsCompress` rather
// than through `collectives/CompressorDispatch.cuh`, so this core-level TU
// carries no dependency on the collectives layer.
// `compressor_copyop<AnsCompressor, N>` resolves to exactly this type.
//
// The StatsTag is left at its `void` default: `worst_case_chunk_stride` does
// not touch the counters, and nothing here calls `AnsCompress::send`, so no
// `ans_counters::*` symbols are instantiated.
template <int NumWarps>
std::size_t stride_for(std::size_t chunk_bytes) {
  return AnsCompress<NumWarps, PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES, false>::
      worst_case_chunk_stride(chunk_bytes);
}

} // namespace

std::size_t ans_worst_case_chunk_stride(
    int num_warps,
    std::size_t chunk_bytes) {
  switch (num_warps) {
    case 1:
      return stride_for<1>(chunk_bytes);
    case 2:
      return stride_for<2>(chunk_bytes);
    case 4:
      return stride_for<4>(chunk_bytes);
    case 8:
      return stride_for<8>(chunk_bytes);
    case 16:
      return stride_for<16>(chunk_bytes);
    case 32:
      return stride_for<32>(chunk_bytes);
    default:
      return 0;
  }
}

} // namespace comms::prims
