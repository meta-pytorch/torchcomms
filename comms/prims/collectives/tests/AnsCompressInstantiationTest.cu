// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Compile-only regression guard for the AnsCompress CopyOp policy.
//
// This TU includes core/CopyOp.cuh with -DPIPES_ENABLE_ANS_COMPRESSION (via the
// :copy_op_compress dep) and instantiates the policy's device-callable sizing
// methods in *whole-program* mode (-rdc=false) — the same mode the OSS wheel /
// CI builds use when compiling consumers of this header. Building it forces
// nvcc to compile the AnsCompress class body and these device methods, which
// catches header-level regressions that a headers-only target cannot:
//   - an `inline __device__` global with external linkage (illegal at
//     -rdc=false), and
//   - a device-side read of a non-constexpr host static.
//
// Full send()/recv() instantiation (nvcompdx execute + device-link of the LTO
// fatbin) is exercised end-to-end by the standalone ANS microbench, which is
// compiled with relocatable device code.

#include "comms/prims/core/CopyOp.cuh"

namespace comms::prims {
namespace {

template <int NumWarps>
__device__ __forceinline__ void instantiate_ans_sizing(
    std::size_t* out,
    std::size_t n) {
  using Policy = AnsCompress<NumWarps>;
  out[0] = Policy::worst_case_chunk_stride(n);
  out[1] = Policy::max_safe_chunk_size_for_slot(n);
}

} // namespace

__global__ void ans_compress_instantiation_probe(
    std::size_t* out,
    std::size_t n) {
  instantiate_ans_sizing<1>(out, n);
  instantiate_ans_sizing<8>(out + 2, n);
}

} // namespace comms::prims
