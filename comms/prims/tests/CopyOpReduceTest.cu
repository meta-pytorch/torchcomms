// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/CopyOpReduceTest.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims::test {
namespace {

constexpr int kBlockSize = 384;
using TilePolicy = TileReduce<float, SumOp, 24576, kBlockSize>;
using StagedPolicy = TileReduceStaged<float, SumOp, 24576, kBlockSize>;
using SmemPolicy = CpAsyncSmemReduce<float, SumOp, 8192, kBlockSize, 2>;

template <typename Policy>
__global__ void reduceKernel(
    float* output,
    const float* staging,
    const float* localInput,
    std::size_t byteOffset,
    std::size_t nbytes) {
  auto group = make_block_group();
  Policy::recv(
      reinterpret_cast<char*>(output),
      reinterpret_cast<const char*>(staging),
      nbytes,
      group,
      byteOffset,
      reinterpret_cast<const char*>(localInput));
}

template <typename Policy>
void launch(
    float* output,
    const float* staging,
    const float* localInput,
    std::size_t byteOffset,
    std::size_t nbytes,
    std::size_t dynamicSmem = 0) {
  if (dynamicSmem > 0) {
    const auto status = cudaFuncSetAttribute(
        reduceKernel<Policy>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamicSmem));
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
  }
  reduceKernel<Policy><<<1, kBlockSize, dynamicSmem>>>(
      output, staging, localInput, byteOffset, nbytes);
}

} // namespace

void launchCopyOpReduce(
    CopyOpReducePolicy policy,
    float* output,
    const float* staging,
    const float* localInput,
    std::size_t byteOffset,
    std::size_t nbytes) {
  switch (policy) {
    case CopyOpReducePolicy::TileReduce:
      launch<TilePolicy>(output, staging, localInput, byteOffset, nbytes);
      return;
    case CopyOpReducePolicy::TileReduceStaged:
      launch<StagedPolicy>(output, staging, localInput, byteOffset, nbytes);
      return;
    case CopyOpReducePolicy::CpAsyncSmemReduce:
      launch<SmemPolicy>(
          output,
          staging,
          localInput,
          byteOffset,
          nbytes,
          SmemPolicy::smem_bytes());
      return;
  }
  throw std::invalid_argument("unknown CopyOp reduce policy");
}

} // namespace comms::prims::test
