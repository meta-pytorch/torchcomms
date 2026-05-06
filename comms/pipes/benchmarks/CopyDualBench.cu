// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/CopyDualBench.cuh"

namespace comms::pipes::benchmark {

__global__ void copy_dual_kernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns) {
  auto group = make_block_group();

  for (int run = 0; run < nRuns; ++run) {
    memcpy_dual(dst1, dst2, src, nBytes, group);
  }
}

__global__ void copy_two_sequential_kernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns) {
  auto group = make_block_group();

  for (int run = 0; run < nRuns; ++run) {
    memcpy_vectorized(dst1, src, nBytes, group);
    group.sync();
    memcpy_vectorized(dst2, dst1, nBytes, group);
  }
}

void launch_copy_dual(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  copy_dual_kernel<<<numBlocks, numThreads, 0, stream>>>(
      dst1, dst2, src, nBytes, nRuns);
}

void launch_copy_two_sequential(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  copy_two_sequential_kernel<<<numBlocks, numThreads, 0, stream>>>(
      dst1, dst2, src, nBytes, nRuns);
}

} // namespace comms::pipes::benchmark
