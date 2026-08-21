// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/prims/tests/MultimemNvlSignalTrapTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims::tests {

TEST(MultimemNvlSignalRankBoundaryTest, Supports64_65And72Ranks) {
  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "No CUDA devices available";
  }
  CUDACHECK_TEST(cudaSetDevice(0));

  uint64_t* output = nullptr;
  CUDACHECK_TEST(cudaMalloc(&output, sizeof(uint64_t)));
  for (const int nvlRanks : {64, 65, 72}) {
    CUDACHECK_TEST(cudaMemset(output, 0, sizeof(uint64_t)));
    test::launchNvlSignalRankBoundary(nvlRanks, output);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    uint64_t observed = 0;
    CUDACHECK_TEST(cudaMemcpy(
        &observed, output, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(observed, static_cast<uint64_t>(nvlRanks));
  }
  CUDACHECK_TEST(cudaFree(output));
}

} // namespace comms::prims::tests
