// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <array>
#include <cstddef>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/prims/tests/MultimemNvlSignalTrapTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims::tests {

TEST(
    MultimemNvlSignalRankBoundaryTest,
    Supports64_65And72RanksAcrossWaitPolicies) {
  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "No CUDA devices available";
  }
  CUDACHECK_TEST(cudaSetDevice(0));

  uint64_t* output = nullptr;
  CUDACHECK_TEST(cudaMalloc(&output, sizeof(uint64_t)));
  constexpr std::array waitPolicies{
      test::NvlSignalRankBoundaryWaitPolicy::WaitAll,
      test::NvlSignalRankBoundaryWaitPolicy::SerialMin,
      test::NvlSignalRankBoundaryWaitPolicy::TreeMin,
      test::NvlSignalRankBoundaryWaitPolicy::ButterflyMin,
  };
  for (const int nvlRanks : {64, 65, 72}) {
    for (std::size_t policyIndex = 0; policyIndex < waitPolicies.size();
         ++policyIndex) {
      const uint64_t roundValue = static_cast<uint64_t>(nvlRanks) *
              static_cast<uint64_t>(waitPolicies.size()) +
          policyIndex + 1;
      CUDACHECK_TEST(cudaMemset(output, 0, sizeof(uint64_t)));
      test::launchNvlSignalRankBoundary(
          nvlRanks, waitPolicies[policyIndex], roundValue, output);
      CUDACHECK_TEST(cudaDeviceSynchronize());
      uint64_t observed = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &observed, output, sizeof(uint64_t), cudaMemcpyDeviceToHost));
      EXPECT_EQ(observed, roundValue);
    }
  }
  CUDACHECK_TEST(cudaFree(output));
}

} // namespace comms::prims::tests
