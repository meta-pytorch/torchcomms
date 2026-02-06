// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Tests for MultipeerIbgdaDeviceTransport
// Note: MultipeerIbgdaDeviceTransport.cuh includes CUDA headers that cannot
// be compiled by a regular C++ compiler. Device-side tests are implemented in
// MultipeerIbgdaDeviceTransportTest.cu and launched via kernel wrapper
// functions.

#include <gtest/gtest.h>

#include <vector>

#include "comms/pipes/tests/MultipeerIbgdaDeviceTransportTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

// =============================================================================
// Parameterized Device-side Rank Mapping Test
// =============================================================================

struct RankMappingTestCase {
  int myRank;
  int nRanks;
  std::vector<int> expectedResults;
};

class RankMappingTest : public ::testing::TestWithParam<RankMappingTestCase> {};

TEST_P(RankMappingTest, DeviceRankMapping) {
  const auto& testCase = GetParam();
  const int myRank = testCase.myRank;
  const int nRanks = testCase.nRanks;
  const int numPeers = nRanks - 1;
  const auto& expectedResults = testCase.expectedResults;

  // Allocate device memory
  DeviceBuffer resultsBuf(numPeers * sizeof(int));
  DeviceBuffer expectedBuf(numPeers * sizeof(int));
  DeviceBuffer successBuf(sizeof(bool));

  auto* d_results = static_cast<int*>(resultsBuf.get());
  auto* d_expected = static_cast<int*>(expectedBuf.get());
  auto* d_success = static_cast<bool*>(successBuf.get());

  // Copy expected results to device
  CUDACHECK_TEST(cudaMemcpy(
      d_expected,
      expectedResults.data(),
      numPeers * sizeof(int),
      cudaMemcpyHostToDevice));

  // Initialize success to false
  bool initSuccess = false;
  CUDACHECK_TEST(cudaMemcpy(
      d_success, &initSuccess, sizeof(bool), cudaMemcpyHostToDevice));

  // Run kernel
  runTestRankMappingKernel(
      myRank, nRanks, d_results, d_expected, numPeers, d_success);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy results back
  bool success = false;
  CUDACHECK_TEST(
      cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));

  std::vector<int> actualResults(numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      actualResults.data(),
      d_results,
      numPeers * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_TRUE(success) << "Device-side rank mapping test failed for myRank="
                       << myRank;
  EXPECT_EQ(actualResults, expectedResults);
}

INSTANTIATE_TEST_SUITE_P(
    MultipeerIbgdaDeviceTransport,
    RankMappingTest,
    ::testing::Values(
        // Rank 0: all peers have higher ranks [1, 2, 3]
        RankMappingTestCase{0, 4, {1, 2, 3}},
        // Rank 2: peers on both sides [0, 1, 3]
        RankMappingTestCase{2, 4, {0, 1, 3}},
        // Rank 3: all peers have lower ranks [0, 1, 2]
        RankMappingTestCase{3, 4, {0, 1, 2}}));

} // namespace comms::pipes::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
