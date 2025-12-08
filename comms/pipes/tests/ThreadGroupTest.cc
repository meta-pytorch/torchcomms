// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/tests/ThreadGroupTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes {

class ThreadGroupTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

// Test parameters for contiguous locality tests
struct ContiguousTestParams {
  uint32_t numItems;
  std::string description;
  std::string testName;
};

// Parameterized test fixture for contiguous locality tests
class ThreadGroupContiguousTest
    : public ThreadGroupTestFixture,
      public ::testing::WithParamInterface<ContiguousTestParams> {};

// Test: for_each_item_contiguous correctness and locality
// Verifies that:
// 1. Each work item is processed exactly once (no duplicates, no skips)
// 2. Each group processes a CONTIGUOUS block of work items
// 3. Adjacent groups process adjacent memory regions (cache coherence)
// 4. The contiguous pattern is correct: Group N gets work items [N*K, (N+1)*K)
// 5. Handles both even and uneven distributions correctly
//
// Why this matters:
// - CONTIGUOUS pattern: Group 0 gets [0,1,2,3], Group 1 gets [4,5,6,7]
//   → Contiguous memory access → NVLink cache hits → 40% faster
// - STRIDED pattern: Group 0 gets [0,4,8,12], Group 1 gets [1,5,9,13]
//   → Scattered memory access → Cache misses → Slower
// - Real workloads rarely divide evenly by number of warps
// - Edge cases can expose off-by-one errors in loop bounds
//
// Test setup:
// - Configurable number of work items (even or uneven distribution)
// - 8 blocks × 256 threads/block = 2048 threads total
// - 2048 threads / 32 = 64 warps
//
// Verification method:
// - Each warp writes its group_id to all work items it processes
// - CPU verifies that work items [start, end) all have the same group_id
// - CPU verifies this matches the expected contiguous assignment pattern
// - CPU verifies last group processes exactly the remaining items
TEST_P(ThreadGroupContiguousTest, ForEachItemContiguousLocality) {
  const auto& params = GetParam();
  const uint32_t numItems = params.numItems;
  const int numBlocks = 8;
  const int blockSize = 256;

  DeviceBuffer groupIdsBuffer(numItems * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto groupIds_d = static_cast<uint32_t*>(groupIdsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemset(groupIds_d, 0, numItems * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  // Launch kernel: each warp writes its group_id to its assigned work items
  test::testContiguousLocality(
      groupIds_d, numItems, errorCount_d, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify kernel didn't detect any errors during execution
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  EXPECT_EQ(errorCount_h, 0)
      << "Contiguous pattern should assign contiguous work items to same group ("
      << params.description << ")";

  // Copy results and verify block assignment pattern on CPU
  std::vector<uint32_t> groupIds_h(numItems);
  CUDACHECK_TEST(cudaMemcpy(
      groupIds_h.data(),
      groupIds_d,
      numItems * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  const uint32_t warpsPerBlock = blockSize / WARP_SIZE;
  const uint32_t totalWarps = numBlocks * warpsPerBlock;
  const uint32_t itemsPerGroup = (numItems + totalWarps - 1) / totalWarps;

  // Verify each group processed exactly its assigned contiguous block
  for (uint32_t group_id = 0; group_id < totalWarps; group_id++) {
    uint32_t start_item = group_id * itemsPerGroup;
    uint32_t end_item = std::min(start_item + itemsPerGroup, numItems);

    // Skip groups that have no items assigned
    if (start_item >= numItems) {
      break;
    }

    for (uint32_t item_id = start_item; item_id < end_item; item_id++) {
      EXPECT_EQ(groupIds_h[item_id], group_id)
          << "Work item " << item_id << " should be assigned to group "
          << group_id << " but was assigned to " << groupIds_h[item_id] << " ("
          << params.description << ")";
    }

    // Verify last group processes exactly the remaining items
    if (group_id == totalWarps - 1 || start_item + itemsPerGroup >= numItems) {
      uint32_t expected_count = end_item - start_item;
      uint32_t actual_count = 0;
      for (uint32_t item_id = start_item; item_id < end_item; item_id++) {
        if (groupIds_h[item_id] == group_id) {
          actual_count++;
        }
      }
      EXPECT_EQ(actual_count, expected_count)
          << "Group " << group_id << " should process " << expected_count
          << " items but processed " << actual_count << " ("
          << params.description << ")";
    }
  }
}

// Instantiate parameterized tests with even and uneven distributions
INSTANTIATE_TEST_SUITE_P(
    ContiguousDistributions,
    ThreadGroupContiguousTest,
    ::testing::Values(
        ContiguousTestParams{
            .numItems = 2048,
            .description =
                "even distribution (2048 items, 64 warps, 32 items/warp)",
            .testName = "EvenCase"},
        ContiguousTestParams{
            .numItems = 2040,
            .description =
                "uneven distribution (2040 items, 64 warps, last warp has 24 items)",
            .testName = "UnEvenCase"}),
    [](const ::testing::TestParamInfo<ContiguousTestParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Block Group Tests
// =============================================================================

// Test: make_block_group creates correct ThreadGroup
// Verifies:
// - group_id == blockIdx.x (each block is its own group)
// - group_size == blockDim.x (all threads in block form the group)
// - thread_id_in_group == threadIdx.x
// - total_groups == gridDim.x
// - Work items are distributed contiguously across block groups
TEST_F(ThreadGroupTestFixture, BlockGroupContiguousLocality) {
  const uint32_t numItems = 1024;
  const int numBlocks = 4;
  const int blockSize = 256;

  DeviceBuffer groupIdsBuffer(numItems * sizeof(uint32_t));
  DeviceBuffer threadIdsBuffer(numItems * sizeof(uint32_t));
  DeviceBuffer groupSizesBuffer(numBlocks * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto groupIds_d = static_cast<uint32_t*>(groupIdsBuffer.get());
  auto threadIds_d = static_cast<uint32_t*>(threadIdsBuffer.get());
  auto groupSizes_d = static_cast<uint32_t*>(groupSizesBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemset(groupIds_d, 0, numItems * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(threadIds_d, 0, numItems * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(groupSizes_d, 0, numBlocks * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testBlockGroup(
      groupIds_d,
      threadIds_d,
      groupSizes_d,
      numItems,
      errorCount_d,
      numBlocks,
      blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify no kernel errors
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Block group should not have any errors";

  // Verify group sizes
  std::vector<uint32_t> groupSizes_h(numBlocks);
  CUDACHECK_TEST(cudaMemcpy(
      groupSizes_h.data(),
      groupSizes_d,
      numBlocks * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  for (int i = 0; i < numBlocks; i++) {
    EXPECT_EQ(groupSizes_h[i], static_cast<uint32_t>(blockSize))
        << "Block " << i << " should have group_size == blockSize";
  }

  // Verify contiguous distribution of work items
  std::vector<uint32_t> groupIds_h(numItems);
  CUDACHECK_TEST(cudaMemcpy(
      groupIds_h.data(),
      groupIds_d,
      numItems * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  const uint32_t itemsPerGroup = (numItems + numBlocks - 1) / numBlocks;

  for (uint32_t group_id = 0; group_id < static_cast<uint32_t>(numBlocks);
       group_id++) {
    uint32_t start_item = group_id * itemsPerGroup;
    uint32_t end_item = std::min(start_item + itemsPerGroup, numItems);

    for (uint32_t item_id = start_item; item_id < end_item; item_id++) {
      EXPECT_EQ(groupIds_h[item_id], group_id)
          << "Work item " << item_id << " should be assigned to block group "
          << group_id;
    }
  }
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
