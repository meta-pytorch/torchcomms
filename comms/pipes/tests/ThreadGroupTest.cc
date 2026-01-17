// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/tests/ThreadGroupTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

class ThreadGroupTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    // Don't throw on sync errors - trap tests may leave device in bad state
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      // Clear the error so it doesn't affect subsequent tests
      cudaGetLastError();
    }
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

  const uint32_t warpsPerBlock = blockSize / comms::device::kWarpSize;
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

// =============================================================================
// Partition Tests (Parameterized)
// =============================================================================

struct PartitionTestParams {
  uint32_t numPartitions;
  int numBlocks;
  int blockSize;
  std::string testName;
};

class ThreadGroupPartitionTest
    : public ThreadGroupTestFixture,
      public ::testing::WithParamInterface<PartitionTestParams> {};

TEST_P(ThreadGroupPartitionTest, PartitionEven) {
  const auto& params = GetParam();
  const uint32_t totalWarps =
      params.numBlocks * (params.blockSize / comms::device::kWarpSize);
  const uint32_t numPartitions = params.numPartitions;

  DeviceBuffer partitionIdsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer subgroupIdsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer subgroupTotalGroupsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto partitionIds_d = static_cast<uint32_t*>(partitionIdsBuffer.get());
  auto subgroupIds_d = static_cast<uint32_t*>(subgroupIdsBuffer.get());
  auto subgroupTotalGroups_d =
      static_cast<uint32_t*>(subgroupTotalGroupsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(
      cudaMemset(partitionIds_d, 0xFF, totalWarps * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupIds_d, 0xFF, totalWarps * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupTotalGroups_d, 0xFF, totalWarps * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testPartition(
      partitionIds_d,
      subgroupIds_d,
      subgroupTotalGroups_d,
      numPartitions,
      errorCount_d,
      params.numBlocks,
      params.blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify no kernel errors
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Partition should not produce errors";

  // Copy results to host
  std::vector<uint32_t> partitionIds_h(totalWarps);
  std::vector<uint32_t> subgroupIds_h(totalWarps);
  std::vector<uint32_t> subgroupTotalGroups_h(totalWarps);

  CUDACHECK_TEST(cudaMemcpy(
      partitionIds_h.data(),
      partitionIds_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupIds_h.data(),
      subgroupIds_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupTotalGroups_h.data(),
      subgroupTotalGroups_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  // Verify partition assignments using floor division + remainder distribution
  const uint32_t groupsPerPartition = totalWarps / numPartitions;
  const uint32_t remainder = totalWarps % numPartitions;
  const uint32_t boundary = remainder * (groupsPerPartition + 1);

  for (uint32_t warpId = 0; warpId < totalWarps; warpId++) {
    // Use direct calculation (same as partition() function)
    uint32_t expectedPartition;
    uint32_t partitionStart;
    uint32_t partitionSize;

    if (warpId < boundary) {
      // First 'remainder' partitions (larger size)
      expectedPartition = warpId / (groupsPerPartition + 1);
      partitionStart = expectedPartition * (groupsPerPartition + 1);
      partitionSize = groupsPerPartition + 1;
    } else {
      // Remaining partitions (normal size)
      uint32_t offset = warpId - boundary;
      uint32_t partitionOffset = offset / groupsPerPartition;
      expectedPartition = remainder + partitionOffset;
      partitionStart = boundary + partitionOffset * groupsPerPartition;
      partitionSize = groupsPerPartition;
    }

    uint32_t expectedSubgroupId = warpId - partitionStart;

    EXPECT_EQ(partitionIds_h[warpId], expectedPartition)
        << "Warp " << warpId << " should be in partition " << expectedPartition;

    EXPECT_EQ(subgroupIds_h[warpId], expectedSubgroupId)
        << "Warp " << warpId << " should have subgroup.group_id "
        << expectedSubgroupId;

    EXPECT_EQ(subgroupTotalGroups_h[warpId], partitionSize)
        << "Warp " << warpId << " should have subgroup.total_groups "
        << partitionSize;
  }

  // Verify all partitions that should have warps actually do
  std::vector<uint32_t> partitionCounts(numPartitions, 0);
  for (uint32_t warpId = 0; warpId < totalWarps; warpId++) {
    if (partitionIds_h[warpId] < numPartitions) {
      partitionCounts[partitionIds_h[warpId]]++;
    }
  }

  // Count distinct partitions actually used
  uint32_t distinctPartitions = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    if (partitionCounts[i] > 0) {
      distinctPartitions++;
    }
  }

  // we should get exactly numPartitions
  EXPECT_EQ(distinctPartitions, numPartitions)
      << "Should have " << numPartitions << " distinct partition_ids "
      << "(totalWarps=" << totalWarps << ", numPartitions=" << numPartitions
      << ")";

  // Check partition sizes
  uint32_t totalAssigned = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    totalAssigned += partitionCounts[i];
  }
  EXPECT_EQ(totalAssigned, totalWarps) << "All warps should be assigned";
}

INSTANTIATE_TEST_SUITE_P(
    PartitionConfigs,
    ThreadGroupPartitionTest,
    ::testing::Values(
        // Even 2-way split: 64 / 2 = 32 each
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "TwoPartitions_Even"},
        // Uneven 3-way split: 64 / 3 = 22 + 21 + 21
        PartitionTestParams{
            .numPartitions = 3,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "ThreePartitions_Uneven"},
        // Single partition: all 64 warps in partition 0
        PartitionTestParams{
            .numPartitions = 1,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "SinglePartition"},
        // Partition count equals group count: 64 partitions, 64 warps
        PartitionTestParams{
            .numPartitions = 64,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "OneWarpPerPartition"},
        // 4-way split: 64 / 4 = 16 each
        PartitionTestParams{
            .numPartitions = 4,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "FourPartitions_Even"},
        // 15-way split: 32 warps / 15 partitions (uneven distribution)
        // 4 blocks × 256 threads = 32 warps total
        // Floor division: groups_per_partition = 32 / 15 = 2, remainder = 2
        // First 2 partitions get 3 warps each, remaining 13 get 2 warps each
        // Partition boundaries: [0,3), [3,6), [6,8), [8,10), [10,12), [12,14),
        //                       [14,16), [16,18), [18,20), [20,22), [22,24),
        //                       [24,26), [26,28), [28,30), [30,32)
        // This verifies exactly 15 distinct partition_ids (0-14) are generated
        PartitionTestParams{
            .numPartitions = 15,
            .numBlocks = 4,
            .blockSize = 256,
            .testName = "FifteenPartitions_FourBlocks"}),
    [](const ::testing::TestParamInfo<PartitionTestParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Partition Interleaved Tests (Parameterized)
// =============================================================================

// Parameterized test fixture for partition_interleaved tests
// Uses the same PartitionTestParams as regular partition tests
class ThreadGroupPartitionInterleavedTest
    : public ThreadGroupTestFixture,
      public ::testing::WithParamInterface<PartitionTestParams> {};

// Test: partition_interleaved round-robin assignment
// Verifies:
// - partition_id = group_id % num_partitions (round-robin)
// - subgroup.group_id = group_id / num_partitions (renumbered within partition)
// - subgroup.total_groups = (total_groups + num_partitions - 1 - pid) /
// num_partitions
//
// Example with 8 warps and 2 partitions:
// - Partition 0 gets warps 0, 2, 4, 6 (even)
// - Partition 1 gets warps 1, 3, 5, 7 (odd)
// - Warp 4 has partition_id=0, subgroup.group_id=2, subgroup.total_groups=4
TEST_P(ThreadGroupPartitionInterleavedTest, PartitionInterleaved) {
  const auto& params = GetParam();
  const uint32_t totalWarps =
      params.numBlocks * (params.blockSize / comms::device::kWarpSize);
  const uint32_t numPartitions = params.numPartitions;

  DeviceBuffer partitionIdsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer subgroupIdsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer subgroupTotalGroupsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto partitionIds_d = static_cast<uint32_t*>(partitionIdsBuffer.get());
  auto subgroupIds_d = static_cast<uint32_t*>(subgroupIdsBuffer.get());
  auto subgroupTotalGroups_d =
      static_cast<uint32_t*>(subgroupTotalGroupsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(
      cudaMemset(partitionIds_d, 0xFF, totalWarps * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupIds_d, 0xFF, totalWarps * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupTotalGroups_d, 0xFF, totalWarps * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testPartitionInterleaved(
      partitionIds_d,
      subgroupIds_d,
      subgroupTotalGroups_d,
      numPartitions,
      errorCount_d,
      params.numBlocks,
      params.blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify no kernel errors
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "partition_interleaved should not produce errors";

  // Copy results to host
  std::vector<uint32_t> partitionIds_h(totalWarps);
  std::vector<uint32_t> subgroupIds_h(totalWarps);
  std::vector<uint32_t> subgroupTotalGroups_h(totalWarps);

  CUDACHECK_TEST(cudaMemcpy(
      partitionIds_h.data(),
      partitionIds_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupIds_h.data(),
      subgroupIds_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupTotalGroups_h.data(),
      subgroupTotalGroups_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  // Build expected data based on semantic understanding of interleaved
  // partitioning: warps are assigned to partitions in round-robin fashion
  // (0,1,2,...,N-1,0,1,2,...), and within each partition, warps are numbered
  // sequentially starting from 0.

  // First, determine which warps belong to each partition by simulation
  std::vector<std::vector<uint32_t>> partitionMembers(numPartitions);
  for (uint32_t warpId = 0; warpId < totalWarps; warpId++) {
    // Round-robin: warp goes to next partition in sequence
    uint32_t partition = warpId % numPartitions;
    partitionMembers[partition].push_back(warpId);
  }

  // Build expected vectors from partition membership
  std::vector<uint32_t> expectedPartitionIds(totalWarps);
  std::vector<uint32_t> expectedSubgroupIds(totalWarps);
  std::vector<uint32_t> expectedTotalGroups(totalWarps);

  for (uint32_t partition = 0; partition < numPartitions; partition++) {
    const auto& members = partitionMembers[partition];
    for (uint32_t subgroupId = 0; subgroupId < members.size(); subgroupId++) {
      uint32_t warpId = members[subgroupId];
      expectedPartitionIds[warpId] = partition;
      expectedSubgroupIds[warpId] = subgroupId;
      expectedTotalGroups[warpId] = static_cast<uint32_t>(members.size());
    }
  }

  // Compare entire vectors
  EXPECT_EQ(partitionIds_h, expectedPartitionIds)
      << "Partition IDs should follow interleaved (round-robin) pattern";
  EXPECT_EQ(subgroupIds_h, expectedSubgroupIds)
      << "Subgroup IDs should be sequential within each partition";
  EXPECT_EQ(subgroupTotalGroups_h, expectedTotalGroups)
      << "Total groups should equal partition size";

  // Verify all partitions are used and have correct distribution
  std::vector<uint32_t> partitionCounts(numPartitions, 0);
  for (uint32_t warpId = 0; warpId < totalWarps; warpId++) {
    if (partitionIds_h[warpId] < numPartitions) {
      partitionCounts[partitionIds_h[warpId]]++;
    }
  }

  // All partitions should have warps
  for (uint32_t i = 0; i < numPartitions; i++) {
    uint32_t expectedCount =
        (totalWarps + numPartitions - 1 - i) / numPartitions;
    EXPECT_EQ(partitionCounts[i], expectedCount)
        << "Partition " << i << " should have " << expectedCount << " warps";
  }

  // Total assigned should equal total warps
  uint32_t totalAssigned = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    totalAssigned += partitionCounts[i];
  }
  EXPECT_EQ(totalAssigned, totalWarps) << "All warps should be assigned";
}

INSTANTIATE_TEST_SUITE_P(
    PartitionInterleavedConfigs,
    ThreadGroupPartitionInterleavedTest,
    ::testing::Values(
        // Even 2-way split: 64 warps, partition 0 gets 32 (even), partition 1
        // gets 32 (odd)
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "TwoPartitions_Even"},
        // 3-way split: 64 warps
        // Partition 0: warps 0,3,6,... = 22 warps (ceil(64/3))
        // Partition 1: warps 1,4,7,... = 21 warps
        // Partition 2: warps 2,5,8,... = 21 warps
        PartitionTestParams{
            .numPartitions = 3,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "ThreePartitions_Uneven"},
        // Single partition: all warps in partition 0
        PartitionTestParams{
            .numPartitions = 1,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "SinglePartition"},
        // Partition count equals group count: each warp is its own partition
        PartitionTestParams{
            .numPartitions = 64,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "OneWarpPerPartition"},
        // 4-way split: 64 / 4 = 16 each
        PartitionTestParams{
            .numPartitions = 4,
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "FourPartitions_Even"},
        // Smaller config: 8 warps with 2 partitions
        // Partition 0: warps 0,2,4,6 = 4 warps
        // Partition 1: warps 1,3,5,7 = 4 warps
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 1,
            .blockSize = 256,
            .testName = "SmallConfig_TwoPartitions"}),
    [](const ::testing::TestParamInfo<PartitionTestParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Subgroup Properties Preservation Test
// =============================================================================

// Test: Verify that subgroup preserves thread-level properties
TEST_F(ThreadGroupTestFixture, SubgroupPropertiesPreserved) {
  const int numBlocks = 4;
  const int blockSize = 256;
  const uint32_t totalWarps =
      numBlocks * (blockSize / comms::device::kWarpSize);
  const uint32_t numPartitions = 2;

  DeviceBuffer threadIdsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer groupSizesBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer scopesBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto threadIds_d = static_cast<uint32_t*>(threadIdsBuffer.get());
  auto groupSizes_d = static_cast<uint32_t*>(groupSizesBuffer.get());
  auto scopes_d = static_cast<uint32_t*>(scopesBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testPartitionSubgroupProperties(
      threadIds_d,
      groupSizes_d,
      scopes_d,
      numPartitions,
      errorCount_d,
      numBlocks,
      blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify no errors from GPU-side checks
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "Subgroup should preserve thread_id_in_group, group_size, and scope";

  // Copy and verify on CPU side
  std::vector<uint32_t> groupSizes_h(totalWarps);

  CUDACHECK_TEST(cudaMemcpy(
      groupSizes_h.data(),
      groupSizes_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  // All warps should have comms::device::kWarpSize group_size
  for (uint32_t warpId = 0; warpId < totalWarps; warpId++) {
    EXPECT_EQ(groupSizes_h[warpId], comms::device::kWarpSize)
        << "Warp " << warpId
        << " subgroup should have group_size == comms::device::kWarpSize";
  }
}

// =============================================================================
// Weighted Partition Tests
// =============================================================================

struct WeightedPartitionTestParams {
  std::vector<uint32_t> weights;
  int numBlocks;
  int blockSize;
  std::string testName;
};

class ThreadGroupWeightedPartitionTest
    : public ThreadGroupTestFixture,
      public ::testing::WithParamInterface<WeightedPartitionTestParams> {};

// Helper to compute expected partition boundaries from weights
// With zero-weight handling: zero-weight partitions get 0 groups,
// non-zero-weight partitions get at least 1 group each.
std::vector<uint32_t> computePartitionBoundaries(
    const std::vector<uint32_t>& weights,
    uint32_t totalGroups) {
  std::vector<uint32_t> boundaries;
  uint32_t totalWeight = 0;
  uint32_t nonZeroCount = 0;
  for (auto w : weights) {
    totalWeight += w;
    if (w > 0) {
      nonZeroCount++;
    }
  }

  // Handle edge case: all weights are zero
  if (totalWeight == 0) {
    // All groups go to partition 0
    boundaries.push_back(totalGroups);
    for (size_t i = 1; i < weights.size(); i++) {
      boundaries.push_back(totalGroups);
    }
    return boundaries;
  }

  uint32_t distributableGroups = totalGroups - nonZeroCount;

  uint32_t accumulatedWeight = 0;
  uint32_t nonZeroSeen = 0;
  uint32_t partitionStart = 0;

  for (size_t i = 0; i < weights.size(); i++) {
    accumulatedWeight += weights[i];

    uint32_t boundary;
    if (weights[i] == 0) {
      // Zero-weight partitions get no groups
      boundary = partitionStart;
    } else {
      nonZeroSeen++;
      // Each non-zero partition gets: 1 (guaranteed) + proportional share
      uint32_t proportionalGroups =
          (accumulatedWeight * distributableGroups + totalWeight - 1) /
          totalWeight;
      boundary = nonZeroSeen + proportionalGroups;

      // Clamp to totalGroups
      if (boundary > totalGroups) {
        boundary = totalGroups;
      }
    }
    boundaries.push_back(boundary);
    partitionStart = boundary;
  }
  return boundaries;
}

TEST_P(ThreadGroupWeightedPartitionTest, WeightedPartition) {
  const auto& params = GetParam();
  const uint32_t totalWarps =
      params.numBlocks * (params.blockSize / comms::device::kWarpSize);
  const uint32_t numPartitions = static_cast<uint32_t>(params.weights.size());

  DeviceBuffer partitionIdsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer subgroupIdsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer subgroupTotalGroupsBuffer(totalWarps * sizeof(uint32_t));
  DeviceBuffer weightsBuffer(numPartitions * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto partitionIds_d = static_cast<uint32_t*>(partitionIdsBuffer.get());
  auto subgroupIds_d = static_cast<uint32_t*>(subgroupIdsBuffer.get());
  auto subgroupTotalGroups_d =
      static_cast<uint32_t*>(subgroupTotalGroupsBuffer.get());
  auto weights_d = static_cast<uint32_t*>(weightsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(
      cudaMemset(partitionIds_d, 0xFF, totalWarps * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupIds_d, 0xFF, totalWarps * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupTotalGroups_d, 0xFF, totalWarps * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemcpy(
      weights_d,
      params.weights.data(),
      numPartitions * sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testWeightedPartition(
      partitionIds_d,
      subgroupIds_d,
      subgroupTotalGroups_d,
      weights_d,
      numPartitions,
      errorCount_d,
      params.numBlocks,
      params.blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify no kernel errors
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Weighted partition should not produce errors";

  // Copy results to host
  std::vector<uint32_t> partitionIds_h(totalWarps);
  std::vector<uint32_t> subgroupIds_h(totalWarps);
  std::vector<uint32_t> subgroupTotalGroups_h(totalWarps);

  CUDACHECK_TEST(cudaMemcpy(
      partitionIds_h.data(),
      partitionIds_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupIds_h.data(),
      subgroupIds_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupTotalGroups_h.data(),
      subgroupTotalGroups_d,
      totalWarps * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  // Compute expected partition boundaries
  auto boundaries = computePartitionBoundaries(params.weights, totalWarps);

  // Verify assignments
  for (uint32_t warpId = 0; warpId < totalWarps; warpId++) {
    // Find expected partition
    uint32_t expectedPartition = 0;
    uint32_t partitionStart = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(boundaries.size()); i++) {
      if (warpId < boundaries[i]) {
        expectedPartition = i;
        break;
      }
      partitionStart = boundaries[i];
    }

    uint32_t partitionEnd = boundaries[expectedPartition];
    uint32_t expectedSubgroupId = warpId - partitionStart;
    uint32_t expectedTotalGroups = partitionEnd - partitionStart;

    EXPECT_EQ(partitionIds_h[warpId], expectedPartition)
        << "Warp " << warpId << " should be in partition " << expectedPartition;

    EXPECT_EQ(subgroupIds_h[warpId], expectedSubgroupId)
        << "Warp " << warpId << " should have subgroup.group_id "
        << expectedSubgroupId;

    EXPECT_EQ(subgroupTotalGroups_h[warpId], expectedTotalGroups)
        << "Warp " << warpId << " should have subgroup.total_groups "
        << expectedTotalGroups;
  }

  // Verify partition sizes match expected proportions
  std::vector<uint32_t> partitionCounts(numPartitions, 0);
  for (uint32_t warpId = 0; warpId < totalWarps; warpId++) {
    ASSERT_LT(partitionIds_h[warpId], numPartitions)
        << "Partition ID should be < " << numPartitions;
    partitionCounts[partitionIds_h[warpId]]++;
  }

  // Count distinct partitions actually used
  uint32_t distinctPartitions = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    if (partitionCounts[i] > 0) {
      distinctPartitions++;
    }
  }

  // Count expected non-zero weight partitions
  uint32_t expectedNonZeroPartitions = 0;
  for (auto w : params.weights) {
    if (w > 0) {
      expectedNonZeroPartitions++;
    }
  }

  // We should get exactly the number of non-zero weight partitions
  // (zero-weight partitions don't receive any groups)
  EXPECT_EQ(distinctPartitions, expectedNonZeroPartitions)
      << "Should have " << expectedNonZeroPartitions
      << " distinct partition_ids "
      << "(totalWarps=" << totalWarps << ", numPartitions=" << numPartitions
      << ", nonZeroPartitions=" << expectedNonZeroPartitions << ").";

  uint32_t prevBoundary = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    uint32_t expectedSize = boundaries[i] - prevBoundary;
    EXPECT_EQ(partitionCounts[i], expectedSize)
        << "Partition " << i << " should have " << expectedSize << " warps";
    prevBoundary = boundaries[i];
  }
}

INSTANTIATE_TEST_SUITE_P(
    WeightedPartitions,
    ThreadGroupWeightedPartitionTest,
    ::testing::Values(
        WeightedPartitionTestParams{
            .weights = {1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "EvenSplit_2way"},
        WeightedPartitionTestParams{
            .weights = {3, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "Weighted_3_1"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 1},
            .numBlocks = 6,
            .blockSize = 256,
            .testName = "EvenSplit_3way"},
        WeightedPartitionTestParams{
            .weights = {2, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "Weighted_2_1_1"},
        WeightedPartitionTestParams{
            .weights = {1, 2, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "Weighted_1_2_1"},
        // Uneven rounding: 64 warps with {1,1,1} = 22 + 21 + 21
        WeightedPartitionTestParams{
            .weights = {1, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "UnevenRounding_3way"},
        // Extreme weight ratio: 99:1 split
        WeightedPartitionTestParams{
            .weights = {99, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "ExtremeRatio_99_1"},
        // Very extreme weight ratio: 1000:1 split - ensures minimum 1 group per
        // partition
        WeightedPartitionTestParams{
            .weights = {1000, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "ExtremeRatio_1000_1"},
        // Edge case: num_partitions == total_groups with extreme weights
        // Each partition must get exactly 1 group regardless of weight
        WeightedPartitionTestParams{
            .weights = {1000, 1},
            .numBlocks = 1,
            .blockSize = 64,
            .testName = "ExtremeRatio_MinimumGuarantee"},
        // Single partition (all warps in one partition)
        WeightedPartitionTestParams{
            .weights = {1},
            .numBlocks = 4,
            .blockSize = 256,
            .testName = "SinglePartition"},
        // Many small partitions: 4-way split
        WeightedPartitionTestParams{
            .weights = {1, 1, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "FourWaySplit"},
        // Zero weight tests: zero-weight partitions get 0 groups
        WeightedPartitionTestParams{
            .weights = {3, 0, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "ZeroWeight_Middle"},
        WeightedPartitionTestParams{
            .weights = {0, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "ZeroWeight_First"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 0},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "ZeroWeight_Last"},
        WeightedPartitionTestParams{
            .weights = {1, 0, 0, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .testName = "ZeroWeight_MultipleMiddle"},
        WeightedPartitionTestParams{
            .weights = {0, 0, 1},
            .numBlocks = 4,
            .blockSize = 256,
            .testName = "ZeroWeight_MultipleFirst"},
        // More partitions than groups is OK when zero-weights reduce non-zero
        // count
        WeightedPartitionTestParams{
            .weights = {1, 0, 0, 0},
            .numBlocks = 1,
            .blockSize = 64,
            .testName = "ZeroWeight_MorePartitionsThanGroupsOK"}),
    [](const ::testing::TestParamInfo<WeightedPartitionTestParams>& info) {
      return info.param.testName;
    });

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
