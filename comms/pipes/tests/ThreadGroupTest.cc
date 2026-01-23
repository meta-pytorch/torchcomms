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

// Warpgroup size constant (4 warps = 128 threads)
constexpr uint32_t kWarpgroupSize = 4 * comms::device::kWarpSize;

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

// =============================================================================
// Contiguous Locality Tests
// =============================================================================

struct ContiguousTestParams {
  uint32_t numItems;
  SyncScope scope;
  std::string description;
  std::string testName;
};

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

  test::testContiguousLocality(
      groupIds_d, numItems, errorCount_d, numBlocks, blockSize, params.scope);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  EXPECT_EQ(errorCount_h, 0)
      << "Contiguous pattern should assign contiguous work items to same group ("
      << params.description << ")";

  std::vector<uint32_t> groupIds_h(numItems);
  CUDACHECK_TEST(cudaMemcpy(
      groupIds_h.data(),
      groupIds_d,
      numItems * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  uint32_t totalGroups;
  if (params.scope == SyncScope::WARP) {
    const uint32_t warpsPerBlock = blockSize / comms::device::kWarpSize;
    totalGroups = numBlocks * warpsPerBlock;
  } else if (params.scope == SyncScope::WARPGROUP) {
    const uint32_t warpgroupsPerBlock = blockSize / kWarpgroupSize;
    totalGroups = numBlocks * warpgroupsPerBlock;
  } else {
    totalGroups = numBlocks;
  }
  const uint32_t itemsPerGroup = (numItems + totalGroups - 1) / totalGroups;

  for (uint32_t group_id = 0; group_id < totalGroups; group_id++) {
    uint32_t start_item = group_id * itemsPerGroup;
    uint32_t end_item = std::min(start_item + itemsPerGroup, numItems);

    if (start_item >= numItems) {
      break;
    }

    for (uint32_t item_id = start_item; item_id < end_item; item_id++) {
      EXPECT_EQ(groupIds_h[item_id], group_id)
          << "Work item " << item_id << " should be assigned to group "
          << group_id << " but was assigned to " << groupIds_h[item_id] << " ("
          << params.description << ")";
    }

    if (group_id == totalGroups - 1 || start_item + itemsPerGroup >= numItems) {
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

INSTANTIATE_TEST_SUITE_P(
    ContiguousDistributions,
    ThreadGroupContiguousTest,
    ::testing::Values(
        ContiguousTestParams{
            .numItems = 2048,
            .scope = SyncScope::WARP,
            .description = "WARP: even distribution (2048 items, 64 warps)",
            .testName = "Warp_EvenCase"},
        ContiguousTestParams{
            .numItems = 2040,
            .scope = SyncScope::WARP,
            .description = "WARP: uneven distribution (2040 items, 64 warps)",
            .testName = "Warp_UnevenCase"},
        ContiguousTestParams{
            .numItems = 1024,
            .scope = SyncScope::TILE,
            .description = "TILE: even distribution (1024 items, 8 blocks)",
            .testName = "Tile_EvenCase"},
        ContiguousTestParams{
            .numItems = 1000,
            .scope = SyncScope::TILE,
            .description = "TILE: uneven distribution (1000 items, 8 blocks)",
            .testName = "Tile_UnevenCase"},
        ContiguousTestParams{
            .numItems = 1024,
            .scope = SyncScope::WARPGROUP,
            .description =
                "WARPGROUP: even distribution (1024 items, 16 warpgroups)",
            .testName = "Warpgroup_EvenCase"},
        ContiguousTestParams{
            .numItems = 1000,
            .scope = SyncScope::WARPGROUP,
            .description =
                "WARPGROUP: uneven distribution (1000 items, 16 warpgroups)",
            .testName = "Warpgroup_UnevenCase"}),
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

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Block group should not have any errors";

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
// Partition Tests (Parameterized for WARP and TILE)
// =============================================================================

struct PartitionTestParams {
  uint32_t numPartitions;
  int numBlocks;
  int blockSize;
  SyncScope scope;
  std::string testName;
};

class ThreadGroupPartitionTest
    : public ThreadGroupTestFixture,
      public ::testing::WithParamInterface<PartitionTestParams> {};

TEST_P(ThreadGroupPartitionTest, PartitionEven) {
  const auto& params = GetParam();
  const uint32_t numPartitions = params.numPartitions;

  uint32_t totalGroups;
  if (params.scope == SyncScope::WARP) {
    totalGroups =
        params.numBlocks * (params.blockSize / comms::device::kWarpSize);
  } else if (params.scope == SyncScope::WARPGROUP) {
    totalGroups = params.numBlocks * (params.blockSize / kWarpgroupSize);
  } else {
    totalGroups = params.numBlocks;
  }

  DeviceBuffer partitionIdsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer subgroupIdsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer subgroupTotalGroupsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto partitionIds_d = static_cast<uint32_t*>(partitionIdsBuffer.get());
  auto subgroupIds_d = static_cast<uint32_t*>(subgroupIdsBuffer.get());
  auto subgroupTotalGroups_d =
      static_cast<uint32_t*>(subgroupTotalGroupsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(
      cudaMemset(partitionIds_d, 0xFF, totalGroups * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupIds_d, 0xFF, totalGroups * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupTotalGroups_d, 0xFF, totalGroups * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testPartition(
      partitionIds_d,
      subgroupIds_d,
      subgroupTotalGroups_d,
      numPartitions,
      errorCount_d,
      params.numBlocks,
      params.blockSize,
      params.scope);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Partition should not produce errors";

  std::vector<uint32_t> partitionIds_h(totalGroups);
  std::vector<uint32_t> subgroupIds_h(totalGroups);
  std::vector<uint32_t> subgroupTotalGroups_h(totalGroups);

  CUDACHECK_TEST(cudaMemcpy(
      partitionIds_h.data(),
      partitionIds_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupIds_h.data(),
      subgroupIds_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupTotalGroups_h.data(),
      subgroupTotalGroups_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  const uint32_t groupsPerPartition = totalGroups / numPartitions;
  const uint32_t remainder = totalGroups % numPartitions;
  const uint32_t boundary = remainder * (groupsPerPartition + 1);

  for (uint32_t groupId = 0; groupId < totalGroups; groupId++) {
    uint32_t expectedPartition;
    uint32_t partitionStart;
    uint32_t partitionSize;

    if (groupId < boundary) {
      expectedPartition = groupId / (groupsPerPartition + 1);
      partitionStart = expectedPartition * (groupsPerPartition + 1);
      partitionSize = groupsPerPartition + 1;
    } else {
      uint32_t offset = groupId - boundary;
      uint32_t partitionOffset = offset / groupsPerPartition;
      expectedPartition = remainder + partitionOffset;
      partitionStart = boundary + partitionOffset * groupsPerPartition;
      partitionSize = groupsPerPartition;
    }

    uint32_t expectedSubgroupId = groupId - partitionStart;

    EXPECT_EQ(partitionIds_h[groupId], expectedPartition)
        << "Group " << groupId << " should be in partition "
        << expectedPartition;

    EXPECT_EQ(subgroupIds_h[groupId], expectedSubgroupId)
        << "Group " << groupId << " should have subgroup.group_id "
        << expectedSubgroupId;

    EXPECT_EQ(subgroupTotalGroups_h[groupId], partitionSize)
        << "Group " << groupId << " should have subgroup.total_groups "
        << partitionSize;
  }

  std::vector<uint32_t> partitionCounts(numPartitions, 0);
  for (uint32_t groupId = 0; groupId < totalGroups; groupId++) {
    if (partitionIds_h[groupId] < numPartitions) {
      partitionCounts[partitionIds_h[groupId]]++;
    }
  }

  uint32_t distinctPartitions = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    if (partitionCounts[i] > 0) {
      distinctPartitions++;
    }
  }

  EXPECT_EQ(distinctPartitions, numPartitions)
      << "Should have " << numPartitions << " distinct partition_ids "
      << "(totalGroups=" << totalGroups << ", numPartitions=" << numPartitions
      << ")";

  uint32_t totalAssigned = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    totalAssigned += partitionCounts[i];
  }
  EXPECT_EQ(totalAssigned, totalGroups) << "All groups should be assigned";
}

INSTANTIATE_TEST_SUITE_P(
    PartitionConfigs,
    ThreadGroupPartitionTest,
    ::testing::Values(
        // WARP group tests
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_TwoPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 3,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_ThreePartitions_Uneven"},
        PartitionTestParams{
            .numPartitions = 1,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_SinglePartition"},
        PartitionTestParams{
            .numPartitions = 64,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_OneWarpPerPartition"},
        PartitionTestParams{
            .numPartitions = 4,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_FourPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 15,
            .numBlocks = 4,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_FifteenPartitions_FourBlocks"},
        // TILE (Block) group tests
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_TwoPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 3,
            .numBlocks = 9,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_ThreePartitions_Even"},
        PartitionTestParams{
            .numPartitions = 1,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_SinglePartition"},
        PartitionTestParams{
            .numPartitions = 8,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_OneBlockPerPartition"},
        PartitionTestParams{
            .numPartitions = 4,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_FourPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 5,
            .numBlocks = 12,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_FivePartitions_Uneven"},
        // WARPGROUP (4 warps = 128 threads) tests
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_TwoPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 3,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_ThreePartitions_Uneven"},
        PartitionTestParams{
            .numPartitions = 1,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_SinglePartition"},
        PartitionTestParams{
            .numPartitions = 16,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_OneWarpgroupPerPartition"},
        PartitionTestParams{
            .numPartitions = 4,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_FourPartitions_Even"}),
    [](const ::testing::TestParamInfo<PartitionTestParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Partition Interleaved Tests (Parameterized for WARP and TILE)
// =============================================================================

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
  const uint32_t numPartitions = params.numPartitions;

  uint32_t totalGroups;
  if (params.scope == SyncScope::WARP) {
    totalGroups =
        params.numBlocks * (params.blockSize / comms::device::kWarpSize);
  } else if (params.scope == SyncScope::WARPGROUP) {
    totalGroups = params.numBlocks * (params.blockSize / kWarpgroupSize);
  } else {
    totalGroups = params.numBlocks;
  }

  DeviceBuffer partitionIdsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer subgroupIdsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer subgroupTotalGroupsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto partitionIds_d = static_cast<uint32_t*>(partitionIdsBuffer.get());
  auto subgroupIds_d = static_cast<uint32_t*>(subgroupIdsBuffer.get());
  auto subgroupTotalGroups_d =
      static_cast<uint32_t*>(subgroupTotalGroupsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(
      cudaMemset(partitionIds_d, 0xFF, totalGroups * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupIds_d, 0xFF, totalGroups * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupTotalGroups_d, 0xFF, totalGroups * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testPartitionInterleaved(
      partitionIds_d,
      subgroupIds_d,
      subgroupTotalGroups_d,
      numPartitions,
      errorCount_d,
      params.numBlocks,
      params.blockSize,
      params.scope);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "partition_interleaved should not produce errors";

  std::vector<uint32_t> partitionIds_h(totalGroups);
  std::vector<uint32_t> subgroupIds_h(totalGroups);
  std::vector<uint32_t> subgroupTotalGroups_h(totalGroups);

  CUDACHECK_TEST(cudaMemcpy(
      partitionIds_h.data(),
      partitionIds_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupIds_h.data(),
      subgroupIds_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupTotalGroups_h.data(),
      subgroupTotalGroups_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  std::vector<std::vector<uint32_t>> partitionMembers(numPartitions);
  for (uint32_t groupId = 0; groupId < totalGroups; groupId++) {
    uint32_t partition = groupId % numPartitions;
    partitionMembers[partition].push_back(groupId);
  }

  std::vector<uint32_t> expectedPartitionIds(totalGroups);
  std::vector<uint32_t> expectedSubgroupIds(totalGroups);
  std::vector<uint32_t> expectedTotalGroups(totalGroups);

  for (uint32_t partition = 0; partition < numPartitions; partition++) {
    const auto& members = partitionMembers[partition];
    for (uint32_t subgroupId = 0; subgroupId < members.size(); subgroupId++) {
      uint32_t groupId = members[subgroupId];
      expectedPartitionIds[groupId] = partition;
      expectedSubgroupIds[groupId] = subgroupId;
      expectedTotalGroups[groupId] = static_cast<uint32_t>(members.size());
    }
  }

  EXPECT_EQ(partitionIds_h, expectedPartitionIds)
      << "Partition IDs should follow interleaved (round-robin) pattern";
  EXPECT_EQ(subgroupIds_h, expectedSubgroupIds)
      << "Subgroup IDs should be sequential within each partition";
  EXPECT_EQ(subgroupTotalGroups_h, expectedTotalGroups)
      << "Total groups should equal partition size";

  std::vector<uint32_t> partitionCounts(numPartitions, 0);
  for (uint32_t groupId = 0; groupId < totalGroups; groupId++) {
    if (partitionIds_h[groupId] < numPartitions) {
      partitionCounts[partitionIds_h[groupId]]++;
    }
  }

  for (uint32_t i = 0; i < numPartitions; i++) {
    uint32_t expectedCount =
        (totalGroups + numPartitions - 1 - i) / numPartitions;
    EXPECT_EQ(partitionCounts[i], expectedCount)
        << "Partition " << i << " should have " << expectedCount << " groups";
  }

  uint32_t totalAssigned = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    totalAssigned += partitionCounts[i];
  }
  EXPECT_EQ(totalAssigned, totalGroups) << "All groups should be assigned";
}

INSTANTIATE_TEST_SUITE_P(
    PartitionInterleavedConfigs,
    ThreadGroupPartitionInterleavedTest,
    ::testing::Values(
        // WARP group tests
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_TwoPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 3,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_ThreePartitions_Uneven"},
        PartitionTestParams{
            .numPartitions = 1,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_SinglePartition"},
        PartitionTestParams{
            .numPartitions = 64,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_OneWarpPerPartition"},
        PartitionTestParams{
            .numPartitions = 4,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_FourPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 1,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_SmallConfig_TwoPartitions"},
        // TILE (Block) group tests
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_TwoPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 3,
            .numBlocks = 9,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_ThreePartitions_Even"},
        PartitionTestParams{
            .numPartitions = 1,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_SinglePartition"},
        PartitionTestParams{
            .numPartitions = 8,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_OneBlockPerPartition"},
        PartitionTestParams{
            .numPartitions = 4,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_FourPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 4,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_SmallConfig_TwoPartitions"},
        // WARPGROUP (4 warps = 128 threads) tests
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_TwoPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 3,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_ThreePartitions_Uneven"},
        PartitionTestParams{
            .numPartitions = 1,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_SinglePartition"},
        PartitionTestParams{
            .numPartitions = 16,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_OneWarpgroupPerPartition"},
        PartitionTestParams{
            .numPartitions = 4,
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_FourPartitions_Even"},
        PartitionTestParams{
            .numPartitions = 2,
            .numBlocks = 1,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_SmallConfig_TwoPartitions"}),
    [](const ::testing::TestParamInfo<PartitionTestParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Subgroup Properties Preservation Test (Parameterized for WARP and TILE)
// =============================================================================

struct SubgroupPropertiesTestParams {
  int numBlocks;
  int blockSize;
  uint32_t numPartitions;
  SyncScope scope;
  std::string testName;
};

class ThreadGroupSubgroupPropertiesTest
    : public ThreadGroupTestFixture,
      public ::testing::WithParamInterface<SubgroupPropertiesTestParams> {};

TEST_P(ThreadGroupSubgroupPropertiesTest, SubgroupPropertiesPreserved) {
  const auto& params = GetParam();

  uint32_t totalGroups;
  uint32_t expectedGroupSize;
  if (params.scope == SyncScope::WARP) {
    totalGroups =
        params.numBlocks * (params.blockSize / comms::device::kWarpSize);
    expectedGroupSize = comms::device::kWarpSize;
  } else if (params.scope == SyncScope::WARPGROUP) {
    totalGroups = params.numBlocks * (params.blockSize / kWarpgroupSize);
    expectedGroupSize = kWarpgroupSize;
  } else {
    totalGroups = params.numBlocks;
    expectedGroupSize = params.blockSize;
  }

  DeviceBuffer threadIdsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer groupSizesBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer scopesBuffer(totalGroups * sizeof(uint32_t));
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
      params.numPartitions,
      errorCount_d,
      params.numBlocks,
      params.blockSize,
      params.scope);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "Subgroup should preserve thread_id_in_group, group_size, and scope";

  std::vector<uint32_t> groupSizes_h(totalGroups);

  CUDACHECK_TEST(cudaMemcpy(
      groupSizes_h.data(),
      groupSizes_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  for (uint32_t groupId = 0; groupId < totalGroups; groupId++) {
    EXPECT_EQ(groupSizes_h[groupId], expectedGroupSize)
        << "Group " << groupId
        << " subgroup should have group_size == " << expectedGroupSize;
  }
}

INSTANTIATE_TEST_SUITE_P(
    SubgroupPropertiesConfigs,
    ThreadGroupSubgroupPropertiesTest,
    ::testing::Values(
        SubgroupPropertiesTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numPartitions = 2,
            .scope = SyncScope::WARP,
            .testName = "Warp_TwoPartitions"},
        SubgroupPropertiesTestParams{
            .numBlocks = 8,
            .blockSize = 256,
            .numPartitions = 4,
            .scope = SyncScope::WARP,
            .testName = "Warp_FourPartitions"},
        SubgroupPropertiesTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numPartitions = 2,
            .scope = SyncScope::TILE,
            .testName = "Tile_TwoPartitions"},
        SubgroupPropertiesTestParams{
            .numBlocks = 8,
            .blockSize = 256,
            .numPartitions = 4,
            .scope = SyncScope::TILE,
            .testName = "Tile_FourPartitions"},
        SubgroupPropertiesTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numPartitions = 2,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_TwoPartitions"},
        SubgroupPropertiesTestParams{
            .numBlocks = 8,
            .blockSize = 256,
            .numPartitions = 4,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_FourPartitions"}),
    [](const ::testing::TestParamInfo<SubgroupPropertiesTestParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Weighted Partition Tests (Parameterized for WARP and TILE)
// =============================================================================

struct WeightedPartitionTestParams {
  std::vector<uint32_t> weights;
  int numBlocks;
  int blockSize;
  SyncScope scope;
  std::string testName;
};

class ThreadGroupWeightedPartitionTest
    : public ThreadGroupTestFixture,
      public ::testing::WithParamInterface<WeightedPartitionTestParams> {};

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

  if (totalWeight == 0) {
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
      boundary = partitionStart;
    } else {
      nonZeroSeen++;
      uint32_t proportionalGroups =
          (accumulatedWeight * distributableGroups + totalWeight - 1) /
          totalWeight;
      boundary = nonZeroSeen + proportionalGroups;

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
  const uint32_t numPartitions = static_cast<uint32_t>(params.weights.size());

  uint32_t totalGroups;
  if (params.scope == SyncScope::WARP) {
    totalGroups =
        params.numBlocks * (params.blockSize / comms::device::kWarpSize);
  } else if (params.scope == SyncScope::WARPGROUP) {
    totalGroups = params.numBlocks * (params.blockSize / kWarpgroupSize);
  } else {
    totalGroups = params.numBlocks;
  }

  DeviceBuffer partitionIdsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer subgroupIdsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer subgroupTotalGroupsBuffer(totalGroups * sizeof(uint32_t));
  DeviceBuffer weightsBuffer(numPartitions * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto partitionIds_d = static_cast<uint32_t*>(partitionIdsBuffer.get());
  auto subgroupIds_d = static_cast<uint32_t*>(subgroupIdsBuffer.get());
  auto subgroupTotalGroups_d =
      static_cast<uint32_t*>(subgroupTotalGroupsBuffer.get());
  auto weights_d = static_cast<uint32_t*>(weightsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(
      cudaMemset(partitionIds_d, 0xFF, totalGroups * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupIds_d, 0xFF, totalGroups * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(subgroupTotalGroups_d, 0xFF, totalGroups * sizeof(uint32_t)));
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
      params.blockSize,
      params.scope);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Weighted partition should not produce errors";

  std::vector<uint32_t> partitionIds_h(totalGroups);
  std::vector<uint32_t> subgroupIds_h(totalGroups);
  std::vector<uint32_t> subgroupTotalGroups_h(totalGroups);

  CUDACHECK_TEST(cudaMemcpy(
      partitionIds_h.data(),
      partitionIds_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupIds_h.data(),
      subgroupIds_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      subgroupTotalGroups_h.data(),
      subgroupTotalGroups_d,
      totalGroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  auto boundaries = computePartitionBoundaries(params.weights, totalGroups);

  for (uint32_t groupId = 0; groupId < totalGroups; groupId++) {
    uint32_t expectedPartition = 0;
    uint32_t partitionStart = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(boundaries.size()); i++) {
      if (groupId < boundaries[i]) {
        expectedPartition = i;
        break;
      }
      partitionStart = boundaries[i];
    }

    uint32_t partitionEnd = boundaries[expectedPartition];
    uint32_t expectedSubgroupId = groupId - partitionStart;
    uint32_t expectedTotalGroups = partitionEnd - partitionStart;

    EXPECT_EQ(partitionIds_h[groupId], expectedPartition)
        << "Group " << groupId << " should be in partition "
        << expectedPartition;

    EXPECT_EQ(subgroupIds_h[groupId], expectedSubgroupId)
        << "Group " << groupId << " should have subgroup.group_id "
        << expectedSubgroupId;

    EXPECT_EQ(subgroupTotalGroups_h[groupId], expectedTotalGroups)
        << "Group " << groupId << " should have subgroup.total_groups "
        << expectedTotalGroups;
  }

  std::vector<uint32_t> partitionCounts(numPartitions, 0);
  for (uint32_t groupId = 0; groupId < totalGroups; groupId++) {
    ASSERT_LT(partitionIds_h[groupId], numPartitions)
        << "Partition ID should be < " << numPartitions;
    partitionCounts[partitionIds_h[groupId]]++;
  }

  uint32_t distinctPartitions = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    if (partitionCounts[i] > 0) {
      distinctPartitions++;
    }
  }

  uint32_t expectedNonZeroPartitions = 0;
  for (auto w : params.weights) {
    if (w > 0) {
      expectedNonZeroPartitions++;
    }
  }

  EXPECT_EQ(distinctPartitions, expectedNonZeroPartitions)
      << "Should have " << expectedNonZeroPartitions
      << " distinct partition_ids "
      << "(totalGroups=" << totalGroups << ", numPartitions=" << numPartitions
      << ", nonZeroPartitions=" << expectedNonZeroPartitions << ").";

  uint32_t prevBoundary = 0;
  for (uint32_t i = 0; i < numPartitions; i++) {
    uint32_t expectedSize = boundaries[i] - prevBoundary;
    EXPECT_EQ(partitionCounts[i], expectedSize)
        << "Partition " << i << " should have " << expectedSize << " groups";
    prevBoundary = boundaries[i];
  }
}

INSTANTIATE_TEST_SUITE_P(
    WeightedPartitions,
    ThreadGroupWeightedPartitionTest,
    ::testing::Values(
        // WARP group tests
        WeightedPartitionTestParams{
            .weights = {1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_EvenSplit_2way"},
        WeightedPartitionTestParams{
            .weights = {3, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_Weighted_3_1"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 1},
            .numBlocks = 6,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_EvenSplit_3way"},
        WeightedPartitionTestParams{
            .weights = {2, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_Weighted_2_1_1"},
        WeightedPartitionTestParams{
            .weights = {1, 2, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_Weighted_1_2_1"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_UnevenRounding_3way"},
        WeightedPartitionTestParams{
            .weights = {99, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_ExtremeRatio_99_1"},
        WeightedPartitionTestParams{
            .weights = {1000, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_ExtremeRatio_1000_1"},
        WeightedPartitionTestParams{
            .weights = {1000, 1},
            .numBlocks = 1,
            .blockSize = 64,
            .scope = SyncScope::WARP,
            .testName = "Warp_ExtremeRatio_MinimumGuarantee"},
        WeightedPartitionTestParams{
            .weights = {1},
            .numBlocks = 4,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_SinglePartition"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_FourWaySplit"},
        WeightedPartitionTestParams{
            .weights = {3, 0, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_ZeroWeight_Middle"},
        WeightedPartitionTestParams{
            .weights = {0, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_ZeroWeight_First"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 0},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_ZeroWeight_Last"},
        WeightedPartitionTestParams{
            .weights = {1, 0, 0, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_ZeroWeight_MultipleMiddle"},
        WeightedPartitionTestParams{
            .weights = {0, 0, 1},
            .numBlocks = 4,
            .blockSize = 256,
            .scope = SyncScope::WARP,
            .testName = "Warp_ZeroWeight_MultipleFirst"},
        WeightedPartitionTestParams{
            .weights = {1, 0, 0, 0},
            .numBlocks = 1,
            .blockSize = 64,
            .scope = SyncScope::WARP,
            .testName = "Warp_ZeroWeight_MorePartitionsThanGroupsOK"},
        // TILE (Block) group tests
        WeightedPartitionTestParams{
            .weights = {1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_EvenSplit_2way"},
        WeightedPartitionTestParams{
            .weights = {3, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_Weighted_3_1"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 1},
            .numBlocks = 6,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_EvenSplit_3way"},
        WeightedPartitionTestParams{
            .weights = {2, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_Weighted_2_1_1"},
        WeightedPartitionTestParams{
            .weights = {1},
            .numBlocks = 4,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_SinglePartition"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_FourWaySplit"},
        WeightedPartitionTestParams{
            .weights = {3, 0, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_ZeroWeight_Middle"},
        WeightedPartitionTestParams{
            .weights = {0, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::TILE,
            .testName = "Tile_ZeroWeight_First"},
        // WARPGROUP (4 warps = 128 threads) tests
        WeightedPartitionTestParams{
            .weights = {1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_EvenSplit_2way"},
        WeightedPartitionTestParams{
            .weights = {3, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_Weighted_3_1"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 1},
            .numBlocks = 6,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_EvenSplit_3way"},
        WeightedPartitionTestParams{
            .weights = {2, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_Weighted_2_1_1"},
        WeightedPartitionTestParams{
            .weights = {1},
            .numBlocks = 4,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_SinglePartition"},
        WeightedPartitionTestParams{
            .weights = {1, 1, 1, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_FourWaySplit"},
        WeightedPartitionTestParams{
            .weights = {3, 0, 1},
            .numBlocks = 8,
            .blockSize = 256,
            .scope = SyncScope::WARPGROUP,
            .testName = "Warpgroup_ZeroWeight_Middle"}),
    [](const ::testing::TestParamInfo<WeightedPartitionTestParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Warpgroup Tests (4 warps = 128 threads per group)
// =============================================================================

// Test: make_warpgroup_group creates correct ThreadGroup
// Verifies:
// - group_id is computed correctly across all warpgroups
// - group_size == 128 (4 * warpSize)
// - thread_id_in_group == tid % 128
// - total_groups == (threads_per_block / 128) * num_blocks
// - Work items are distributed contiguously across warpgroup groups
TEST_F(ThreadGroupTestFixture, WarpgroupGroupContiguousLocality) {
  const uint32_t numItems = 1024;
  const int numBlocks = 4;
  const int blockSize = 512; // Must be multiple of 128 (warpgroup size)

  const uint32_t warpgroupsPerBlock = blockSize / kWarpgroupSize;
  const uint32_t totalWarpgroups = numBlocks * warpgroupsPerBlock;

  DeviceBuffer groupIdsBuffer(numItems * sizeof(uint32_t));
  DeviceBuffer threadIdsBuffer(numItems * sizeof(uint32_t));
  DeviceBuffer groupSizesBuffer(totalWarpgroups * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto groupIds_d = static_cast<uint32_t*>(groupIdsBuffer.get());
  auto threadIds_d = static_cast<uint32_t*>(threadIdsBuffer.get());
  auto groupSizes_d = static_cast<uint32_t*>(groupSizesBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemset(groupIds_d, 0, numItems * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(threadIds_d, 0, numItems * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(groupSizes_d, 0, totalWarpgroups * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testWarpgroupGroup(
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
  EXPECT_EQ(errorCount_h, 0) << "Warpgroup group should not have any errors";

  // Verify group sizes (all warpgroups should have size 128)
  std::vector<uint32_t> groupSizes_h(totalWarpgroups);
  CUDACHECK_TEST(cudaMemcpy(
      groupSizes_h.data(),
      groupSizes_d,
      totalWarpgroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < totalWarpgroups; i++) {
    EXPECT_EQ(groupSizes_h[i], kWarpgroupSize)
        << "Warpgroup " << i << " should have group_size == " << kWarpgroupSize;
  }

  // Verify contiguous distribution of work items
  std::vector<uint32_t> groupIds_h(numItems);
  CUDACHECK_TEST(cudaMemcpy(
      groupIds_h.data(),
      groupIds_d,
      numItems * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  const uint32_t itemsPerGroup =
      (numItems + totalWarpgroups - 1) / totalWarpgroups;

  for (uint32_t group_id = 0; group_id < totalWarpgroups; group_id++) {
    uint32_t start_item = group_id * itemsPerGroup;
    uint32_t end_item = std::min(start_item + itemsPerGroup, numItems);

    // Skip groups that have no items assigned
    if (start_item >= numItems) {
      break;
    }

    for (uint32_t item_id = start_item; item_id < end_item; item_id++) {
      EXPECT_EQ(groupIds_h[item_id], group_id)
          << "Work item " << item_id << " should be assigned to warpgroup "
          << group_id;
    }
  }
}

// Test: Warpgroup synchronization correctness
// Verifies:
// - All 128 threads in a warpgroup synchronize correctly via named barriers
// - Multiple warpgroups can synchronize independently within a block
// - sync() uses PTX bar.sync instruction correctly
TEST_F(ThreadGroupTestFixture, WarpgroupSync) {
  const int numBlocks = 2;
  const int blockSize = 512; // 4 warpgroups per block

  const uint32_t warpgroupsPerBlock = blockSize / kWarpgroupSize;
  const uint32_t totalWarpgroups = numBlocks * warpgroupsPerBlock;

  DeviceBuffer syncResultsBuffer(totalWarpgroups * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto syncResults_d = static_cast<uint32_t*>(syncResultsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(
      cudaMemset(syncResults_d, 0, totalWarpgroups * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testWarpgroupSync(syncResults_d, errorCount_d, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify no synchronization errors
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "Warpgroup sync should synchronize all 128 threads correctly";

  // Verify all warpgroups completed successfully
  std::vector<uint32_t> syncResults_h(totalWarpgroups);
  CUDACHECK_TEST(cudaMemcpy(
      syncResults_h.data(),
      syncResults_d,
      totalWarpgroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < totalWarpgroups; i++) {
    EXPECT_EQ(syncResults_h[i], 1U)
        << "Warpgroup " << i << " should have completed synchronization";
  }
}

// Parameterized test for warpgroup with different block sizes
struct WarpgroupTestParams {
  int numBlocks;
  int blockSize;
  uint32_t numItems;
  std::string testName;
};

class ThreadGroupWarpgroupTest
    : public ThreadGroupTestFixture,
      public ::testing::WithParamInterface<WarpgroupTestParams> {};

TEST_P(ThreadGroupWarpgroupTest, WarpgroupContiguousDistribution) {
  const auto& params = GetParam();
  const uint32_t warpgroupsPerBlock = params.blockSize / kWarpgroupSize;
  const uint32_t totalWarpgroups = params.numBlocks * warpgroupsPerBlock;
  const uint32_t numItems = params.numItems;

  DeviceBuffer groupIdsBuffer(numItems * sizeof(uint32_t));
  DeviceBuffer threadIdsBuffer(numItems * sizeof(uint32_t));
  DeviceBuffer groupSizesBuffer(totalWarpgroups * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto groupIds_d = static_cast<uint32_t*>(groupIdsBuffer.get());
  auto threadIds_d = static_cast<uint32_t*>(threadIdsBuffer.get());
  auto groupSizes_d = static_cast<uint32_t*>(groupSizesBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemset(groupIds_d, 0, numItems * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(threadIds_d, 0, numItems * sizeof(uint32_t)));
  CUDACHECK_TEST(
      cudaMemset(groupSizes_d, 0, totalWarpgroups * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testWarpgroupGroup(
      groupIds_d,
      threadIds_d,
      groupSizes_d,
      numItems,
      errorCount_d,
      params.numBlocks,
      params.blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify no errors
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0);

  // Verify all group sizes are 128
  std::vector<uint32_t> groupSizes_h(totalWarpgroups);
  CUDACHECK_TEST(cudaMemcpy(
      groupSizes_h.data(),
      groupSizes_d,
      totalWarpgroups * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < totalWarpgroups; i++) {
    EXPECT_EQ(groupSizes_h[i], kWarpgroupSize)
        << "Warpgroup " << i << " should have group_size == " << kWarpgroupSize;
  }

  // Verify contiguous work distribution
  std::vector<uint32_t> groupIds_h(numItems);
  CUDACHECK_TEST(cudaMemcpy(
      groupIds_h.data(),
      groupIds_d,
      numItems * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  const uint32_t itemsPerGroup =
      (numItems + totalWarpgroups - 1) / totalWarpgroups;

  for (uint32_t group_id = 0; group_id < totalWarpgroups; group_id++) {
    uint32_t start_item = group_id * itemsPerGroup;
    uint32_t end_item = std::min(start_item + itemsPerGroup, numItems);

    if (start_item >= numItems) {
      break;
    }

    for (uint32_t item_id = start_item; item_id < end_item; item_id++) {
      EXPECT_EQ(groupIds_h[item_id], group_id)
          << "Work item " << item_id << " should be assigned to warpgroup "
          << group_id;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    WarpgroupConfigs,
    ThreadGroupWarpgroupTest,
    ::testing::Values(
        // Single warpgroup per block
        WarpgroupTestParams{
            .numBlocks = 4,
            .blockSize = 128,
            .numItems = 512,
            .testName = "SingleWarpgroupPerBlock"},
        // Multiple warpgroups per block (4 warpgroups)
        WarpgroupTestParams{
            .numBlocks = 2,
            .blockSize = 512,
            .numItems = 1024,
            .testName = "FourWarpgroupsPerBlock"},
        // Large configuration (16 warpgroups total)
        WarpgroupTestParams{
            .numBlocks = 8,
            .blockSize = 256,
            .numItems = 2048,
            .testName = "SixteenWarpgroupsTotal"},
        // Uneven item distribution
        WarpgroupTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numItems = 1000,
            .testName = "UnevenItemDistribution"},
        // Maximum warpgroups per block (16 = 2048/128, hardware limit)
        WarpgroupTestParams{
            .numBlocks = 1,
            .blockSize = 1024,
            .numItems = 1024,
            .testName = "EightWarpgroupsPerBlock"}),
    [](const ::testing::TestParamInfo<WarpgroupTestParams>& info) {
      return info.param.testName;
    });

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
