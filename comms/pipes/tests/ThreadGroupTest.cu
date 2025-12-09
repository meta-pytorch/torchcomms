// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/tests/Utils.h"

namespace comms::pipes::test {

using namespace comms::pipes;

__global__ void testContiguousLocalityKernel(
    uint32_t* groupIds,
    uint32_t numItems,
    uint32_t* errorCount) {
  auto warp = make_warp_group();

  // Each warp writes its group_id to its assigned work items
  warp.for_each_item_contiguous(numItems, [&](uint32_t item_id) {
    if (item_id >= numItems) {
      atomicAdd(errorCount, 1);
      return;
    }

    groupIds[item_id] = warp.group_id;
  });

  __syncthreads();

  // Leader of first warp verifies the contiguous pattern on GPU
  // (CPU will also verify this for thoroughness)
  if (warp.is_global_leader()) {
    uint32_t items_per_group =
        (numItems + warp.total_groups - 1) / warp.total_groups;

    for (uint32_t group_id = 0; group_id < warp.total_groups; group_id++) {
      uint32_t start_item = group_id * items_per_group;
      uint32_t end_item = (start_item + items_per_group < numItems)
          ? start_item + items_per_group
          : numItems;

      // Verify all items in [start, end) belong to this group
      for (uint32_t item_id = start_item; item_id < end_item; item_id++) {
        if (groupIds[item_id] != group_id) {
          atomicAdd(errorCount, 1);
        }
      }
    }
  }
}

void testContiguousLocality(
    uint32_t* groupIds_d,
    uint32_t numItems,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testContiguousLocalityKernel<<<numBlocks, blockSize>>>(
      groupIds_d, numItems, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Block Group Tests
// =============================================================================

__global__ void testBlockGroupKernel(
    uint32_t* groupIds,
    uint32_t* threadIdsInGroup,
    uint32_t* groupSizes,
    uint32_t numItems,
    uint32_t* errorCount) {
  auto block = make_block_group();

  // Record group properties for verification
  if (threadIdx.x == 0) {
    groupSizes[blockIdx.x] = block.group_size;
  }

  // Each block writes its group_id to its assigned work items
  block.for_each_item_contiguous(numItems, [&](uint32_t item_id) {
    if (item_id >= numItems) {
      atomicAdd(errorCount, 1);
      return;
    }

    groupIds[item_id] = block.group_id;
    threadIdsInGroup[item_id] = block.thread_id_in_group;
  });
}

void testBlockGroup(
    uint32_t* groupIds_d,
    uint32_t* threadIdsInGroup_d,
    uint32_t* groupSizes_d,
    uint32_t numItems,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testBlockGroupKernel<<<numBlocks, blockSize>>>(
      groupIds_d, threadIdsInGroup_d, groupSizes_d, numItems, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Partition Tests
// =============================================================================

__global__ void testPartitionKernel(
    uint32_t* partitionIds,
    uint32_t* subgroupIds,
    uint32_t* subgroupTotalGroups,
    uint32_t numPartitions,
    uint32_t* errorCount) {
  auto warp = make_warp_group();

  auto [partition_id, subgroup] = warp.partition(numPartitions);

  // Bounds check
  if (partition_id >= numPartitions) {
    atomicAdd(errorCount, 1);
    return;
  }

  // Record results for CPU verification (one write per warp)
  if (warp.is_leader()) {
    partitionIds[warp.group_id] = partition_id;
    subgroupIds[warp.group_id] = subgroup.group_id;
    subgroupTotalGroups[warp.group_id] = subgroup.total_groups;
  }
}

void testPartition(
    uint32_t* partitionIds_d,
    uint32_t* subgroupIds_d,
    uint32_t* subgroupTotalGroups_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testPartitionKernel<<<numBlocks, blockSize>>>(
      partitionIds_d,
      subgroupIds_d,
      subgroupTotalGroups_d,
      numPartitions,
      errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Subgroup Properties Verification Tests
// =============================================================================

__global__ void testPartitionSubgroupPropertiesKernel(
    uint32_t* threadIdsInGroup,
    uint32_t* groupSizes,
    uint32_t* scopes,
    uint32_t numPartitions,
    uint32_t* errorCount) {
  auto warp = make_warp_group();

  auto [partition_id, subgroup] = warp.partition(numPartitions);

  // Verify preserved properties match original warp
  if (subgroup.thread_id_in_group != warp.thread_id_in_group) {
    atomicAdd(errorCount, 1);
  }
  if (subgroup.group_size != warp.group_size) {
    atomicAdd(errorCount, 1);
  }
  if (subgroup.scope != warp.scope) {
    atomicAdd(errorCount, 1);
  }

  // Record for CPU verification (one write per warp)
  if (warp.is_leader()) {
    threadIdsInGroup[warp.group_id] = subgroup.thread_id_in_group;
    groupSizes[warp.group_id] = subgroup.group_size;
    scopes[warp.group_id] = static_cast<uint32_t>(subgroup.scope);
  }
}

void testPartitionSubgroupProperties(
    uint32_t* threadIdsInGroup_d,
    uint32_t* groupSizes_d,
    uint32_t* scopes_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testPartitionSubgroupPropertiesKernel<<<numBlocks, blockSize>>>(
      threadIdsInGroup_d, groupSizes_d, scopes_d, numPartitions, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Weighted Partition Tests
// =============================================================================

__global__ void testWeightedPartitionKernel(
    uint32_t* partitionIds,
    uint32_t* subgroupIds,
    uint32_t* subgroupTotalGroups,
    const uint32_t* weights,
    uint32_t numPartitions,
    uint32_t* errorCount) {
  auto warp = make_warp_group();

  auto [partition_id, subgroup] =
      warp.partition(make_device_span(weights, numPartitions));

  // Bounds check
  if (partition_id >= numPartitions) {
    atomicAdd(errorCount, 1);
    return;
  }

  // Record results for CPU verification (one write per warp)
  if (warp.is_leader()) {
    partitionIds[warp.group_id] = partition_id;
    subgroupIds[warp.group_id] = subgroup.group_id;
    subgroupTotalGroups[warp.group_id] = subgroup.total_groups;
  }
}

void testWeightedPartition(
    uint32_t* partitionIds_d,
    uint32_t* subgroupIds_d,
    uint32_t* subgroupTotalGroups_d,
    const uint32_t* weights_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testWeightedPartitionKernel<<<numBlocks, blockSize>>>(
      partitionIds_d,
      subgroupIds_d,
      subgroupTotalGroups_d,
      weights_d,
      numPartitions,
      errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Invalid Partition Test (more partitions than groups)
// =============================================================================

__global__ void testWeightedPartitionMorePartitionsThanGroupsKernel(
    const uint32_t* weights,
    uint32_t numPartitions) {
  auto warp = make_warp_group();

  // This should trigger an assertion failure because numPartitions >
  // total_groups The assertion in partition(weights) checks: num_partitions <=
  // total_groups
  auto [partition_id, subgroup] =
      warp.partition(make_device_span(weights, numPartitions));

  // Suppress unused variable warnings (we won't reach here due to assertion)
  (void)partition_id;
  (void)subgroup;
}

void testWeightedPartitionMorePartitionsThanGroups(
    const uint32_t* weights_d,
    uint32_t numPartitions,
    int numBlocks,
    int blockSize) {
  testWeightedPartitionMorePartitionsThanGroupsKernel<<<numBlocks, blockSize>>>(
      weights_d, numPartitions);
  // No PIPES_KERNEL_LAUNCH_CHECK() here - this test expects the kernel to trap
}

// =============================================================================
// Invalid Partition Test (more partitions than groups) - Non-weighted version
// =============================================================================

__global__ void testPartitionMorePartitionsThanGroupsKernel(
    uint32_t numPartitions) {
  auto warp = make_warp_group();

  // This should trigger a trap because numPartitions > total_groups
  // The check in partition(num_partitions) validates: num_partitions <=
  // total_groups
  auto [partition_id, subgroup] = warp.partition(numPartitions);

  // Suppress unused variable warnings (we won't reach here due to trap)
  (void)partition_id;
  (void)subgroup;
}

void testPartitionMorePartitionsThanGroups(
    uint32_t numPartitions,
    int numBlocks,
    int blockSize) {
  testPartitionMorePartitionsThanGroupsKernel<<<numBlocks, blockSize>>>(
      numPartitions);
  // No PIPES_KERNEL_LAUNCH_CHECK() here - this test expects the kernel to trap
}

} // namespace comms::pipes::test
