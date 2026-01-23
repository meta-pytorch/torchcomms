// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/ThreadGroupTest.cuh"

namespace comms::pipes::test {

using namespace comms::pipes;

// =============================================================================
// Contiguous Locality Tests
// =============================================================================

template <SyncScope Scope>
__global__ void testContiguousLocalityKernel(
    uint32_t* groupIds,
    uint32_t numItems,
    uint32_t* errorCount) {
  auto group = make_thread_group(Scope);

  group.for_each_item_contiguous(numItems, [&](uint32_t item_id) {
    if (item_id >= numItems) {
      atomicAdd(errorCount, 1);
      return;
    }

    groupIds[item_id] = group.group_id;
  });

  __syncthreads();

  if (group.is_global_leader()) {
    uint32_t items_per_group =
        (numItems + group.total_groups - 1) / group.total_groups;

    for (uint32_t group_id = 0; group_id < group.total_groups; group_id++) {
      uint32_t start_item = group_id * items_per_group;
      uint32_t end_item = (start_item + items_per_group < numItems)
          ? start_item + items_per_group
          : numItems;

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
    int blockSize,
    SyncScope scope) {
  if (scope == SyncScope::WARP) {
    testContiguousLocalityKernel<SyncScope::WARP>
        <<<numBlocks, blockSize>>>(groupIds_d, numItems, errorCount_d);
  } else if (scope == SyncScope::WARPGROUP) {
    testContiguousLocalityKernel<SyncScope::WARPGROUP>
        <<<numBlocks, blockSize>>>(groupIds_d, numItems, errorCount_d);
  } else {
    testContiguousLocalityKernel<SyncScope::TILE>
        <<<numBlocks, blockSize>>>(groupIds_d, numItems, errorCount_d);
  }
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

  if (threadIdx.x == 0) {
    groupSizes[blockIdx.x] = block.group_size;
  }

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

template <SyncScope Scope>
__global__ void testPartitionKernel(
    uint32_t* partitionIds,
    uint32_t* subgroupIds,
    uint32_t* subgroupTotalGroups,
    uint32_t numPartitions,
    uint32_t* errorCount) {
  auto group = make_thread_group(Scope);

  auto [partition_id, subgroup] = group.partition(numPartitions);

  if (partition_id >= numPartitions) {
    atomicAdd(errorCount, 1);
    return;
  }

  if (group.is_leader()) {
    partitionIds[group.group_id] = partition_id;
    subgroupIds[group.group_id] = subgroup.group_id;
    subgroupTotalGroups[group.group_id] = subgroup.total_groups;
  }
}

void testPartition(
    uint32_t* partitionIds_d,
    uint32_t* subgroupIds_d,
    uint32_t* subgroupTotalGroups_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize,
    SyncScope scope) {
  if (scope == SyncScope::WARP) {
    testPartitionKernel<SyncScope::WARP><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        numPartitions,
        errorCount_d);
  } else if (scope == SyncScope::WARPGROUP) {
    testPartitionKernel<SyncScope::WARPGROUP><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        numPartitions,
        errorCount_d);
  } else {
    testPartitionKernel<SyncScope::TILE><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        numPartitions,
        errorCount_d);
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Subgroup Properties Verification Tests
// =============================================================================

template <SyncScope Scope>
__global__ void testPartitionSubgroupPropertiesKernel(
    uint32_t* threadIdsInGroup,
    uint32_t* groupSizes,
    uint32_t* scopes,
    uint32_t numPartitions,
    uint32_t* errorCount) {
  auto group = make_thread_group(Scope);

  auto [partition_id, subgroup] = group.partition(numPartitions);

  if (subgroup.thread_id_in_group != group.thread_id_in_group) {
    atomicAdd(errorCount, 1);
  }
  if (subgroup.group_size != group.group_size) {
    atomicAdd(errorCount, 1);
  }
  if (subgroup.scope != group.scope) {
    atomicAdd(errorCount, 1);
  }

  if (group.is_leader()) {
    threadIdsInGroup[group.group_id] = subgroup.thread_id_in_group;
    groupSizes[group.group_id] = subgroup.group_size;
    scopes[group.group_id] = static_cast<uint32_t>(subgroup.scope);
  }
}

void testPartitionSubgroupProperties(
    uint32_t* threadIdsInGroup_d,
    uint32_t* groupSizes_d,
    uint32_t* scopes_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize,
    SyncScope scope) {
  if (scope == SyncScope::WARP) {
    testPartitionSubgroupPropertiesKernel<SyncScope::WARP>
        <<<numBlocks, blockSize>>>(
            threadIdsInGroup_d,
            groupSizes_d,
            scopes_d,
            numPartitions,
            errorCount_d);
  } else if (scope == SyncScope::WARPGROUP) {
    testPartitionSubgroupPropertiesKernel<SyncScope::WARPGROUP>
        <<<numBlocks, blockSize>>>(
            threadIdsInGroup_d,
            groupSizes_d,
            scopes_d,
            numPartitions,
            errorCount_d);
  } else {
    testPartitionSubgroupPropertiesKernel<SyncScope::TILE>
        <<<numBlocks, blockSize>>>(
            threadIdsInGroup_d,
            groupSizes_d,
            scopes_d,
            numPartitions,
            errorCount_d);
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Partition Interleaved Tests
// =============================================================================

template <SyncScope Scope>
__global__ void testPartitionInterleavedKernel(
    uint32_t* partitionIds,
    uint32_t* subgroupIds,
    uint32_t* subgroupTotalGroups,
    uint32_t numPartitions,
    uint32_t* errorCount) {
  auto group = make_thread_group(Scope);

  auto [partition_id, subgroup] = group.partition_interleaved(numPartitions);

  if (partition_id >= numPartitions) {
    atomicAdd(errorCount, 1);
    return;
  }

  if (group.is_leader()) {
    partitionIds[group.group_id] = partition_id;
    subgroupIds[group.group_id] = subgroup.group_id;
    subgroupTotalGroups[group.group_id] = subgroup.total_groups;
  }
}

void testPartitionInterleaved(
    uint32_t* partitionIds_d,
    uint32_t* subgroupIds_d,
    uint32_t* subgroupTotalGroups_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize,
    SyncScope scope) {
  if (scope == SyncScope::WARP) {
    testPartitionInterleavedKernel<SyncScope::WARP><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        numPartitions,
        errorCount_d);
  } else if (scope == SyncScope::WARPGROUP) {
    testPartitionInterleavedKernel<SyncScope::WARPGROUP>
        <<<numBlocks, blockSize>>>(
            partitionIds_d,
            subgroupIds_d,
            subgroupTotalGroups_d,
            numPartitions,
            errorCount_d);
  } else {
    testPartitionInterleavedKernel<SyncScope::TILE><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        numPartitions,
        errorCount_d);
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Weighted Partition Tests
// =============================================================================

template <SyncScope Scope>
__global__ void testWeightedPartitionKernel(
    uint32_t* partitionIds,
    uint32_t* subgroupIds,
    uint32_t* subgroupTotalGroups,
    const uint32_t* weights,
    uint32_t numPartitions,
    uint32_t* errorCount) {
  auto group = make_thread_group(Scope);

  auto [partition_id, subgroup] =
      group.partition(make_device_span(weights, numPartitions));

  if (partition_id >= numPartitions) {
    atomicAdd(errorCount, 1);
    return;
  }

  if (group.is_leader()) {
    partitionIds[group.group_id] = partition_id;
    subgroupIds[group.group_id] = subgroup.group_id;
    subgroupTotalGroups[group.group_id] = subgroup.total_groups;
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
    int blockSize,
    SyncScope scope) {
  if (scope == SyncScope::WARP) {
    testWeightedPartitionKernel<SyncScope::WARP><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        weights_d,
        numPartitions,
        errorCount_d);
  } else if (scope == SyncScope::WARPGROUP) {
    testWeightedPartitionKernel<SyncScope::WARPGROUP><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        weights_d,
        numPartitions,
        errorCount_d);
  } else {
    testWeightedPartitionKernel<SyncScope::TILE><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        weights_d,
        numPartitions,
        errorCount_d);
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Warpgroup Tests (4 warps = 128 threads per group)
// =============================================================================

__global__ void testWarpgroupGroupKernel(
    uint32_t* groupIds,
    uint32_t* threadIdsInGroup,
    uint32_t* groupSizes,
    uint32_t numItems,
    uint32_t* errorCount) {
  auto warpgroup = make_warpgroup_group();

  // Record group properties for verification (one write per warpgroup)
  if (warpgroup.is_leader()) {
    groupSizes[warpgroup.group_id] = warpgroup.group_size;
  }

  // Each warpgroup writes its group_id to its assigned work items
  warpgroup.for_each_item_contiguous(numItems, [&](uint32_t item_id) {
    if (item_id >= numItems) {
      atomicAdd(errorCount, 1);
      return;
    }

    groupIds[item_id] = warpgroup.group_id;
    threadIdsInGroup[item_id] = warpgroup.thread_id_in_group;
  });
}

void testWarpgroupGroup(
    uint32_t* groupIds_d,
    uint32_t* threadIdsInGroup_d,
    uint32_t* groupSizes_d,
    uint32_t numItems,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testWarpgroupGroupKernel<<<numBlocks, blockSize>>>(
      groupIds_d, threadIdsInGroup_d, groupSizes_d, numItems, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// Test warpgroup synchronization using named barriers
// This test verifies that all 128 threads in a warpgroup synchronize correctly.
// Each thread writes a value, then after sync, verifies all threads wrote.
__global__ void testWarpgroupSyncKernel(
    uint32_t* syncResults,
    uint32_t* errorCount) {
  __shared__ uint32_t sharedData[2048]; // Support up to 2048 threads per block

  auto warpgroup = make_warpgroup_group();

  uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;

  // Phase 1: Each thread writes its thread ID to shared memory
  sharedData[tid] = tid + 1; // +1 so we can distinguish from zero-initialized

  // Synchronize within warpgroup using named barrier
  warpgroup.sync();

  // Phase 2: Each thread verifies all threads in its warpgroup wrote their
  // values
  constexpr uint32_t kWarpgroupSize = 128;
  uint32_t warpgroupStart = (tid / kWarpgroupSize) * kWarpgroupSize;

  for (uint32_t i = 0; i < kWarpgroupSize; i++) {
    uint32_t expectedTid = warpgroupStart + i;
    if (sharedData[expectedTid] != expectedTid + 1) {
      atomicAdd(errorCount, 1);
    }
  }

  // Record success (one write per warpgroup)
  if (warpgroup.is_leader()) {
    syncResults[warpgroup.group_id] = 1;
  }
}

void testWarpgroupSync(
    uint32_t* syncResults_d,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testWarpgroupSyncKernel<<<numBlocks, blockSize>>>(
      syncResults_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
