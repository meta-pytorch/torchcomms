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
  } else if (scope == SyncScope::MULTIWARP) {
    testContiguousLocalityKernel<SyncScope::MULTIWARP>
        <<<numBlocks, blockSize>>>(groupIds_d, numItems, errorCount_d);
  } else if (scope == SyncScope::BLOCK) {
    testContiguousLocalityKernel<SyncScope::BLOCK>
        <<<numBlocks, blockSize>>>(groupIds_d, numItems, errorCount_d);
  } else {
    testContiguousLocalityKernel<SyncScope::CLUSTER>
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
  } else if (scope == SyncScope::MULTIWARP) {
    testPartitionKernel<SyncScope::MULTIWARP><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        numPartitions,
        errorCount_d);
  } else if (scope == SyncScope::BLOCK) {
    testPartitionKernel<SyncScope::BLOCK><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        numPartitions,
        errorCount_d);
  } else {
    testPartitionKernel<SyncScope::CLUSTER><<<numBlocks, blockSize>>>(
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
  } else if (scope == SyncScope::MULTIWARP) {
    testPartitionSubgroupPropertiesKernel<SyncScope::MULTIWARP>
        <<<numBlocks, blockSize>>>(
            threadIdsInGroup_d,
            groupSizes_d,
            scopes_d,
            numPartitions,
            errorCount_d);
  } else if (scope == SyncScope::BLOCK) {
    testPartitionSubgroupPropertiesKernel<SyncScope::BLOCK>
        <<<numBlocks, blockSize>>>(
            threadIdsInGroup_d,
            groupSizes_d,
            scopes_d,
            numPartitions,
            errorCount_d);
  } else {
    testPartitionSubgroupPropertiesKernel<SyncScope::CLUSTER>
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
  } else if (scope == SyncScope::MULTIWARP) {
    testPartitionInterleavedKernel<SyncScope::MULTIWARP>
        <<<numBlocks, blockSize>>>(
            partitionIds_d,
            subgroupIds_d,
            subgroupTotalGroups_d,
            numPartitions,
            errorCount_d);
  } else if (scope == SyncScope::BLOCK) {
    testPartitionInterleavedKernel<SyncScope::BLOCK><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        numPartitions,
        errorCount_d);
  } else {
    testPartitionInterleavedKernel<SyncScope::CLUSTER>
        <<<numBlocks, blockSize>>>(
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
  } else if (scope == SyncScope::MULTIWARP) {
    testWeightedPartitionKernel<SyncScope::MULTIWARP><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        weights_d,
        numPartitions,
        errorCount_d);
  } else if (scope == SyncScope::BLOCK) {
    testWeightedPartitionKernel<SyncScope::BLOCK><<<numBlocks, blockSize>>>(
        partitionIds_d,
        subgroupIds_d,
        subgroupTotalGroups_d,
        weights_d,
        numPartitions,
        errorCount_d);
  } else {
    testWeightedPartitionKernel<SyncScope::CLUSTER><<<numBlocks, blockSize>>>(
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
// Multiwarp Tests (4 warps = 128 threads per group)
// =============================================================================

__global__ void testMultiwarpGroupKernel(
    uint32_t* groupIds,
    uint32_t* threadIdsInGroup,
    uint32_t* groupSizes,
    uint32_t numItems,
    uint32_t* errorCount) {
  auto multiwarp = make_multiwarp_group();

  // Record group properties for verification (one write per multiwarp)
  if (multiwarp.is_leader()) {
    groupSizes[multiwarp.group_id] = multiwarp.group_size;
  }

  // Each multiwarp writes its group_id to its assigned work items
  multiwarp.for_each_item_contiguous(numItems, [&](uint32_t item_id) {
    if (item_id >= numItems) {
      atomicAdd(errorCount, 1);
      return;
    }

    groupIds[item_id] = multiwarp.group_id;
    threadIdsInGroup[item_id] = multiwarp.thread_id_in_group;
  });
}

void testMultiwarpGroup(
    uint32_t* groupIds_d,
    uint32_t* threadIdsInGroup_d,
    uint32_t* groupSizes_d,
    uint32_t numItems,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testMultiwarpGroupKernel<<<numBlocks, blockSize>>>(
      groupIds_d, threadIdsInGroup_d, groupSizes_d, numItems, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// Test multiwarp synchronization using named barriers
// This test verifies that all 128 threads in a multiwarp synchronize correctly.
// Each thread writes a value, then after sync, verifies all threads wrote.
__global__ void testMultiwarpSyncKernel(
    uint32_t* syncResults,
    uint32_t* errorCount) {
  __shared__ uint32_t sharedData[2048]; // Support up to 2048 threads per block

  auto multiwarp = make_multiwarp_group();

  uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;

  // Phase 1: Each thread writes its thread ID to shared memory
  sharedData[tid] = tid + 1; // +1 so we can distinguish from zero-initialized

  // Synchronize within multiwarp using named barrier
  multiwarp.sync();

  // Phase 2: Each thread verifies all threads in its multiwarp wrote their
  // values
  constexpr uint32_t kMultiwarpSize = 128;
  uint32_t multiwarpStart = (tid / kMultiwarpSize) * kMultiwarpSize;

  for (uint32_t i = 0; i < kMultiwarpSize; i++) {
    uint32_t expectedTid = multiwarpStart + i;
    if (sharedData[expectedTid] != expectedTid + 1) {
      atomicAdd(errorCount, 1);
    }
  }

  // Record success (one write per multiwarp)
  if (multiwarp.is_leader()) {
    syncResults[multiwarp.group_id] = 1;
  }
}

void testMultiwarpSync(
    uint32_t* syncResults_d,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testMultiwarpSyncKernel<<<numBlocks, blockSize>>>(
      syncResults_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Cluster Tests (Hopper SM90+ cluster synchronization)
// =============================================================================

// Kernel for testing make_cluster_group()
// All blocks in a cluster form one group
__global__ void testBlockClusterGroupKernel(
    uint32_t* groupIds,
    uint32_t* threadIdsInGroup,
    uint32_t* groupSizes,
    uint32_t numItems,
    uint32_t* errorCount) {
  auto cluster = make_cluster_group();

  // Record group properties for verification (one write per cluster)
  if (cluster.is_leader()) {
    groupSizes[cluster.group_id] = cluster.group_size;
  }

  // Each cluster writes its group_id to its assigned work items
  cluster.for_each_item_contiguous(numItems, [&](uint32_t item_id) {
    if (item_id >= numItems) {
      atomicAdd(errorCount, 1);
      return;
    }

    groupIds[item_id] = cluster.group_id;
    threadIdsInGroup[item_id] = cluster.thread_id_in_group;
  });
}

// Test block cluster synchronization using barrier.cluster.arrive/wait
// Each thread writes to shared memory, then after cluster sync,
// verifies all threads in the cluster wrote their values.
__global__ void testBlockClusterSyncKernel(
    uint32_t* syncResults,
    uint32_t* errorCount) {
  // Use distributed shared memory for cluster-wide communication
  // Each block has its own shared memory portion
  __shared__ uint32_t sharedData[1024]; // Support up to 1024 threads per block

  auto cluster = make_cluster_group();

  uint32_t tid_in_block = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;

  // Phase 1: Each thread writes to its block's shared memory
  if (tid_in_block < 1024) {
    sharedData[tid_in_block] = tid_in_block + 1; // +1 to distinguish from zero
  }

  // Synchronize within cluster
  cluster.sync();

  // Phase 2: Verify local block's shared memory is consistent
  // (cluster sync ensures all threads across cluster have reached this point)
  if (tid_in_block < 1024) {
    // Verify threads in this block wrote correctly
    uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = 0; i < threads_per_block && i < 1024; i++) {
      if (sharedData[i] != i + 1) {
        atomicAdd(errorCount, 1);
      }
    }
  }

  // Record success (one write per cluster)
  if (cluster.is_leader()) {
    syncResults[cluster.group_id] = 1;
  }
}

} // namespace comms::pipes::test
