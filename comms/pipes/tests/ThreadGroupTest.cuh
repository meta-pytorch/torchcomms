// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::test {

// Kernel: testContiguousLocalityKernel
// Tests that for_each_item_contiguous assigns CONTIGUOUS blocks of work items
// to each group. Each group writes its group_id to all work items it processes.
// The CPU then verifies that work items [start, end) all have the same
// group_id, confirming contiguous-based assignment.
void testContiguousLocality(
    uint32_t* groupIds_d,
    uint32_t numItems,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize,
    SyncScope scope);

// Tests make_thread_solo() - where each thread forms its own group of size 1
// Verifies:
// - group_size == 1 for every thread
// - thread_id_in_group == 0 for every thread (always the leader)
// - is_leader() == true for every thread
// - group_id == global_thread_index (unique per thread)
// - total_groups == total thread count (numBlocks * blockSize)
// - scope == SyncScope::THREAD
// - sync() completes without deadlock (compiler barrier only, no hardware sync)
void testThreadSoloGroup(
    uint32_t* groupIds_d,
    uint32_t* groupSizes_d,
    uint32_t* threadIdsInGroup_d,
    uint32_t* isLeader_d,
    uint32_t* syncResults_d,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize);

// Tests make_block_group() - where all threads in a block form one group
// Verifies:
// - group_id == blockIdx.x
// - group_size == blockDim.x
// - thread_id_in_group == threadIdx.x
// - total_groups == gridDim.x
void testBlockGroup(
    uint32_t* groupIds_d,
    uint32_t* threadIdsInGroup_d,
    uint32_t* groupSizes_d,
    uint32_t numItems,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize);

// Tests partition(num_partitions) - even partition of groups
// Verifies:
// - Each group gets a valid partition_id in [0, num_partitions)
// - subgroup.group_id is renumbered within partition
// - subgroup.total_groups is correct for each partition
void testPartition(
    uint32_t* partitionIds_d,
    uint32_t* subgroupIds_d,
    uint32_t* subgroupTotalGroups_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize,
    SyncScope scope);

// Tests that subgroup preserves thread_id_in_group, group_size, and scope
// from the original group
void testPartitionSubgroupProperties(
    uint32_t* threadIdsInGroup_d,
    uint32_t* groupSizes_d,
    uint32_t* scopes_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize,
    SyncScope scope);

// Tests partition(cuda::std::span<const uint32_t>) - weighted partition
// Verifies proportional assignment based on weights
void testWeightedPartition(
    uint32_t* partitionIds_d,
    uint32_t* subgroupIds_d,
    uint32_t* subgroupTotalGroups_d,
    const uint32_t* weights_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize,
    SyncScope scope);

// Tests partition_interleaved(num_partitions) - round-robin partition
// Verifies:
// - Each group gets partition_id = group_id % num_partitions
// - subgroup.group_id is renumbered as group_id / num_partitions
// - subgroup.total_groups is correctly computed for interleaved assignment
void testPartitionInterleaved(
    uint32_t* partitionIds_d,
    uint32_t* subgroupIds_d,
    uint32_t* subgroupTotalGroups_d,
    uint32_t numPartitions,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize,
    SyncScope scope);

// =============================================================================
// Multiwarp Tests (4 warps = 128 threads per group)
// =============================================================================

// Tests make_multiwarp_group() - where 4 warps (128 threads) form one group
// Verifies:
// - group_size == 128 (4 * warpSize)
// - thread_id_in_group == tid % 128 (linear thread ID within multiwarp)
// - group_id is computed correctly across all multiwarps
// - total_groups == (threads_per_block / 128) * num_blocks
// - Work items are distributed contiguously across multiwarps
void testMultiwarpGroup(
    uint32_t* groupIds_d,
    uint32_t* threadIdsInGroup_d,
    uint32_t* groupSizes_d,
    uint32_t numItems,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize);

// Tests multiwarp synchronization using named barriers
// Verifies:
// - All 128 threads in a multiwarp synchronize correctly
// - sync() uses bar.sync PTX instruction with correct barrier ID
// - Multiple multiwarps can synchronize independently within a block
void testMultiwarpSync(
    uint32_t* syncResults_d,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize);

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
    uint32_t* errorCount);

// Test cluster synchronization using barrier.cluster.arrive/wait
// Each thread writes to shared memory, then after cluster sync,
// verifies all threads in the cluster wrote their values.
__global__ void testBlockClusterSyncKernel(
    uint32_t* syncResults,
    uint32_t* errorCount);

} // namespace comms::pipes::test
