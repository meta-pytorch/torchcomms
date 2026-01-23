// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/common/AtomicUtils.cuh"
#include "comms/common/DeviceConstants.cuh"
#include "comms/pipes/DeviceSpan.cuh"

namespace comms::pipes {

enum class SyncScope { WARP, WARPGROUP, TILE };

/**
 * ThreadGroup - Abstraction for cooperative thread group operations
 *
 * Represents a group of threads that work together on parallel tasks.
 * Typically created with make_warp_group() for 32-thread warps or
 * make_tile_group(N) for custom-sized groups.
 *
 * KEY CONCEPTS:
 * =============
 *
 * Example kernel configuration:
 *   - Launch: 4 blocks × 256 threads/block = 1024 total threads
 *   - Groups: Using warps (32 threads each)
 *   - Result: 32 total warps (4 blocks × 8 warps/block)
 *
 * Example breakdown for thread at global position 290:
 *   - blockIdx=1, threadIdx=34 (global_thread_id = 290)
 *   - group_id = 9 (warp 1 in block 1: 8 warps in block 0 + 1 warp)
 *   - thread_id_in_group = 2 (position within warp: 34 % 32 = 2)
 */
struct ThreadGroup {
  // LOCAL IDENTITY (within group):
  // ===============================

  // thread_id_in_group - Local thread ID within group [0, group_size)
  // For warps: lane ID [0..31]. Use for strided loops, leader checks, shuffles.
  uint32_t thread_id_in_group;

  // group_size - Number of threads in this group
  // Common values: 32 (warps), 64/128/256 (tiles)
  uint32_t group_size;

  // GLOBAL IDENTITY (across entire kernel):
  // ========================================

  // group_id - Global group ID across entire kernel [0, total_groups)
  // For warps: global warp ID. Use for work distribution.
  uint32_t group_id;

  // total_groups - Total number of groups in entire kernel
  // For warps: gridDim.x × (blockDim.x / 32)
  uint32_t total_groups;

  // SYNCHRONIZATION:
  // ================

  // scope - Synchronization scope for sync() calls
  // WARP: uses __syncwarp() (fast). TILE: uses __syncthreads() (block-wide).
  SyncScope scope;

  __device__ inline void sync() {
#ifdef __CUDA_ARCH__
    switch (scope) {
      case SyncScope::WARP:
        __syncwarp();
        break;
      case SyncScope::WARPGROUP: {
        // Warpgroup = 4 warps = 128 threads
        // Uses named barriers for synchronization within a warpgroup
        constexpr uint32_t kWarpgroupSize =
            4 * comms::device::kWarpSize; // 128 threads
        uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
        uint32_t barrierId = tid / kWarpgroupSize;
        // Hardware supports max 16 named barriers per block
        // This limits block size to 16 * 128 = 2048 threads
        asm volatile("bar.sync %0, %1;"
                     :
                     : "r"(barrierId), "r"(kWarpgroupSize));
        break;
      }
      case SyncScope::TILE:
        __syncthreads();
        break;
    }
#endif
  }

  __device__ inline bool is_leader() const {
    return thread_id_in_group == 0;
  }

  __device__ inline bool is_global_leader() const {
    return is_leader() && group_id == 0;
  }

  /**
   * for_each_item_contiguous - Distribute work items using CONTIGUOUS
   * assignment
   *
   * WHAT IT DOES:
   * Assigns each thread-group a contiguous block of work items to maximize
   * cache locality. A work item is a unit of work assigned to one thread-group.
   * The thread-group processes multiple work items in contiguous memory.
   * All threads in the group execute the lambda for each assigned work item.
   *
   * MAPPING FORMULA:
   * items_per_group = ceil(total_items / total_groups)
   * Group K processes work items: [K × items_per_group, (K+1) ×
   * items_per_group)
   *
   * EXAMPLE (2040 work items, 32 warps):
   * ================================
   *
   * items_per_group = ceil(2040/32) = 64
   *
   * Assignment:
   *   Warp 0:  [0..63]       Warp 1:  [64..127]     Warp 2:  [128..191]
   *   Warp 30: [1920..1983]  Warp 31: [1984..2039] ← Last warp: 56 items
   *
   * Thread execution (Warp 5 processing work items [320..383]):
   *   - Work item 320: All 32 threads execute lambda(320) simultaneously
   *   - Work item 321: All 32 threads execute lambda(321) simultaneously
   *   - ... (threads cooperate within lambda using thread_id_in_group)
   *
   * SIMPLE USAGE EXAMPLE:
   * =====================
   *   auto warp = make_warp_group();
   *
   *   // Process 2048 items, each warp processes ~64 contiguous items
   *   warp.for_each_item_contiguous(2048, [&](uint32_t item_id) {
   *     // All 32 threads in warp execute this for each item
   *     if (warp.is_leader()) {
   *       // Leader does atomic operation
   *       atomicAdd(&counters[buffer_id], 1);
   *     }
   *     // Or all threads cooperate on the item
   *     for (int i = warp.thread_id_in_group; i < item_size; i += 32) {
   *       output[item_id][i] = input[item_id][i] * 2;
   *     }
   *   });
   *
   * MEMORY ACCESS PATTERN (work items in contiguous memory):
   * ====================================================
   *
   * CONTIGUOUS (this method):
   *   Warp 0 → [0..63]   Warp 1 → [64..127]   ← CONTIGUOUS access
   *   ✅ Cache hits, optimal coalescing
   *
   * STRIDED (alternative):
   *   Warp 0 → [0,32,64,...]   Warp 1 → [1,33,65,...]   ← SCATTERED access
   *   ❌ Cache misses, poor coalescing
   *
   * @param total_items Total number of work items to distribute
   * @param func Lambda: void(uint32_t item_id) - executed by all threads
   */
  template <typename Func>
  __device__ inline void for_each_item_contiguous(
      uint32_t total_items,
      Func&& func) {
#ifdef __CUDA_ARCH__
    const uint32_t items_per_group =
        (total_items + total_groups - 1) / total_groups;
    const uint32_t start_item = group_id * items_per_group;
    const uint32_t end_item = (start_item + items_per_group < total_items)
        ? start_item + items_per_group
        : total_items;

    for (uint32_t item_id = start_item; item_id < end_item; item_id++) {
      func(item_id);
    }
#endif
  }

  // Partition methods declared here, defined after PartitionResult
  __device__ inline struct PartitionResult partition(
      uint32_t num_partitions) const;
  __device__ inline struct PartitionResult partition(
      DeviceSpan<const uint32_t> weights) const;
  __device__ inline struct PartitionResult partition_interleaved(
      uint32_t num_partitions) const;
};

/**
 * PartitionResult - Result of partitioning a ThreadGroup
 */
struct PartitionResult {
  uint32_t partition_id;
  ThreadGroup subgroup;
};

// Partition method implementations (after PartitionResult is defined)

/**
 * partition - Divide groups evenly into partitions
 *
 * Divides groups into N equal partitions and returns which partition
 * this group belongs to along with a renumbered subgroup.
 *
 * REQUIREMENTS:
 * =============
 * - num_partitions must be <= total_groups
 * - If num_partitions > total_groups, the kernel will trap with an error
 *   message showing both values
 *
 * WHY THIS CONSTRAINT:
 * ====================
 * When num_partitions > total_groups, some partitions would receive zero
 * groups, and group assignment would skip partitions non-deterministically
 * based on rounding. This is almost always a bug in the caller's logic.
 *
 * EXAMPLE (32 warps, 2 partitions):
 * ==================================
 *   auto [partition_id, subgroup] = warp.partition(2);
 *   if (partition_id == 0) {
 *     p2p.send(subgroup, sendBuff, nBytes);  // warps 0-15
 *   } else {
 *     p2p.recv(subgroup, recvBuff, nBytes);  // warps 16-31
 *   }
 *
 * @param num_partitions Number of partitions to create (must be <=
 * total_groups)
 * @return {partition_id, subgroup} for this group
 */
__device__ inline PartitionResult ThreadGroup::partition(
    uint32_t num_partitions) const {
#ifdef __CUDA_ARCH__
  // More partitions than groups is invalid - some partitions would be empty
  // and group assignment would skip partitions non-deterministically.
  // Use __trap() instead of assert() to ensure this check is active in both
  // debug and release builds.
  if (num_partitions > total_groups) {
    printf(
        "partition(num_partitions): num_partitions (%u) must be <= total_groups (%u)\n",
        num_partitions,
        total_groups);
    __trap();
  }

  // Use floor division and distribute remainder to first partitions
  // groups_per_partition = total_groups / num_partitions (floor)
  // remainder = total_groups % num_partitions
  // First 'remainder' partitions get (groups_per_partition + 1) groups
  // Remaining partitions get groups_per_partition groups
  //
  // Example: 32 warps / 15 partitions
  //   groups_per_partition = 32 / 15 = 2
  //   remainder = 32 % 15 = 2
  //   Partition 0: [0,3) - 3 groups
  //   Partition 1: [3,6) - 3 groups
  //   Partition 2-14: 2 groups each
  const uint32_t groups_per_partition = total_groups / num_partitions;
  const uint32_t remainder = total_groups % num_partitions;

  // Boundary between larger and smaller partitions
  const uint32_t boundary = remainder * (groups_per_partition + 1);

  uint32_t pid;
  uint32_t partition_start;
  uint32_t partition_size;

  if (group_id < boundary) {
    // This group is in one of the first 'remainder' partitions (larger size)
    pid = group_id / (groups_per_partition + 1);
    partition_start = pid * (groups_per_partition + 1);
    partition_size = groups_per_partition + 1;
  } else {
    // This group is in one of the remaining partitions (normal size)
    uint32_t offset = group_id - boundary;
    uint32_t partition_offset = offset / groups_per_partition;
    pid = remainder + partition_offset;
    partition_start = boundary + partition_offset * groups_per_partition;
    partition_size = groups_per_partition;
  }

  return PartitionResult{
      .partition_id = pid,
      .subgroup = ThreadGroup{
          .thread_id_in_group = thread_id_in_group,
          .group_size = group_size,
          .group_id = group_id - partition_start,
          .total_groups = partition_size,
          .scope = scope}};
#endif
  return PartitionResult{};
}

/**
 * partition - Divide groups according to weights
 *
 * Divides groups into N partitions proportionally based on weights.
 * All groups are assigned to exactly one partition.
 *
 * REQUIREMENTS:
 * =============
 * - The number of partitions with non-zero weight must be <= total_groups
 * - If non_zero_weight_count > total_groups, the kernel will trap with an error
 *   message showing both values
 *
 * GUARANTEES:
 * ===========
 * - Each partition with non-zero weight receives at least 1 group
 * - Partitions with zero weight receive 0 groups
 * - Groups are distributed proportionally to weights (after minimum guarantee)
 * - All groups are assigned to exactly one partition
 *
 * ALGORITHM: Reserve-Then-Distribute (with zero-weight handling)
 * ===============================================================
 * For partitions with non-zero weight:
 *   Each gets: 1 (guaranteed) + proportional share of remaining groups
 *
 *   partition_end[i] = non_zero_count_so_far + ceil(accumulated_weight *
 *                      distributable / total_weight)
 *
 * where distributable = total_groups - non_zero_weight_count
 *
 * For partitions with zero weight:
 *   partition_end[i] = partition_start[i] (i.e., 0 groups)
 *
 * EXAMPLE (32 warps, weights {3, 0, 1} -> 24 + 0 + 8 split):
 * ===========================================================
 *   uint32_t weights[] = {3, 0, 1};
 *   auto [partition_id, subgroup] = warp.partition(weights);
 *   if (partition_id == 0) {
 *     p2p.send(subgroup, sendBuff, nBytes);  // 24 warps
 *   } else if (partition_id == 1) {
 *     // No warps assigned (zero weight)
 *   } else {
 *     p2p.recv(subgroup, recvBuff, nBytes);  // 8 warps
 *   }
 *
 * @param weights Span of relative weights (non-zero count must be <=
 * total_groups)
 * @return {partition_id, subgroup} for this group
 */
__device__ inline PartitionResult ThreadGroup::partition(
    DeviceSpan<const uint32_t> weights) const {
#ifdef __CUDA_ARCH__
  const uint32_t num_partitions = static_cast<uint32_t>(weights.size());

  // Count non-zero weights and calculate total weight
  uint32_t total_weight = 0;
  uint32_t non_zero_count = 0;
  for (uint32_t i = 0; i < num_partitions; i++) {
    total_weight += weights[i];
    if (weights[i] > 0) {
      non_zero_count++;
    }
  }

  // Only partitions with non-zero weight need groups.
  // Use __trap() instead of assert() to ensure this check is active in both
  // debug and release builds.
  if (non_zero_count > total_groups) {
    printf(
        "partition(weights): non_zero_weight_count (%u) must be <= total_groups (%u)\n",
        non_zero_count,
        total_groups);
    __trap();
  }

  // Handle edge case: all weights are zero
  if (total_weight == 0) {
    // Assign all groups to partition 0 (arbitrary but deterministic)
    return PartitionResult{
        .partition_id = 0,
        .subgroup = ThreadGroup{
            .thread_id_in_group = thread_id_in_group,
            .group_size = group_size,
            .group_id = group_id,
            .total_groups = total_groups,
            .scope = scope}};
  }

  // Calculate distributable groups (after guaranteeing 1 per non-zero
  // partition)
  const uint32_t distributable_groups = total_groups - non_zero_count;

  uint32_t partition_start = 0;
  uint32_t accumulated_weight = 0;
  uint32_t non_zero_seen = 0;

  for (uint32_t i = 0; i < num_partitions; i++) {
    accumulated_weight += weights[i];

    uint32_t partition_end;
    if (weights[i] == 0) {
      // Zero-weight partitions get no groups
      partition_end = partition_start;
    } else {
      non_zero_seen++;
      // Each non-zero partition gets: 1 (guaranteed) + proportional share
      // Use ceiling division for the proportional part
      uint32_t proportional_groups =
          (accumulated_weight * distributable_groups + total_weight - 1) /
          total_weight;
      partition_end = non_zero_seen + proportional_groups;

      // Clamp to total_groups (last partition gets remainder)
      if (partition_end > total_groups) {
        partition_end = total_groups;
      }
    }

    if (group_id < partition_end) {
      return PartitionResult{
          .partition_id = i,
          .subgroup = ThreadGroup{
              .thread_id_in_group = thread_id_in_group,
              .group_size = group_size,
              .group_id = group_id - partition_start,
              .total_groups = partition_end - partition_start,
              .scope = scope}};
    }
    partition_start = partition_end;
  }
#endif
  return PartitionResult{};
}

/**
 * partition_interleaved - Interleaved partitioning (odd/even for 2 partitions)
 *
 * Unlike partition() which creates contiguous partitions (0-15, 16-31),
 * partition_interleaved distributes groups in a round-robin fashion:
 * - Partition 0: groups 0, 2, 4, 6, ... (even groups)
 * - Partition 1: groups 1, 3, 5, 7, ... (odd groups)
 *
 * This interleaves send/recv blocks across SMs for better load distribution
 * and can improve performance with clustered launches.
 *
 * EXAMPLE (32 blocks, 2 partitions):
 * =================================
 *   auto [partition_id, subgroup] = group.partition_interleaved(2);
 *   if (partition_id == 0) {
 *     p2p.recv(subgroup, recvBuff, nBytes);  // blocks 0,2,4,...,30
 *   } else {
 *     p2p.send(subgroup, sendBuff, nBytes);  // blocks 1,3,5,...,31
 *   }
 *
 * @param num_partitions Number of partitions (typically 2 for send/recv)
 * @return {partition_id, subgroup} where subgroup has renumbered group_id
 */
__device__ inline PartitionResult ThreadGroup::partition_interleaved(
    uint32_t num_partitions) const {
#ifdef __CUDA_ARCH__
  if (num_partitions > total_groups) {
    printf(
        "partition_interleaved: num_partitions (%u) must be <= total_groups (%u)\n",
        num_partitions,
        total_groups);
    __trap();
  }

  // Interleaved assignment: group_id % num_partitions
  uint32_t pid = group_id % num_partitions;

  // Count how many groups are in this partition
  // Groups assigned: pid, pid+num_partitions, pid+2*num_partitions, ...
  uint32_t groups_in_partition =
      (total_groups + num_partitions - 1 - pid) / num_partitions;

  // Renumber group_id within partition: 0, 1, 2, ...
  uint32_t new_group_id = group_id / num_partitions;

  return PartitionResult{
      .partition_id = pid,
      .subgroup = ThreadGroup{
          .thread_id_in_group = thread_id_in_group,
          .group_size = group_size,
          .group_id = new_group_id,
          .total_groups = groups_in_partition,
          .scope = scope}};
#endif
  return PartitionResult{};
}

__device__ inline ThreadGroup make_warp_group() {
#ifdef __CUDA_ARCH__
  uint32_t warps_per_block = blockDim.x / comms::device::kWarpSize;
  uint32_t warp_id_in_block = threadIdx.x / comms::device::kWarpSize;
  uint32_t global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
  uint32_t total_warps = gridDim.x * warps_per_block;

  uint32_t lane_id = threadIdx.x % comms::device::kWarpSize;

  return ThreadGroup{
      .thread_id_in_group = lane_id,
      .group_size = comms::device::kWarpSize,
      .group_id = global_warp_id,
      .total_groups = total_warps,
      .scope = SyncScope::WARP};
#else
  return ThreadGroup{};
#endif
}

/**
 * make_block_group - Create a ThreadGroup where all threads in a block
 *                    work together as a single group
 *
 * Use case: When work items need more parallelism than a warp provides,
 * or when __syncthreads() synchronization is acceptable.
 *
 * Example with 4 blocks × 256 threads:
 *   - total_groups = 4 (one per block)
 *   - group_size = 256
 *   - Each block processes work items cooperatively
 */
__device__ inline ThreadGroup make_block_group() {
#ifdef __CUDA_ARCH__
  return ThreadGroup{
      .thread_id_in_group = threadIdx.x,
      .group_size = blockDim.x,
      .group_id = blockIdx.x,
      .total_groups = gridDim.x,
      .scope = SyncScope::TILE};
#else
  return ThreadGroup{};
#endif
}

/**
 * make_warpgroup_group - Create a ThreadGroup where 4 warps (128 threads)
 *                        work together as a single warpgroup
 *
 * Use case: For Hopper GPU tensor core operations (wgmma instructions) that
 * operate at warpgroup granularity, or when you need synchronization
 * granularity between a single warp and the entire block.
 *
 * REQUIREMENTS:
 * - Block size must be a multiple of 128 (warpgroup size)
 * - Maximum 16 warpgroups per block (hardware named barrier limit)
 *
 * Example with 4 blocks × 512 threads:
 *   - total_groups = 16 (4 warpgroups per block × 4 blocks)
 *   - group_size = 128
 *   - Each warpgroup can execute wgmma instructions or other
 *     warpgroup-level operations
 *
 * HOPPER GPU BENEFITS:
 * - Enables efficient tensor core utilization through wgmma instructions
 * - Allows asynchronous warpgroup-level matrix multiply-accumulate
 * - Better synchronization granularity for producer-consumer patterns
 */
// TODO: Add support for configurable warpgroup size, 4/8/16.. warps as a
// warpgroup.
__device__ inline ThreadGroup make_warpgroup_group() {
#ifdef __CUDA_ARCH__
  constexpr uint32_t kWarpgroupSize =
      4 * comms::device::kWarpSize; // 128 threads
  uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;

  uint32_t warpgroups_per_block = threads_per_block / kWarpgroupSize;
  uint32_t warpgroup_id_in_block = tid / kWarpgroupSize;
  uint32_t global_warpgroup_id =
      blockIdx.x * warpgroups_per_block + warpgroup_id_in_block;
  uint32_t total_warpgroups = gridDim.x * warpgroups_per_block;

  uint32_t thread_id_in_warpgroup = tid % kWarpgroupSize;

  return ThreadGroup{
      .thread_id_in_group = thread_id_in_warpgroup,
      .group_size = kWarpgroupSize,
      .group_id = global_warpgroup_id,
      .total_groups = total_warpgroups,
      .scope = SyncScope::WARPGROUP};
#else
  return ThreadGroup{};
#endif
}

/**
 * make_thread_group - Create a ThreadGroup based on the specified SyncScope
 *
 * Convenience function that dispatches to the appropriate factory function
 * based on the scope parameter:
 *   - SyncScope::WARP → make_warp_group()
 *   - SyncScope::WARPGROUP → make_warpgroup_group()
 *   - SyncScope::TILE → make_block_group()
 *
 * @param scope The synchronization scope determining the group type
 * @return ThreadGroup configured for the specified scope
 */
__device__ inline ThreadGroup make_thread_group(SyncScope scope) {
#ifdef __CUDA_ARCH__
  switch (scope) {
    case SyncScope::WARP:
      return make_warp_group();
    case SyncScope::WARPGROUP:
      return make_warpgroup_group();
    case SyncScope::TILE:
      return make_block_group();
  }
#endif
  return ThreadGroup{};
}
} // namespace comms::pipes
