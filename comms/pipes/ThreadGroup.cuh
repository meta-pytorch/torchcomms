// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/pipes/DeviceSpan.cuh"

namespace comms::pipes {

constexpr uint32_t WARP_SIZE = 32;

enum class SyncScope { WARP, TILE };

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

  const uint32_t groups_per_partition =
      (total_groups + num_partitions - 1) / num_partitions;
  const uint32_t pid = group_id / groups_per_partition;
  const uint32_t partition_start = pid * groups_per_partition;
  const uint32_t partition_end =
      (partition_start + groups_per_partition < total_groups)
      ? partition_start + groups_per_partition
      : total_groups;

  return PartitionResult{
      .partition_id = pid,
      .subgroup = ThreadGroup{
          .thread_id_in_group = thread_id_in_group,
          .group_size = group_size,
          .group_id = group_id - partition_start,
          .total_groups = partition_end - partition_start,
          .scope = scope}};
#else
  return PartitionResult{};
#endif
}

/**
 * partition - Divide groups according to weights
 *
 * Divides groups into N partitions proportionally based on weights.
 * All groups are assigned to exactly one partition.
 *
 * REQUIREMENTS:
 * =============
 * - num_partitions (weights.size()) must be <= total_groups
 * - If num_partitions > total_groups, the kernel will trap with an error
 *   message showing both values
 *
 * GUARANTEES:
 * ===========
 * - Each partition receives at least 1 group (regardless of weight skew)
 * - Groups are distributed proportionally to weights (after minimum guarantee)
 * - All groups are assigned to exactly one partition
 *
 * ALGORITHM: Reserve-Then-Distribute
 * ===================================
 * Each partition gets: 1 (guaranteed) + proportional share of remaining groups
 *
 *   partition_end[i] = (i + 1) + ceil(accumulated_weight[i] * distributable /
 * total_weight)
 *
 * where distributable = total_groups - num_partitions
 *
 * MATHEMATICAL PROOF OF MINIMUM GUARANTEE:
 * ========================================
 * For partition i, the size is:
 *   size[i] = partition_end[i] - partition_end[i-1]
 *           = [(i + 1) + proportional[i]] - [i + proportional[i-1]]
 *           = 1 + (proportional[i] - proportional[i-1])
 *           >= 1  (because proportional is monotonically non-decreasing)
 *
 * Since accumulated_weight always increases, proportional[i] >=
 * proportional[i-1], so the difference is always >= 0. Therefore, size[i] >= 1
 * for all partitions.
 *
 * WHY THIS CONSTRAINT:
 * ====================
 * When num_partitions > total_groups, some partitions would receive zero
 * groups, and group assignment would skip partitions non-deterministically
 * based on rounding. This is almost always a bug in the caller's logic.
 *
 * EXAMPLE (32 warps, weights {3, 1} -> 24 + 8 split):
 * ====================================================
 *   uint32_t weights[] = {3, 1};
 *   auto [partition_id, subgroup] = warp.partition(weights);
 *   if (partition_id == 0) {
 *     p2p.send(subgroup, sendBuff, nBytes);  // 24 warps
 *   } else {
 *     p2p.recv(subgroup, recvBuff, nBytes);  // 8 warps
 *   }
 *
 * @param weights Span of relative weights (size must be <= total_groups)
 * @return {partition_id, subgroup} for this group
 */
__device__ inline PartitionResult ThreadGroup::partition(
    DeviceSpan<const uint32_t> weights) const {
#ifdef __CUDA_ARCH__
  const uint32_t num_partitions = static_cast<uint32_t>(weights.size());

  // More partitions than groups is invalid - some partitions would be empty
  // and group assignment would skip partitions non-deterministically.
  // Use __trap() instead of assert() to ensure this check is active in both
  // debug and release builds.
  if (num_partitions > total_groups) {
    printf(
        "partition(weights): num_partitions (%u) must be <= total_groups (%u)\n",
        num_partitions,
        total_groups);
    __trap();
  }

  uint32_t total_weight = 0;
  for (uint32_t i = 0; i < num_partitions; i++) {
    total_weight += weights[i];
  }

  // Calculate distributable groups (after guaranteeing 1 per partition)
  const uint32_t distributable_groups = total_groups - num_partitions;

  uint32_t partition_start = 0;
  uint32_t accumulated_weight = 0;

  for (uint32_t i = 0; i < num_partitions; i++) {
    accumulated_weight += weights[i];

    // Each partition gets: 1 (guaranteed) + proportional share of remaining
    // Use ceiling division for the proportional part to avoid rounding down
    uint32_t proportional_groups =
        (accumulated_weight * distributable_groups + total_weight - 1) /
        total_weight;
    uint32_t partition_end = (i + 1) + proportional_groups;

    // Clamp to total_groups (last partition gets remainder)
    if (partition_end > total_groups) {
      partition_end = total_groups;
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

__device__ inline ThreadGroup make_warp_group() {
#ifdef __CUDA_ARCH__
  uint32_t warps_per_block = blockDim.x / WARP_SIZE;
  uint32_t warp_id_in_block = threadIdx.x / WARP_SIZE;
  uint32_t global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
  uint32_t total_warps = gridDim.x * warps_per_block;

  uint32_t lane_id = threadIdx.x % WARP_SIZE;

  return ThreadGroup{
      .thread_id_in_group = lane_id,
      .group_size = WARP_SIZE,
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

} // namespace comms::pipes
