// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "comms/common/AtomicUtils.cuh"
#include "comms/common/DeviceConstants.cuh"
#include "comms/pipes/DeviceSpan.cuh"

namespace comms::pipes {

using comms::device::kWarpSize;

constexpr uint32_t kMultiwarpSize = 4 * kWarpSize;

// Hardware supports max 16 named barriers per block, limiting the number
// of multiwarps per block to 16 (i.e., max block size = 16 * 128 = 2048).
constexpr uint32_t kMaxMultiwarpsPerBlock = 2048 / kMultiwarpSize;

/**
 * SyncScope - Defines the synchronization and grouping scope for ThreadGroup
 *
 * This enum is used both for:
 * 1. Internal synchronization (sync() method behavior)
 * 2. Selecting which factory function to use when creating ThreadGroups
 *
 * Available scopes:
 * - THREAD:    Single thread per group (no-op sync, finest granularity)
 * - WARP:      32 threads per group (uses __syncwarp)
 * - MULTIWARP: 128 threads per group (4 warps, uses named barriers)
 * - BLOCK:     All threads in a block form one group (uses __syncthreads)
 * - CLUSTER:   All threads in a cluster form one group (uses cluster
 *              barriers)
 *
 * Usage example:
 *   __global__ void myKernel(SyncScope scope) {
 *     auto group = make_thread_group(scope);
 *     // ...
 *   }
 */
enum class SyncScope { THREAD, WARP, MULTIWARP, BLOCK, CLUSTER };

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
  // WARP: uses __syncwarp() (fast). BLOCK: uses __syncthreads() (block-wide).
  // CLUSTER: uses cluster.sync().
  SyncScope scope;

  __device__ inline void sync() {
#ifdef __CUDA_ARCH__
    switch (scope) {
      case SyncScope::THREAD:
        // Single-thread group: emit a compiler barrier to prevent reordering
        // across this sync point. No hardware instruction needed since there
        // is only one thread, but the compiler must not hoist or sink memory
        // operations across sync() — matching the invariant that sync()
        // establishes a happens-before boundary within a thread's instruction
        // stream.
        asm volatile("" ::: "memory");
        break;
      case SyncScope::WARP:
        __syncwarp();
        break;
      case SyncScope::MULTIWARP: {
        // Multiwarp = 4 warps = 128 threads
        // Uses named barriers for synchronization within a multiwarp
        uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
        uint32_t barrierId = tid / kMultiwarpSize;
        asm volatile("bar.sync %0, %1;"
                     :
                     : "r"(barrierId), "r"(kMultiwarpSize));
        break;
      }
      case SyncScope::BLOCK:
        __syncthreads();
        break;
      case SyncScope::CLUSTER:
#if __CUDA_ARCH__ >= 900 && !defined(__clang_llvm_bitcode_lib__)
      {
        cooperative_groups::cluster_group cluster =
            cooperative_groups::this_cluster();
        cluster.sync();
      }
#else
        // Fallback to block sync for older architectures or clang bitcode path
        // (cooperative_groups::cluster_group is not available in clang 19's
        // CUDA headers; Triton kernels do not use cluster scope anyway)
        __syncthreads();
#endif
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
   * broadcast - Broadcast a value from the group leader to all
   *             threads in the group
   *
   * Supports uint32_t and uint64_t types.
   *
   * Uses the most efficient mechanism for each scope:
   * - WARP: warp shuffle (register-level, no shared memory)
   * - MULTIWARP: shared memory indexed by multiwarp ID
   * - BLOCK: single shared memory location
   * - CLUSTER: not supported (traps)
   *
   * Double sync pattern prevents race when broadcast is called multiple
   * times in succession: the second sync ensures all threads have read the
   * value before the leader can overwrite it in a subsequent call.
   *
   * NOTE: The __shared__ variables use fixed names (__tg_broadcast_scratch,
   * __tg_broadcast_block) because CUDA deduplicates __shared__ variables in
   * inline functions by name, not by call site. Multiple calls to this
   * function from the same kernel correctly share the same __shared__ storage.
   *
   * @param val The value to broadcast (only leader's value is used)
   * @return The leader's value, received by all threads
   */
  template <typename T>
  __device__ inline T broadcast(T val) {
#ifdef __CUDA_ARCH__
    switch (scope) {
      case SyncScope::WARP:
        return shfl(val, 0);
      case SyncScope::MULTIWARP: {
        // Always use uint64_t shared memory so that broadcast<uint32_t> and
        // broadcast<uint64_t> share the same __shared__ allocation (CUDA
        // deduplicates by name).
        __shared__ uint64_t __tg_broadcast_scratch[kMaxMultiwarpsPerBlock];
        uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
        uint32_t scratch_idx = tid / kMultiwarpSize;
        if (is_leader()) {
          __tg_broadcast_scratch[scratch_idx] = static_cast<uint64_t>(val);
        }
        sync();
        T result = static_cast<T>(__tg_broadcast_scratch[scratch_idx]);
        sync(); // Prevent leader overwriting before all threads read
        return result;
      }
      case SyncScope::BLOCK: {
        __shared__ uint64_t __tg_broadcast_block;
        if (is_leader()) {
          __tg_broadcast_block = static_cast<uint64_t>(val);
        }
        sync();
        T result = static_cast<T>(__tg_broadcast_block);
        sync(); // Prevent leader overwriting before all threads read
        return result;
      }
      case SyncScope::CLUSTER:
        printf("ThreadGroup::broadcast: CLUSTER scope not yet supported\n");
        __trap();
    }
#endif
    return val;
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

 private:
#ifdef __CUDACC__
  __device__ static __forceinline__ uint32_t
  shfl(uint32_t val, unsigned srcLane) {
#ifdef __HIP_PLATFORM_AMD__
    return __shfl(static_cast<int>(val), srcLane, kWarpSize);
#else
    return __shfl_sync(0xFFFFFFFFU, val, srcLane);
#endif
  }

  __device__ static __forceinline__ uint64_t
  shfl(uint64_t val, unsigned srcLane) {
#ifdef __HIP_PLATFORM_AMD__
    return static_cast<uint64_t>(
        __shfl(static_cast<long long>(val), srcLane, kWarpSize));
#else
    constexpr unsigned kFullWarpMask = 0xFFFFFFFFU;
    uint32_t low = static_cast<uint32_t>(val);
    uint32_t high = static_cast<uint32_t>(val >> 32);
    low = __shfl_sync(kFullWarpMask, low, srcLane);
    high = __shfl_sync(kFullWarpMask, high, srcLane);
    return (static_cast<uint64_t>(high) << 32) | low;
#endif
  }
#endif // __CUDACC__
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

/**
 * make_thread_solo - Create a single-thread ThreadGroup for this thread
 *
 * Each thread forms its own group of size 1. sync() is a no-op.
 * Use when an operation must execute on a single thread at a time,
 * or when composing with scope-dispatch code that needs a THREAD scope group.
 *
 * Unlike warp/block groups, thread_id_in_group is always 0 (this thread is
 * always the leader). group_id and total_groups are based on the global
 * thread index and count respectively.
 */
__device__ inline ThreadGroup make_thread_solo() {
#ifdef __CUDA_ARCH__
  uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
  uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  uint32_t global_tid = blockIdx.x * threads_per_block + tid;
  uint32_t total_threads = gridDim.x * threads_per_block;

  return ThreadGroup{
      .thread_id_in_group = 0,
      .group_size = 1,
      .group_id = global_tid,
      .total_groups = total_threads,
      .scope = SyncScope::THREAD};
#else
  return ThreadGroup{};
#endif
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
 * make_cluster_group - Create a ThreadGroup where all threads in a
 *                      cluster work together as a single group
 *
 * Use case: For Hopper GPU cluster-based operations where multiple blocks
 * in a cluster need to synchronize and cooperate on work items.
 *
 * REQUIREMENTS:
 * - Requires SM90 (Hopper) or later architecture
 * - Kernel must be launched with cluster support (cudaLaunchConfig)
 * - Cluster size is determined at kernel launch time
 *
 * Example with 4 clusters × 2 blocks/cluster × 256 threads:
 *   - total_groups = 4 (one per cluster)
 *   - group_size = 512 (2 blocks × 256 threads per cluster)
 *   - Each cluster processes work items cooperatively
 *
 * HOPPER GPU BENEFITS:
 * - Enables efficient distributed shared memory access across cluster
 * - Allows barrier synchronization across multiple blocks
 * - Better locality for inter-block communication patterns
 *
 * HARDWARE SPECS (H100):
 * - ~16 SMs per GPC, 8 GPCs total -> 132 SMs
 * - Maximum cluster size: 16 blocks (limited by GPC)
 *
 * NOTE: On architectures before SM90, falls back to single-block behavior
 * where cluster_size is effectively 1.
 */
__device__ inline ThreadGroup make_cluster_group() {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 900
  // Get cluster grid dimensions using PTX instructions
  uint32_t num_clusters_x, cluster_rank;
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(num_clusters_x));
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(cluster_rank));

  uint32_t cluster_size;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_size));

  uint32_t block_rank_in_cluster;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(block_rank_in_cluster));

  uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  uint32_t threads_per_cluster = cluster_size * threads_per_block;

  uint32_t tid_in_block = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
  uint32_t thread_id_in_cluster =
      block_rank_in_cluster * threads_per_block + tid_in_block;

  return ThreadGroup{
      .thread_id_in_group = thread_id_in_cluster,
      .group_size = threads_per_cluster,
      .group_id = cluster_rank,
      .total_groups = num_clusters_x,
      .scope = SyncScope::CLUSTER};
#else
  // Fallback for non-Hopper: treat each block as its own cluster
  return ThreadGroup{
      .thread_id_in_group = threadIdx.x,
      .group_size = blockDim.x,
      .group_id = blockIdx.x,
      .total_groups = gridDim.x,
      .scope = SyncScope::CLUSTER};
#endif
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
      .scope = SyncScope::BLOCK};
#else
  return ThreadGroup{};
#endif
}

/**
 * make_multiwarp_group - Create a ThreadGroup where 4 warps (128 threads)
 *                        work together as a single multiwarp
 *
 * Use case: For Hopper GPU tensor core operations (wgmma instructions) that
 * operate at multiwarp granularity, or when you need synchronization
 * granularity between a single warp and the entire block.
 *
 * REQUIREMENTS:
 * - Block size must be a multiple of 128 (multiwarp size)
 * - Maximum 16 multiwarps per block (hardware named barrier limit)
 *
 * Example with 4 blocks × 512 threads:
 *   - total_groups = 16 (4 multiwarps per block × 4 blocks)
 *   - group_size = 128
 *   - Each multiwarp can execute wgmma instructions or other
 *     multiwarp-level operations
 *
 * HOPPER GPU BENEFITS:
 * - Enables efficient tensor core utilization through wgmma instructions
 * - Allows asynchronous multiwarp-level matrix multiply-accumulate
 * - Better synchronization granularity for producer-consumer patterns
 */
// TODO: Add support for configurable multiwarp size, 4/8/16.. warps as a
// multiwarp.
__device__ inline ThreadGroup make_multiwarp_group() {
#ifdef __CUDA_ARCH__
  uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;

  uint32_t multiwarps_per_block = threads_per_block / kMultiwarpSize;
  uint32_t multiwarp_id_in_block = tid / kMultiwarpSize;
  uint32_t global_multiwarp_id =
      blockIdx.x * multiwarps_per_block + multiwarp_id_in_block;
  uint32_t total_multiwarps = gridDim.x * multiwarps_per_block;

  uint32_t thread_id_in_multiwarp = tid % kMultiwarpSize;

  return ThreadGroup{
      .thread_id_in_group = thread_id_in_multiwarp,
      .group_size = kMultiwarpSize,
      .group_id = global_multiwarp_id,
      .total_groups = total_multiwarps,
      .scope = SyncScope::MULTIWARP};
#else
  return ThreadGroup{};
#endif
}

/**
 * make_thread_group - Create a ThreadGroup based on the specified scope
 *
 * Convenience function that dispatches to the appropriate factory function
 * based on the scope parameter:
 *   - SyncScope::THREAD    → make_thread_solo() (no-op sync, size 1)
 *   - SyncScope::WARP      → make_warp_group()
 *   - SyncScope::MULTIWARP → make_multiwarp_group()
 *   - SyncScope::BLOCK     → make_block_group()
 *   - SyncScope::CLUSTER   → make_cluster_group()
 *
 * @param scope The desired thread grouping strategy
 * @return ThreadGroup configured for the specified scope
 *
 * Example:
 *   __global__ void myKernel(SyncScope scope) {
 *     auto group = make_thread_group(scope);
 *     group.for_each_item_contiguous(numItems, [&](uint32_t item_id) {
 *       // Process item
 *     });
 *   }
 */
__device__ inline ThreadGroup make_thread_group(SyncScope scope) {
#ifdef __CUDA_ARCH__
  switch (scope) {
    case SyncScope::THREAD:
      return make_thread_solo();
    case SyncScope::WARP:
      return make_warp_group();
    case SyncScope::MULTIWARP:
      return make_multiwarp_group();
    case SyncScope::BLOCK:
      return make_block_group();
    case SyncScope::CLUSTER:
      return make_cluster_group();
    default:
      // Should never reach here, but return warp group as default
      return make_warp_group();
  }
#else
  return ThreadGroup{};
#endif
}

} // namespace comms::pipes
