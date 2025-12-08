// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

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
 *   - global_thread_id = 290 (blockIdx=1, threadIdx=34)
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

  // GLOBAL CONTEXT (for reference):
  // ================================

  // global_thread_id - Unique thread ID across entire kernel
  // Calculated as: blockIdx.x × blockDim.x + threadIdx.x
  uint32_t global_thread_id;

  // total_threads - Total number of threads in entire kernel
  // Calculated as: gridDim.x × blockDim.x
  uint32_t total_threads;

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
};

__device__ inline ThreadGroup make_warp_group() {
#ifdef __CUDA_ARCH__
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total_threads = blockDim.x * gridDim.x;

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
      .scope = SyncScope::WARP,
      .global_thread_id = global_tid,
      .total_threads = total_threads};
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
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total_threads = blockDim.x * gridDim.x;

  return ThreadGroup{
      .thread_id_in_group = threadIdx.x,
      .group_size = blockDim.x,
      .group_id = blockIdx.x,
      .total_groups = gridDim.x,
      .scope = SyncScope::TILE,
      .global_thread_id = global_tid,
      .total_threads = total_threads};
#else
  return ThreadGroup{};
#endif
}

} // namespace comms::pipes
