// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/moe_ep/cpp/intranode/kernels/Notify.cuh"

#include "comms/prims/collectives/moe_ep/cpp/shared/Config.h"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Exception.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelConfigs.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelUtils.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Launch.cuh"

// `notify_dispatch` (lines 11-150).
//
// Performs three tasks:
//
//  1. Per-rank exchange — every rank writes its `num_tokens_per_rank[i]` count
//     for every other rank `i` into a per-peer slot in the shared NVLink
//     buffer. After a barrier, every rank has the full N×N count matrix.
//  2. Per-channel prefix matrix — for each (dst_rank, channel) pair, count
//     how many tokens this rank routes to that destination on that channel
//     and write a prefix sum.
//  3. Memset — zero out the channel-state region of the buffer so the
//     subsequent dispatch kernel sees clean state.
//
// Synchronization: a star-pattern atomicAdd_system / atomicSub_system on
// per-peer FIFO slots. `task_fifo_ptrs` here is
// `barrier_signal_ptrs[rank]` (peer-mapped int* pointers from the runtime's
// barrier slot region). The `head` rotates through `NUM_MAX_FIFO_SLOTS` so
// concurrent kernel invocations don't trample each other.

namespace comms::prims::moe_ep::kernels {

namespace {

// Helper — warp-vote pattern. ALL threads in
// the warp must execute, so the implicit warp sync inside the vote keeps
// lanes converged. Lanes >= kNumRanks always vote "false" (finished).
template <int kNumRanks>
__device__ __forceinline__ bool not_finished_inline(int* task, int expected) {
  const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;
  bool result = false;
  if (lane_id < kNumRanks) {
    result = ld_volatile_global(task + lane_id) != expected;
  }
#ifdef __HIP_PLATFORM_AMD__
  return __any(result);
#else
  return __any_sync(kFullWarpMask, result);
#endif
}

// Star-pattern barrier. Each
// rank atomically increments its own slot in every peer's FIFO and
// atomically decrements every peer's slot in its own FIFO. After both
// sides have done so, the local FIFO entries for every peer should sum
// back to zero, at which point all ranks have reached the barrier.
//
// Spin loop uses `__any_sync` warp-vote so all kNumRanks lanes converge
// when ALL slots are 0. Without that, lanes exit independently and the
// warp can desync mid-spin — possible source of cross-rank atomic ordering
// hazards under AMD's wave64.
template <int kNumRanks>
__device__ __forceinline__ void
barrier_device_inline(int** task_fifo_ptrs, int head, int rank) {
  const int thread_id = static_cast<int>(threadIdx.x);
  EP_DEVICE_ASSERT(kNumRanks <= kWarpSize);

  if (thread_id < kNumRanks) {
    atomicAdd_system(task_fifo_ptrs[rank] + head + thread_id, FINISHED_SUM_TAG);
    memory_fence();
    atomicSub_system(task_fifo_ptrs[thread_id] + head + rank, FINISHED_SUM_TAG);
  }

  long long start = wall_clock64_compat();
  while (not_finished_inline<kNumRanks>(task_fifo_ptrs[rank] + head, 0)) {
    long long now = wall_clock64_compat();
    long long elapsed = now > start ? (now - start) : 0;
    if (elapsed > NUM_TIMEOUT_CYCLES && thread_id == 0) {
      printf("moe_ep notify_dispatch barrier timeout (rank=%d)\n", rank);
      trap_kernel();
    }
  }
}

template <int kNumRanks>
__device__ __forceinline__ void move_fifo_slots_inline(int& head) {
  head = (head + kNumRanks) % NUM_MAX_FIFO_SLOTS;
}

template <int kNumRanks>
__global__ void notify_dispatch_kernel(
    const int* num_tokens_per_rank,
    int* moe_recv_counter_mapped,
    const int* num_tokens_per_expert,
    int* moe_recv_expert_counter_mapped,
    int num_experts,
    int num_tokens,
    int num_channels,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix_copy,
    int num_memset_int,
    int expert_alignment,
    void** buffer_ptrs,
    int** task_fifo_ptrs,
    int head,
    int rank) {
  const int sm_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int num_threads = static_cast<int>(blockDim.x);
  const int lane_id = thread_id % kWarpSize;
  const int warp_id = thread_id / kWarpSize;
  const int num_warps = num_threads / kWarpSize;

  if (sm_id == 0) {
    // Block 0: cross-rank count exchange via peer-mapped buffer.
    barrier_device_inline<kNumRanks>(task_fifo_ptrs, head, rank);
    move_fifo_slots_inline<kNumRanks>(head);
    __syncthreads();

    int* per_rank_buffer = nullptr;
    int* per_expert_buffer = nullptr;
    if (thread_id < kNumRanks) {
      per_rank_buffer = reinterpret_cast<int*>(buffer_ptrs[thread_id]);
      per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
    }

    // Per-rank counts: write `num_tokens_per_rank[i]` into peer i's row,
    // column = our rank.
    const int num_experts_per_rank = num_experts / kNumRanks;
    if (thread_id < kNumRanks) {
      per_rank_buffer[rank * kNumRanks + thread_id] =
          num_tokens_per_rank[thread_id];
      // These peer per-expert writes are consumed only after the
      // __syncthreads() at the end of this block AND the cross-rank
      // barrier_device_inline() below; a sync inside this
      // `if (thread_id < kNumRanks)` block would be a divergent barrier.
      // patternlint-disable-next-line cuda-smem-reduction-missing-final-sync
#pragma unroll
      for (int i = 0; i < num_experts_per_rank; ++i) {
        per_expert_buffer[rank * num_experts_per_rank + i] =
            num_tokens_per_expert[thread_id * num_experts_per_rank + i];
      }
    }
    __syncthreads();

    // Wait for all ranks to finish writing.
    barrier_device_inline<kNumRanks>(task_fifo_ptrs, head, rank);
    move_fifo_slots_inline<kNumRanks>(head);
    __syncthreads();

    // Compute prefix sum over per-rank counts and write back the local
    // total to CPU-mapped memory.
    int* local_per_rank_buffer = reinterpret_cast<int*>(buffer_ptrs[rank]);
    if (thread_id < kNumRanks) {
#pragma unroll
      for (int i = 1; i < kNumRanks; ++i) {
        local_per_rank_buffer[i * kNumRanks + thread_id] +=
            local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
      }
      if (thread_id == rank) {
        *moe_recv_counter_mapped =
            local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
      }
    }

    // Per-expert sums + alignment.
    int* local_per_expert_buffer =
        local_per_rank_buffer + kNumRanks * kNumRanks;
    if (thread_id < num_experts_per_rank) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i) {
        sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
      }
      sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
      moe_recv_expert_counter_mapped[thread_id] = sum;
    }
    __syncthreads();

    // Copy rank prefix matrix to host-readable tensor (bypasses the per-peer
    // shared region).
#pragma unroll
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads) {
      rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];
    }

    // Zero out the channel-state region of the local buffer for the
    // subsequent dispatch kernel.
#pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads) {
      local_per_expert_buffer[i] = 0;
    }

    memory_fence();
    __syncthreads();
    barrier_device_inline<kNumRanks>(task_fifo_ptrs, head, rank);
  } else {
    // Blocks 1..N: per-channel × per-rank token counts.
    const int dst_rank = sm_id - 1;
    for (int channel_id = warp_id; channel_id < num_channels;
         channel_id += num_warps) {
      int token_start_idx = 0, token_end_idx = 0;
      get_channel_task_range(
          num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      int count = 0;
      for (int64_t i = token_start_idx + lane_id; i < token_end_idx;
           i += kWarpSize) {
        count += is_token_in_rank[i * kNumRanks + dst_rank];
      }
      count = warp_reduce_sum(count);
      if (lane_id == 0) {
        channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
      }
    }
    __syncthreads();

    // Block-local prefix sum across channels (single thread is fine; the
    // matrix has at most ~32 channels).
    if (thread_id == 0) {
#pragma unroll
      for (int i = 1; i < num_channels; ++i) {
        channel_prefix_matrix[dst_rank * num_channels + i] +=
            channel_prefix_matrix[dst_rank * num_channels + i - 1];
      }
    }
  }
}

} // namespace

// cached_notify_dispatch — handle-reuse path. Skips the count exchange and
// just (a) memcpys the cached rank_prefix_matrix back into the local
// buffer's prefix region (b) zeroes out `num_memset_int` ints of the
// channel-state region. Used when callers reuse a `handle` from a prior
// dispatch.
template <int kNumRanks>
__global__ void cached_notify_dispatch_kernel(
    const int* rank_prefix_matrix,
    int num_memset_int,
    void** buffer_ptrs,
    int** task_fifo_ptrs,
    int head,
    int rank) {
  barrier_device_inline<kNumRanks>(task_fifo_ptrs, head, rank);
  move_fifo_slots_inline<kNumRanks>(head);
  __syncthreads();

  const int thread_id = static_cast<int>(threadIdx.x);
  const int num_threads = static_cast<int>(blockDim.x);
  int* ptr = reinterpret_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
  for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads) {
    ptr[i] = rank_prefix_matrix[i];
  }
#pragma unroll
  for (int i = thread_id; i < num_memset_int; i += num_threads) {
    ptr[kNumRanks * kNumRanks + i] = 0;
  }
  memory_fence();
  __syncthreads();
  barrier_device_inline<kNumRanks>(task_fifo_ptrs, head, rank);
}

// cached_notify_combine — symmetric prep for combine. Barrier, zero out a
// fixed prefix of the local buffer, then run a per-channel pass that walks
// `send_head` backwards to fill in placeholder entries (`-last_head - 1`)
// for tokens that didn't route to a given destination.
template <int kNumRanks>
__global__ void cached_notify_combine_kernel(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_recv_tokens,
    int num_memset_int,
    int** task_fifo_ptrs,
    int head,
    int rank) {
  const int sm_id = static_cast<int>(blockIdx.x);
  if (sm_id == 0) {
    barrier_device_inline<kNumRanks>(task_fifo_ptrs, head, rank);
    move_fifo_slots_inline<kNumRanks>(head);
    __syncthreads();

    const int thread_id = static_cast<int>(threadIdx.x);
    const int num_threads = static_cast<int>(blockDim.x);
    int* ptr = reinterpret_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads) {
      ptr[i] = 0;
    }
    memory_fence();
    __syncthreads();
    barrier_device_inline<kNumRanks>(task_fifo_ptrs, head, rank);
  } else {
    const int channel_id = sm_id - 1;
    const int thread_id = static_cast<int>(threadIdx.x);
    const int rank_id = thread_id / kWarpSize;
    const int lane_id = get_lane_id();
    if (rank_id >= kNumRanks) {
      return;
    }

    int token_start_idx = 0, token_end_idx = 0;
    get_channel_task_range(
        num_recv_tokens,
        num_channels,
        channel_id,
        token_start_idx,
        token_end_idx);

    int last_head = 1 << 25; // sentinel: heuristic large value
    for (int token_idx_tail = token_end_idx - 1;
         token_idx_tail >= token_start_idx;
         token_idx_tail -= kWarpSize) {
      const int token_idx = token_idx_tail - lane_id;
      int expected_head = 0;
      int current_head = (token_idx >= token_start_idx)
          ? send_head[token_idx * kNumRanks + rank_id]
          : -1;
      const int loop_n = min(kWarpSize, token_idx_tail - token_start_idx + 1);
      for (int i = 0; i < loop_n; ++i) {
        const int h = shfl_sync_compat(current_head, i);
        if (h < 0) {
          if (lane_id == i) {
            expected_head = -last_head - 1;
          }
        } else {
          last_head = h;
        }
      }
      if (current_head < 0 && token_idx >= token_start_idx) {
        send_head[token_idx * kNumRanks + rank_id] = expected_head;
      }
    }
  }
}

void notify_dispatch(
    const int* num_tokens_per_rank,
    int* moe_recv_counter_mapped,
    int num_ranks,
    const int* num_tokens_per_expert,
    int* moe_recv_expert_counter_mapped,
    int num_experts,
    int num_tokens,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix_copy,
    int num_memset_int,
    int expert_alignment,
    void** buffer_ptrs,
    int** task_fifo_ptrs,
    int head,
    int rank,
    cudaStream_t stream,
    int num_channels) {
  constexpr int kNumThreads = 128;
  EP_HOST_ASSERT(num_experts % num_ranks == 0);
  EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads);
  EP_HOST_ASSERT(num_ranks <= kNumThreads);

  SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);

#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks) \
  LAUNCH_KERNEL_NON_COOPERATIVE(           \
      &cfg,                                \
      notify_dispatch_kernel<ranks>,       \
      num_tokens_per_rank,                 \
      moe_recv_counter_mapped,             \
      num_tokens_per_expert,               \
      moe_recv_expert_counter_mapped,      \
      num_experts,                         \
      num_tokens,                          \
      num_channels,                        \
      is_token_in_rank,                    \
      channel_prefix_matrix,               \
      rank_prefix_matrix_copy,             \
      num_memset_int,                      \
      expert_alignment,                    \
      buffer_ptrs,                         \
      task_fifo_ptrs,                      \
      head,                                \
      rank)

  SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

void cached_notify_dispatch(
    const int* rank_prefix_matrix,
    int num_memset_int,
    void** buffer_ptrs,
    int** task_fifo_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int head) {
  SETUP_LAUNCH_CONFIG(1, 128, stream);

#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks) \
  LAUNCH_KERNEL_NON_COOPERATIVE(                  \
      &cfg,                                       \
      cached_notify_dispatch_kernel<ranks>,       \
      rank_prefix_matrix,                         \
      num_memset_int,                             \
      buffer_ptrs,                                \
      task_fifo_ptrs,                             \
      head,                                       \
      rank)

  SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

void cached_notify_combine(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_recv_tokens,
    int num_memset_int,
    int** task_fifo_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int head) {
  // num_threads = max(128, 64 * num_ranks).
  const int num_threads = num_ranks * 64 > 128 ? num_ranks * 64 : 128;
  EP_HOST_ASSERT(num_threads <= 1024);
  SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);

#define CACHED_NOTIFY_COMBINE_LAUNCH_CASE(ranks) \
  LAUNCH_KERNEL_NON_COOPERATIVE(                 \
      &cfg,                                      \
      cached_notify_combine_kernel<ranks>,       \
      buffer_ptrs,                               \
      send_head,                                 \
      num_channels,                              \
      num_recv_tokens,                           \
      num_memset_int,                            \
      task_fifo_ptrs,                            \
      head,                                      \
      rank)

  SWITCH_RANKS(CACHED_NOTIFY_COMBINE_LAUNCH_CASE);
#undef CACHED_NOTIFY_COMBINE_LAUNCH_CASE
}

} // namespace comms::prims::moe_ep::kernels
