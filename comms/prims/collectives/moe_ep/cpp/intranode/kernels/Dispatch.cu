// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/moe_ep/cpp/intranode/kernels/Dispatch.cuh"

#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/EpBuffer.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Exception.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelConfigs.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelUtils.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Launch.cuh"

// `dispatch` (lines 204-528).
//
// Pipeline:
//   - Even-numbered SMs are senders, odd-numbered SMs are receivers.
//   - Sender / receiver pairs are partitioned by (channel, dst_rank): each
//     channel is `kNumRanks` warp-groups deep on the sender side; each
//     channel × dst_rank gets its own per-peer FIFO slot.
//   - Sender writes payloads to peer's recv-buffer region using
//     `MOE_EP_UNROLLED_WARP_COPY` (NVLink → IPC-mapped peer pointer).
//   - Receiver spins on `channel_tail_idx`, then `__syncthreads`-coordinates
//     with sender warps to drain into local `recv_x` etc.
//
// Layout of the per-rank workspace (offsets into `buffer_ptrs[rank]`):
//   `rank_prefix_matrix`  : kNumRanks * kNumRanks * sizeof(int)
//   `channel_start_offset`: num_channels * kNumRanks * sizeof(int)
//   `channel_end_offset`  : num_channels * kNumRanks * sizeof(int)
//   `channel_head_idx`    : num_channels * kNumRanks * sizeof(int)
//   `channel_tail_idx`    : num_channels * kNumRanks * sizeof(int)
//   `channel_x_buffers`   : num_channels * kNumRanks * num_recv_buffer_tokens
//                            * hidden_int4 * sizeof(int4)
//   `channel_src_idx_buffers`     : * sizeof(int)
//   `channel_topk_idx_buffers`    : * num_topk * sizeof(topk_idx_t)
//   `channel_topk_weights_buffers`: * num_topk * sizeof(float)
//   `channel_x_scales_buffers`    : * num_scales * sizeof(float)

namespace comms::prims::moe_ep::kernels {

namespace {

template <int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1) intranode_dispatch_kernel(
    int4* recv_x,
    float* recv_x_scales,
    int* recv_src_idx,
    topk_idx_t* recv_topk_idx,
    float* recv_topk_weights,
    int* recv_channel_offset,
    int* send_head,
    const int4* x,
    const float* x_scales,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    int num_tokens,
    int num_worst_tokens,
    int hidden_int4,
    int num_topk,
    int num_experts,
    int num_scales,
    int scale_token_stride,
    int scale_hidden_stride,
    void** buffer_ptrs,
    int rank,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
  const int num_sms = static_cast<int>(gridDim.x);
  const int sm_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const bool is_sender = sm_id % 2 == 0;
  EP_DEVICE_ASSERT(num_sms % 2 == 0);

  // Threads per (channel × dst_rank) pair.
  const int num_threads_per_rank = kNumThreads / kNumRanks;
  const int num_channels = num_sms / 2;
  const int responsible_rank = thread_id / num_threads_per_rank;
  const int responsible_channel = sm_id / 2;

  const int num_experts_per_rank = num_experts / kNumRanks;
  EP_DEVICE_ASSERT(num_experts_per_rank > 0 || num_topk == 0);
  EP_DEVICE_ASSERT(num_topk <= kWarpSize);
  EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
  EP_DEVICE_ASSERT(
      (recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

  // Buffer cursor advances over the workspace as the layout helpers slice
  // sub-regions out of it.
  void* ptr = reinterpret_cast<void*>(
      reinterpret_cast<int8_t*>(
          buffer_ptrs[is_sender ? responsible_rank : rank]) +
      kNumRanks * kNumRanks * sizeof(int));
  const int target_rank = is_sender ? rank : responsible_rank;
  const int num_channels_total = num_channels * kNumRanks;
  const int channel_rank_offset = responsible_channel * kNumRanks + target_rank;

  auto channel_start_offset =
      Buffer<int>(ptr, num_channels_total, channel_rank_offset);
  auto channel_end_offset =
      Buffer<int>(ptr, num_channels_total, channel_rank_offset);
  auto channel_head_idx =
      Buffer<int>(ptr, num_channels_total, channel_rank_offset);
  auto channel_tail_idx =
      Buffer<int>(ptr, num_channels_total, channel_rank_offset);

  auto channel_x_buffers = Buffer<int4>(
      ptr,
      num_channels_total * num_recv_buffer_tokens * hidden_int4,
      channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
  auto channel_src_idx_buffers = Buffer<int>(
      ptr,
      num_channels_total * num_recv_buffer_tokens,
      channel_rank_offset * num_recv_buffer_tokens);
  auto channel_topk_idx_buffers = Buffer<topk_idx_t>(
      ptr,
      num_channels_total * num_recv_buffer_tokens * num_topk,
      channel_rank_offset * num_recv_buffer_tokens * num_topk);
  auto channel_topk_weights_buffers = Buffer<float>(
      ptr,
      num_channels_total * num_recv_buffer_tokens * num_topk,
      channel_rank_offset * num_recv_buffer_tokens * num_topk);
  auto channel_x_scales_buffers = Buffer<float>(
      ptr,
      num_channels_total * num_recv_buffer_tokens * num_scales,
      channel_rank_offset * num_recv_buffer_tokens * num_scales);

  if (is_sender) {
    // ------------------- SENDER PATH -------------------
    constexpr int num_send_warps = kNumThreads / kWarpSize;
    constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
    const int send_thread_id = thread_id;
    const int send_lane_id = send_thread_id % kWarpSize;
    const int send_warp_id_in_rank =
        (send_thread_id % num_threads_per_rank) / kWarpSize;
    EP_DEVICE_ASSERT(kNumRanks <= kWarpSize);
    EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

    // Encode the start / end offsets as `-value - 1` so we can distinguish
    // "not yet written" (value == 0) from "wrote zero tokens".
    if (send_lane_id == 0 && send_warp_id_in_rank == 0) {
      int value = responsible_channel > 0
          ? channel_prefix_matrix
                [responsible_rank * num_channels + responsible_channel - 1]
          : 0;
      st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
      value = channel_prefix_matrix
          [responsible_rank * num_channels + responsible_channel];
      st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);
    }
    syncwarp();

    int token_start_idx = 0, token_end_idx = 0;
    get_channel_task_range(
        num_tokens,
        num_channels,
        responsible_channel,
        token_start_idx,
        token_end_idx);

    int cached_channel_tail_idx = 0;
    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
      // Wait for receiver to free up at least `num_max_send_tokens` slots.
      long long start_time = wall_clock64_compat();
      while (send_lane_id == 0) {
        int num_used_slots = cached_channel_tail_idx -
            ld_volatile_global(channel_head_idx.buffer());
        if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens) {
          break;
        }
        long long now = wall_clock64_compat();
        long long elapsed = now > start_time ? (now - start_time) : 0;
        if (elapsed > NUM_TIMEOUT_CYCLES) {
          printf(
              "moe_ep dispatch sender timeout, rank %d, channel %d\n",
              rank,
              responsible_channel);
          trap_kernel();
        }
      }
      syncwarp();

      int chunk_token_idx = 0;
      while (chunk_token_idx < num_max_send_tokens &&
             token_idx < token_end_idx) {
        // Save send_head so combine knows which slot this token ended up in
        // (or -1 if not routed to this dst_rank).
        if (send_lane_id == 0 &&
            token_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
          send_head[token_idx * kNumRanks + responsible_rank] =
              is_token_in_rank[token_idx * kNumRanks + responsible_rank]
              ? cached_channel_tail_idx
              : -1;
        }

        if (!is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
          token_idx++;
          continue;
        }

        const int dst_slot_idx =
            (cached_channel_tail_idx++) % num_recv_buffer_tokens;
        if (cached_channel_tail_idx % num_send_warps_per_rank ==
            send_warp_id_in_rank) {
          // Payload copy (hidden_int4 × 16B each) — NVLink to peer.
          // Sender reads its own LOCAL `x` buffer; use cached __ldg.
          // Cache-bypass loads (ld_nc_global) on local data collapse
          // throughput on AMD.
          int4* shifted_dst =
              channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
          const int4* shifted_src = x + token_idx * hidden_int4;
          MOE_EP_UNROLLED_WARP_COPY(
              kIntranodeUnrollFactor,
              send_lane_id,
              hidden_int4,
              shifted_dst,
              shifted_src,
              ld_cached_global,
              st_na_global);

          if (send_lane_id == 0) {
            channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);
          }

          // topk_idx + topk_weights — one lane per topk slot.
          if (send_lane_id < num_topk) {
            const int recv_expert_begin =
                responsible_rank * num_experts_per_rank;
            const int recv_expert_end =
                (responsible_rank + 1) * num_experts_per_rank;
            topk_idx_t idx_value =
                topk_idx[token_idx * num_topk + send_lane_id];
            idx_value =
                (idx_value >= recv_expert_begin && idx_value < recv_expert_end)
                ? idx_value - recv_expert_begin
                : -1;
            channel_topk_idx_buffers[dst_slot_idx * num_topk + send_lane_id] =
                idx_value;

            float weight_value =
                topk_weights[token_idx * num_topk + send_lane_id];
            weight_value = (idx_value >= 0) ? weight_value : 0.0f;
            channel_topk_weights_buffers
                [dst_slot_idx * num_topk + send_lane_id] = weight_value;
          }

          // x_scales (FP8 path; degenerates to no-op when num_scales == 0).
#pragma unroll
          for (int i = send_lane_id; i < num_scales; i += kWarpSize) {
            channel_x_scales_buffers[dst_slot_idx * num_scales + i] =
                x_scales[token_idx * num_scales + i];
          }
        }

        chunk_token_idx++;
        token_idx++;
      }

      // All sender warps for this dst_rank converge before publishing tail.
      __syncthreads();
      // AMD/xGMI: the payload `st_na_global` writes (UNROLLED_WARP_COPY above)
      // are non-temporal and not ordered against the tail publish below, so an
      // explicit system fence flushes them to HBM before the tail becomes
      // visible. The tail itself is then RELEASE-stored to pair with the
      // receiver's acquire-load.
#ifdef __HIP_PLATFORM_AMD__
      memory_fence();
#endif
      if (send_warp_id_in_rank == 0 && send_lane_id == 0) {
        // Release/acquire on both platforms. A relaxed tail store/load
        // handshake races on AMD: a relaxed read establishes no ordering, so
        // the receiver can observe the advanced tail but read stale/unwritten
        // payload across xGMI → illegal access under async execution. The
        // release store (plus the fence above for the non-temporal payload)
        // carries the happens-before to the receiver's ld_acquire_sys_global.
        st_release_sys_global(
            channel_tail_idx.buffer(), cached_channel_tail_idx);
      }
    }
  } else {
    // ------------------- RECEIVER PATH -------------------
    constexpr int num_recv_warps = kNumThreads / kWarpSize;
    constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
    const int recv_thread_id = thread_id;
    const int recv_lane_id = recv_thread_id % kWarpSize;
    const int recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
    const int recv_warp_id_in_rank = recv_thread_id_in_rank / kWarpSize;
    EP_DEVICE_ASSERT(kNumRanks <= kWarpSize);
    EP_DEVICE_ASSERT(num_recv_warps % kNumRanks == 0);

    int* rank_prefix_matrix = reinterpret_cast<int*>(buffer_ptrs[rank]);
    int rank_offset = responsible_rank > 0
        ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank]
        : 0;

    int total_offset = 0, num_tokens_to_recv = 0;
    // Wait for the sender to publish this channel's start/end offsets. Guard
    // both spins with the same timeout/trap as the sibling loops below so a
    // stuck/failed sender traps with a diagnostic instead of hanging the GPU
    // unrecoverably. The offsets are published before any data movement, so a
    // healthy sender never approaches NUM_TIMEOUT_CYCLES.
    long long offset_wait_start = wall_clock64_compat();
    while (recv_lane_id == 0 &&
           (total_offset = ld_volatile_global(channel_start_offset.buffer())) ==
               0) {
      long long now = wall_clock64_compat();
      if ((now > offset_wait_start ? now - offset_wait_start : 0) >
          NUM_TIMEOUT_CYCLES) {
        printf(
            "moe_ep dispatch receiver start-offset timeout rank %d channel %d\n",
            rank,
            responsible_channel);
        trap_kernel();
      }
    }
    offset_wait_start = wall_clock64_compat();
    while (recv_lane_id == 0 &&
           (num_tokens_to_recv =
                ld_volatile_global(channel_end_offset.buffer())) == 0) {
      long long now = wall_clock64_compat();
      if ((now > offset_wait_start ? now - offset_wait_start : 0) >
          NUM_TIMEOUT_CYCLES) {
        printf(
            "moe_ep dispatch receiver end-offset timeout rank %d channel %d\n",
            rank,
            responsible_channel);
        trap_kernel();
      }
    }
    if (recv_lane_id == 0) {
      total_offset = -total_offset - 1;
      num_tokens_to_recv = -num_tokens_to_recv - 1;
      if (recv_warp_id_in_rank == 0) {
        recv_channel_offset
            [responsible_rank * num_channels + responsible_channel] =
                total_offset;
      }
      num_tokens_to_recv -= total_offset;
    }
    total_offset = shfl_sync_compat(total_offset, 0);
    total_offset += rank_offset;
    num_tokens_to_recv = shfl_sync_compat(num_tokens_to_recv, 0);

    __shared__ volatile int shared_channel_tail_idx[kNumRanks];

    long long start_time = wall_clock64_compat();
    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    while (num_tokens_to_recv > 0) {
      while (recv_thread_id_in_rank == 0) {
        // Acquire-load the tail to pair with the sender's release-store: the
        // acquire both orders the subsequent payload reads after this load and
        // makes the producer's payload writes visible across xGMI. A relaxed
        // load would establish neither, racing on stale payload.
        cached_channel_tail_idx =
            ld_acquire_sys_global(channel_tail_idx.buffer());
        if (cached_channel_head_idx != cached_channel_tail_idx) {
          shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
          break;
        }
        long long now = wall_clock64_compat();
        long long elapsed = now > start_time ? (now - start_time) : 0;
        if (elapsed > NUM_TIMEOUT_CYCLES) {
          printf(
              "moe_ep dispatch receiver timeout rank %d channel %d remain %d\n",
              rank,
              responsible_channel,
              num_tokens_to_recv);
          trap_kernel();
        }
      }

      __syncthreads();
      cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

      const int num_recv_tokens =
          cached_channel_tail_idx - cached_channel_head_idx;
      for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens;
           chunk_idx += num_recv_warps_per_rank) {
        const int token_idx_in_buffer =
            (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        const int4* shifted_buffer =
            channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
        int4* shifted_recv = recv_x +
            static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
        MOE_EP_UNROLLED_WARP_COPY(
            kIntranodeUnrollFactor,
            recv_lane_id,
            hidden_int4,
            shifted_recv,
            shifted_buffer,
            ld_nc_global,
            st_na_global);
      }

#pragma unroll 4
      for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank;
           chunk_idx < cached_channel_tail_idx;
           chunk_idx += kWarpSize * num_recv_warps_per_rank) {
        recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] =
            ld_nc_global(
                channel_src_idx_buffers.buffer() +
                chunk_idx % num_recv_buffer_tokens);
      }

#pragma unroll 4
      for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk;
           idx += kWarpSize * num_recv_warps_per_rank) {
        const int chunk_idx = idx / num_topk;
        const int token_topk_idx = idx % num_topk;
        const int token_idx_in_buffer =
            (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        const int64_t recv_idx =
            static_cast<int64_t>(total_offset + chunk_idx) * num_topk +
            token_topk_idx;
        const int buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
        recv_topk_idx[recv_idx] =
            ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
        recv_topk_weights[recv_idx] =
            ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
      }

#pragma unroll 4
      for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales;
           i += kWarpSize * num_recv_warps_per_rank) {
        const int chunk_idx = i / num_scales;
        const int scales_idx = i % num_scales;
        const int token_idx_in_buffer =
            (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        recv_x_scales
            [static_cast<int64_t>(total_offset + chunk_idx) * num_scales +
             scales_idx] =
                ld_nc_global(
                    channel_x_scales_buffers.buffer() +
                    token_idx_in_buffer * num_scales + scales_idx);
      }

      cached_channel_head_idx += num_recv_tokens;
      total_offset += num_recv_tokens;
      __syncthreads();

      if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 &&
          recv_lane_id == 0) {
        st_relaxed_sys_global(
            channel_head_idx.buffer(), cached_channel_head_idx);
      }

      num_tokens_to_recv -= num_recv_tokens;
    }
  }

  // Clean unused recv_topk_idx slots to -1.
  if (num_worst_tokens > 0) {
    int* rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
    const int num_recv_tokens =
        rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
    const int clean_start = num_recv_tokens * num_topk + sm_id * kNumThreads;
    const int clean_end = num_worst_tokens * num_topk;
    const int clean_stride = num_sms * kNumThreads;
#pragma unroll
    for (int i = clean_start + thread_id; i < clean_end; i += clean_stride) {
      recv_topk_idx[i] = -1;
    }
  }
}

} // namespace

void intranode_dispatch(
    void* recv_x,
    float* recv_x_scales,
    std::int64_t* recv_topk_idx,
    float* recv_topk_weights,
    int* recv_src_idx,
    int* recv_channel_offset,
    int* send_head,
    const void* x,
    const float* x_scales,
    const std::int64_t* topk_idx,
    const float* topk_weights,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    int num_tokens,
    int num_worst_tokens,
    int hidden_int4,
    int num_topk,
    int num_experts,
    int scale_token_stride,
    int scale_hidden_stride,
    void** buffer_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
  constexpr int kNumThreads = (kWarpSize == 64 ? 1024 : 512);

  EP_HOST_ASSERT(num_sms % 2 == 0);
  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);

  const int num_scales = (scale_token_stride > 0 || scale_hidden_stride > 0)
      ? scale_hidden_stride
      : 0;

#define INTRANODE_DISPATCH_CASE(ranks)                 \
  LAUNCH_KERNEL_NON_COOPERATIVE(                       \
      &cfg,                                            \
      (intranode_dispatch_kernel<ranks, kNumThreads>), \
      reinterpret_cast<int4*>(recv_x),                 \
      recv_x_scales,                                   \
      recv_src_idx,                                    \
      recv_topk_idx,                                   \
      recv_topk_weights,                               \
      recv_channel_offset,                             \
      send_head,                                       \
      reinterpret_cast<const int4*>(x),                \
      x_scales,                                        \
      topk_idx,                                        \
      topk_weights,                                    \
      is_token_in_rank,                                \
      channel_prefix_matrix,                           \
      num_tokens,                                      \
      num_worst_tokens,                                \
      hidden_int4,                                     \
      num_topk,                                        \
      num_experts,                                     \
      num_scales,                                      \
      scale_token_stride,                              \
      scale_hidden_stride,                             \
      buffer_ptrs,                                     \
      rank,                                            \
      num_max_send_tokens,                             \
      num_recv_buffer_tokens)

  SWITCH_RANKS(INTRANODE_DISPATCH_CASE);
#undef INTRANODE_DISPATCH_CASE
}

} // namespace comms::prims::moe_ep::kernels
