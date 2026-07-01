// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/moe_ep/cpp/intranode/kernels/Combine.cuh"

#include <climits>

#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/EpBuffer.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Exception.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelConfigs.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelUtils.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Launch.cuh"

// Symmetric counterpart to dispatch:
//   - Sender SMs (even) iterate over local tokens, push payload + weights
//     to peer's recv-buffer.
//   - Receiver SMs (odd) split into one head-updater warp + N reducer warps;
//     each reducer warp picks up tokens, sums payload across topk source
//     ranks (with optional bias_0 / bias_1), and writes into recv_x.

namespace comms::prims::moe_ep::kernels {

namespace {

template <typename DType, int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1) intranode_combine_kernel(
    DType* recv_x,
    float* recv_topk_weights,
    const DType* x,
    const float* topk_weights,
    const DType* bias_0,
    const DType* bias_1,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    int* send_head,
    int num_tokens,
    int num_recv_tokens,
    int hidden,
    int num_topk,
    void** buffer_ptrs,
    int rank,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
  const int num_sms = static_cast<int>(gridDim.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int lane_id = thread_id % kWarpSize;
  const int sm_id = static_cast<int>(blockIdx.x);
  const int num_channels = num_sms / 2;
  const bool is_sender = sm_id % 2 == 0;
  const int responsible_channel = sm_id / 2;

  EP_DEVICE_ASSERT(num_topk <= kWarpSize);

  constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(DType);
  const int hidden_int4 = hidden * sizeof(DType) / sizeof(int4);
  const int4* x_int4 = reinterpret_cast<const int4*>(x);
  const int4* bias_0_int4 = reinterpret_cast<const int4*>(bias_0);
  const int4* bias_1_int4 = reinterpret_cast<const int4*>(bias_1);
  int4* recv_int4 = reinterpret_cast<int4*>(recv_x);

  if (is_sender) {
    // ------------------- SENDER PATH -------------------
    constexpr int num_send_warps = kNumThreads / kWarpSize;
    constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
    const int send_rank_id =
        (responsible_channel + thread_id / kWarpSize) % kNumRanks;
    const int send_warp_id_in_rank = thread_id / kWarpSize / kNumRanks;

    void* ptr = reinterpret_cast<void*>(
        reinterpret_cast<int8_t*>(buffer_ptrs[send_rank_id]));
    const int num_channels_total = num_channels * kNumRanks;
    const int channel_rank_offset = responsible_channel * kNumRanks + rank;

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
    auto channel_topk_weights_buffers = Buffer<float>(
        ptr,
        num_channels_total * num_recv_buffer_tokens * num_topk,
        channel_rank_offset * num_recv_buffer_tokens * num_topk);

    const int rank_offset = send_rank_id > 0
        ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank]
        : 0;
    const int num_rank_tokens =
        rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
    const int channel_offset = channel_prefix_matrix
        [send_rank_id * num_channels + responsible_channel];
    const int num_channel_tokens =
        (responsible_channel == num_channels - 1
             ? num_rank_tokens
             : channel_prefix_matrix
                   [send_rank_id * num_channels + responsible_channel + 1]) -
        channel_offset;
    const int token_start_idx = rank_offset + channel_offset;
    const int token_end_idx = token_start_idx + num_channel_tokens;

    int current_channel_tail_idx = 0;
    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
      long long start_time = wall_clock64_compat();
      const int num_round_tokens =
          min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
      while (lane_id == 0) {
        int num_used_slots = current_channel_tail_idx -
            ld_volatile_global(channel_head_idx.buffer());
        if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens) {
          break;
        }
        long long now = wall_clock64_compat();
        long long elapsed = now > start_time ? (now - start_time) : 0;
        if (elapsed > NUM_TIMEOUT_CYCLES) {
          printf(
              "moe_ep combine sender timeout, rank %d channel %d\n",
              rank,
              responsible_channel);
          trap_kernel();
        }
      }
      syncwarp();

#pragma unroll
      for (int i = send_warp_id_in_rank; i < num_round_tokens;
           i += num_send_warps_per_rank) {
        const int dst_slot_idx =
            (current_channel_tail_idx + i) % num_recv_buffer_tokens;
        int4* shifted_x_buffers =
            channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
        const int4* shifted_x = x_int4 + (token_idx + i) * hidden_int4;
        // x_int4 here is recv_x from the prior dispatch (xGMI-written by
        // peers), so reads need cache-bypass on AMD too. Use ld_nc_global on
        // both NVIDIA and AMD.
        MOE_EP_UNROLLED_WARP_COPY(
            kIntranodeUnrollFactor,
            lane_id,
            hidden_int4,
            shifted_x_buffers,
            shifted_x,
            ld_nc_global,
            st_na_global);

        if (lane_id == 0) {
          channel_src_idx_buffers[dst_slot_idx] =
              ld_nc_global(src_idx + token_idx + i);
        }
        if (num_topk > 0 && lane_id < num_topk) {
          channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
              ld_nc_global(topk_weights + (token_idx + i) * num_topk + lane_id);
        }
      }
      token_idx += num_round_tokens;
      current_channel_tail_idx += num_round_tokens;

      __syncthreads();
      // AMD: the payload writes are non-temporal; this explicit system fence
      // flushes them to HBM before the tail publish (see Dispatch.cu).
#ifdef __HIP_PLATFORM_AMD__
      memory_fence();
#endif
      if (lane_id == 0 && send_warp_id_in_rank == 0) {
        // Release-store the tail to pair with the receiver's acquire-load on
        // both platforms; a relaxed handshake races on stale payload across
        // xGMI (see Dispatch.cu).
        st_release_sys_global(
            channel_tail_idx.buffer(), current_channel_tail_idx);
      }
    }
  } else {
    // ------------------- RECEIVER PATH -------------------
    constexpr int num_recv_warps = kNumThreads / kWarpSize;
    const int recv_warp_id = thread_id / kWarpSize;
    EP_DEVICE_ASSERT(kNumRanks <= kWarpSize && kNumThreads > kWarpSize);
    EP_DEVICE_ASSERT(thread_id >= 0 && kNumThreads % kWarpSize == 0);

    __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
    __shared__ volatile int channel_tail_idx[kNumRanks];
    __shared__ volatile bool warp_retired[num_recv_warps];
    if (thread_id < num_recv_warps) {
      warp_retired[thread_id] = false;
    }
    if (lane_id < kNumRanks) {
      warp_channel_head_idx[recv_warp_id][lane_id] = 0;
    }
    if (thread_id < kNumRanks) {
      channel_tail_idx[thread_id] = 0;
    }
    __syncthreads();

    if (thread_id < kWarpSize) {
      int* channel_head_idx_ptr = reinterpret_cast<int*>(buffer_ptrs[rank]) +
          responsible_channel * kNumRanks + lane_id;
      int* channel_tail_idx_ptr =
          channel_head_idx_ptr + num_channels * kNumRanks;

      int last_head = 0;
      while (lane_id < kNumRanks) {
        bool retired = true;
#pragma unroll
        for (int i = 1; i < num_recv_warps; ++i) {
          retired = retired && warp_retired[i];
        }
        if (retired) {
          break;
        }

        // Acquire-load the tail to pair with the sender's release-store on both
        // platforms: the acquire orders the payload reads after it and makes
        // the producer's payload visible across xGMI (a relaxed load races on
        // stale payload — see Dispatch.cu).
        channel_tail_idx[lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);

        int min_head = INT_MAX;
#pragma unroll
        for (int i = 1; i < num_recv_warps; ++i) {
          if (!warp_retired[i]) {
            min_head = min(min_head, warp_channel_head_idx[i][lane_id]);
          }
        }
        if (min_head != INT_MAX && min_head > last_head) {
          last_head = min_head;
          st_relaxed_sys_global(channel_head_idx_ptr, last_head);
        }
      }
    } else {
      Buffer<int4> channel_x_buffers[kNumRanks];
      Buffer<float> channel_topk_weights_buffers[kNumRanks];

#pragma unroll
      for (int i = 0; i < kNumRanks; ++i) {
        const int channel_rank_offset = responsible_channel * kNumRanks + i;
        const int num_channels_total = num_channels * kNumRanks;
        void* ptr = reinterpret_cast<void*>(
            reinterpret_cast<int8_t*>(buffer_ptrs[rank]) +
            2 * num_channels * kNumRanks * sizeof(int));

        channel_x_buffers[i] = Buffer<int4>(
            ptr,
            num_channels_total * num_recv_buffer_tokens * hidden_int4,
            channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
        ptr = reinterpret_cast<void*>(
            reinterpret_cast<int8_t*>(ptr) +
            num_channels_total * num_recv_buffer_tokens * sizeof(int));
        channel_topk_weights_buffers[i] = Buffer<float>(
            ptr,
            num_channels_total * num_recv_buffer_tokens * num_topk,
            channel_rank_offset * num_recv_buffer_tokens * num_topk);
      }

      int token_start_idx = 0, token_end_idx = 0;
      get_channel_task_range(
          num_recv_tokens,
          num_channels,
          responsible_channel,
          token_start_idx,
          token_end_idx);

      for (int64_t token_idx = token_start_idx + recv_warp_id - 1;
           token_idx < token_end_idx;
           token_idx += num_recv_warps - 1) {
        int expected_head = -1;
        if (lane_id < kNumRanks) {
          expected_head =
              ld_nc_global(send_head + token_idx * kNumRanks + lane_id);
        }

        long long start_time = wall_clock64_compat();
        // Spin until every contributing rank has produced this token.
        while (true) {
          bool wait =
              (channel_tail_idx[lane_id] <= expected_head &&
               expected_head >= 0);
#ifdef __HIP_PLATFORM_AMD__
          uint64_t pred = __ballot(wait);
          if ((pred & kFullWarpMask) == 0) {
            break;
          }
#else
          if (!__any_sync(kFullWarpMask, wait)) {
            break;
          }
#endif
          long long now = wall_clock64_compat();
          long long elapsed = now > start_time ? (now - start_time) : 0;
          if (elapsed > NUM_TIMEOUT_CYCLES) {
            printf(
                "moe_ep combine receiver timeout rank %d channel %d expect %d\n",
                rank,
                responsible_channel,
                expected_head);
            trap_kernel();
          }
        }
        syncwarp();

        // Broadcast each lane's expected_head to its corresponding rank
        // index, then collect (rank, slot) pairs.
        int num_topk_ranks = 0;
        int topk_ranks[kNumRanks];
        int slot_indices[kNumRanks];
#pragma unroll
        for (int i = 0; i < kNumRanks; ++i) {
          int expected_head_i = shfl_sync_compat(expected_head, i);
          if (expected_head_i >= 0) {
            slot_indices[num_topk_ranks] =
                expected_head_i % num_recv_buffer_tokens;
            topk_ranks[num_topk_ranks++] = i;
          }
        }

        // Reduce hidden vector across topk source ranks.
#pragma unroll
        for (int i = lane_id; i < hidden_int4; i += kWarpSize) {
          int4 bias_0_value_int4 = bias_0_int4 != nullptr
              ? bias_0_int4[token_idx * hidden_int4 + i]
              : make_int4(0, 0, 0, 0);
          int4 bias_1_value_int4 = bias_1_int4 != nullptr
              ? bias_1_int4[token_idx * hidden_int4 + i]
              : make_int4(0, 0, 0, 0);

          int4 recv_value_int4[kNumRanks];
#pragma unroll
          for (int j = 0; j < num_topk_ranks; ++j) {
            recv_value_int4[j] = ld_nc_global(
                channel_x_buffers[topk_ranks[j]].buffer() +
                slot_indices[j] * hidden_int4 + i);
          }

          float values[kDtypePerInt4];
          const DType* bias_0_values =
              reinterpret_cast<const DType*>(&bias_0_value_int4);
          const DType* bias_1_values =
              reinterpret_cast<const DType*>(&bias_1_value_int4);
#pragma unroll
          for (int j = 0; j < kDtypePerInt4; ++j) {
            values[j] = static_cast<float>(bias_0_values[j]) +
                static_cast<float>(bias_1_values[j]);
          }

#pragma unroll
          for (int j = 0; j < num_topk_ranks; ++j) {
            const DType* recv_dtypes =
                reinterpret_cast<const DType*>(&recv_value_int4[j]);
#pragma unroll
            for (int k = 0; k < kDtypePerInt4; ++k) {
              values[k] += static_cast<float>(recv_dtypes[k]);
            }
          }

          int4 out_int4;
          DType* out_dtypes = reinterpret_cast<DType*>(&out_int4);
#pragma unroll
          for (int j = 0; j < kDtypePerInt4; ++j) {
            out_dtypes[j] = static_cast<DType>(values[j]);
          }
          recv_int4[token_idx * hidden_int4 + i] = out_int4;
        }

        if (lane_id < num_topk) {
          float value = 0;
#pragma unroll
          for (int i = 0; i < num_topk_ranks; ++i) {
            value += ld_nc_global(
                channel_topk_weights_buffers[topk_ranks[i]].buffer() +
                slot_indices[i] * num_topk + lane_id);
          }
          recv_topk_weights[token_idx * num_topk + lane_id] = value;
        }

        if (lane_id < kNumRanks) {
          warp_channel_head_idx[recv_warp_id][lane_id] =
              (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
        }
      }

      syncwarp();
      if (lane_id == 0) {
        warp_retired[recv_warp_id] = true;
      }
    }
  }
}

} // namespace

void intranode_combine(
    void* recv_x,
    float* recv_topk_weights,
    const void* x,
    const float* topk_weights,
    const void* bias_0,
    const void* bias_1,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    int* send_head,
    int num_tokens,
    int num_recv_tokens,
    int hidden,
    int num_topk,
    void** buffer_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
  // Wave-size-dependent thread budget: 1024 for 64-wide waves, else 768.
  constexpr int kNumThreads = (kWarpSize == 64 ? 1024 : 768);

  EP_HOST_ASSERT(num_sms % 2 == 0);
  EP_HOST_ASSERT(kNumThreads >= num_ranks * kWarpSize);
  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);

  // For Phase 1 we only support bf16. FP32 + the DType switch land in a
  // follow-up patch.
  using DType = gpu_bfloat16_t;

#define INTRANODE_COMBINE_CASE(ranks)                        \
  LAUNCH_KERNEL_NON_COOPERATIVE(                             \
      &cfg,                                                  \
      (intranode_combine_kernel<DType, ranks, kNumThreads>), \
      reinterpret_cast<DType*>(recv_x),                      \
      recv_topk_weights,                                     \
      reinterpret_cast<const DType*>(x),                     \
      topk_weights,                                          \
      reinterpret_cast<const DType*>(bias_0),                \
      reinterpret_cast<const DType*>(bias_1),                \
      src_idx,                                               \
      rank_prefix_matrix,                                    \
      channel_prefix_matrix,                                 \
      send_head,                                             \
      num_tokens,                                            \
      num_recv_tokens,                                       \
      hidden,                                                \
      num_topk,                                              \
      buffer_ptrs,                                           \
      rank,                                                  \
      num_max_send_tokens,                                   \
      num_recv_buffer_tokens)

  SWITCH_RANKS(INTRANODE_COMBINE_CASE);
#undef INTRANODE_COMBINE_CASE
}

} // namespace comms::prims::moe_ep::kernels
