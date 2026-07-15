// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// combine kernel. NVLink-only (single-node) version — replaces rocSHMEM
// with direct peer-mapped IPC buffer writes via pipes NVLink primitives.

#include "comms/prims/collectives/moe_ep/cpp/low_latency/kernels/Combine.cuh"

#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/EpBuffer.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Exception.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelConfigs.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelUtils.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Launch.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/ibgda/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::moe_ep::kernels {

namespace {

constexpr int kLLSendPhase = 1;
constexpr int kLLRecvPhase = 2;

__device__ void grid_barrier_ll_comb(int* global_counter, int num_blocks) {
  __syncthreads();
  int ret = 0;
  if (threadIdx.x == 0) {
    __threadfence();
    ret = atomicAdd(global_counter, 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (ret == num_blocks - 1) {
#ifdef __HIP_PLATFORM_AMD__
      __hip_atomic_store(
          global_counter, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
      atomicExch(global_counter, 0);
#endif
    } else {
      while (true) {
#ifdef __HIP_PLATFORM_AMD__
        int val = __hip_atomic_load(
            global_counter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
        int val = atomicAdd(global_counter, 0);
#endif
        if (val == num_blocks || val == 0)
          break;
      }
    }
  }
  __syncthreads();
}

template <
    int kNumWarpGroups,
    int kNumWarpsPerGroup,
    int kHidden,
    int kNumMaxTopk>
__global__
__launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * kWarpSize, 1) void ll_combine_kernel(
    void* combined_x,
    void* rdma_recv_x,
    int64_t* rdma_recv_flag,
    void* rdma_send_x,
    const void* x,
    const int64_t* topk_idx,
    const float* topk_weights,
    const int* src_info,
    const int64_t* layout_range,
    int* global_atomic_counter,
    int64_t* next_clean,
    int num_next_clean_int,
    int* atomic_clean_flag,
    int num_combined_tokens,
    int hidden,
    int num_topk,
    int num_max_dispatch_tokens_per_rank,
    int num_experts,
    int rank,
    int num_ranks,
    void** buffer_ptrs,
    comms::prims::MultipeerIbgdaDeviceTransport* device_transport,
    const comms::prims::IbgdaLocalBuffer* local_x_buf,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_x,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_flag,
    int phases,
    bool zero_copy) {
  const int sm_id = static_cast<int>(blockIdx.x);
  const int num_sms = static_cast<int>(gridDim.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int warp_id = thread_id / kWarpSize;
  const int lane_id = get_lane_id();
  const int num_local_experts = num_experts / num_ranks;
  const int warp_group_id = warp_id / kNumWarpsPerGroup;
  const int sub_warp_id = warp_id % kNumWarpsPerGroup;
  const int responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

  constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(uint16_t);
  const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
  constexpr size_t num_bytes_per_slot =
      sizeof(int4) + kHidden * sizeof(uint16_t);

  // ---- SEND PHASE ----
  if ((phases & kLLSendPhase) == 0)
    goto LL_COMBINE_RECV;

  // Clean next buffer
  if (sm_id == 0 && warp_group_id == 0 && sub_warp_id == 0) {
#pragma unroll
    for (int i = lane_id; i < num_next_clean_int; i += kWarpSize) {
      next_clean[i] = 0;
    }
    syncwarp();
    if (lane_id == 0) {
      atomicAdd(atomic_clean_flag, num_experts);
    }
  }

  if (responsible_expert_idx < num_experts) {
    const int dst_rank = responsible_expert_idx / num_local_experts;
    const int local_expert_idx = responsible_expert_idx % num_local_experts;
    const int global_expert_idx = rank * num_local_experts + local_expert_idx;
    const int64_t layout =
        __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
    const auto* local_x = reinterpret_cast<const int4*>(x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank *
            hidden_bf16_int4;
    const auto* local_src_info = src_info +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;

    // layout = (begin_idx << 32) | count — matches dispatch encoding
    int offset = static_cast<int>(layout >> 32);
    int num_tokens_to_send = static_cast<int>(layout & 0xFFFFFFFF);

    for (int token_idx = offset + sub_warp_id;
         token_idx < offset + num_tokens_to_send;
         token_idx += kNumWarpsPerGroup) {
      const auto* x_int4 = local_x + token_idx * hidden_bf16_int4;
      int src_idx = __ldg(local_src_info + token_idx);

      // Hybrid path — same-node IPC vs cross-node IBGDA.
      if (buffer_ptrs[dst_rank] != nullptr) {
        // Same-node: NVLink IPC write into peer's symmetric buffer.
        auto* peer_base = reinterpret_cast<uint8_t*>(buffer_ptrs[dst_rank]);
        uintptr_t recv_x_offset = reinterpret_cast<uintptr_t>(rdma_recv_x) -
            reinterpret_cast<uintptr_t>(buffer_ptrs[rank]);
        auto* dst_ptr = reinterpret_cast<int4*>(
            peer_base + recv_x_offset +
            (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) *
                num_bytes_per_slot +
            sizeof(int4));

        MOE_EP_UNROLLED_WARP_COPY(
            kIntranodeUnrollFactor,
            lane_id,
            static_cast<int>(hidden_bf16_int4),
            dst_ptr,
            x_int4,
            ld_nc_global,
            st_na_global);
      } else if (
          device_transport != nullptr && local_x_buf != nullptr &&
          peer_remote_recv_x != nullptr) {
        // Cross-node IBGDA RDMA put. The user `x` tensor is NOT
        // RDMA-registered, so we first stage this token's hidden data into
        // the registered `combine_send_x` slot, then RDMA from there.
        auto& peer_transport = device_transport->get(dst_rank);
        // Stage x[token] -> rdma_send_x slot (data area, past the int4 header).
        const std::size_t send_slot =
            static_cast<std::size_t>(local_expert_idx) * num_ranks *
                num_max_dispatch_tokens_per_rank +
            static_cast<std::size_t>(token_idx);
        auto* stage_ptr = reinterpret_cast<int4*>(
            reinterpret_cast<uint8_t*>(rdma_send_x) +
            send_slot * num_bytes_per_slot + sizeof(int4));
        MOE_EP_UNROLLED_WARP_COPY(
            kIntranodeUnrollFactor,
            lane_id,
            static_cast<int>(hidden_bf16_int4),
            stage_ptr,
            x_int4,
            ld_nc_global,
            st_na_global);
        syncwarp();
#ifdef __HIP_PLATFORM_AMD__
        // Retire the non-coherent VMEM stores above before the RDMA reads them.
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        __builtin_amdgcn_s_waitcnt(0);
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
#endif
        // Local source: the staged slot in the registered combine_send_x.
        auto local_buf = local_x_buf->subBuffer(
            send_slot * num_bytes_per_slot + sizeof(int4));
        // Remote dest: peer's combine_recv_x slot for [global_expert_idx,
        // src_idx], offset past the int4 slot header.
        std::size_t remote_offset =
            (static_cast<std::size_t>(global_expert_idx) *
                 num_max_dispatch_tokens_per_rank +
             static_cast<std::size_t>(src_idx)) *
                num_bytes_per_slot +
            sizeof(int4);
        auto remote_slot =
            peer_remote_recv_x[dst_rank].subBuffer(remote_offset);
        auto warp_grp = make_warp_group();
        // Key on this expert via block_id (the transport indexes QPs by
        // block_id; see Dispatch.cu rationale): spreads QP-spinlock contention
        // across the per-expert QP band. Data-before-flag ordering comes from
        // the synchronous put drain + the __syncthreads below, not same-QP
        // FIFO.
        warp_grp.block_id = static_cast<uint32_t>(local_expert_idx);
        peer_transport.put(
            warp_grp,
            local_buf,
            remote_slot,
            kHidden * sizeof(uint16_t),
            IbgdaRemoteBuffer{}); // flag write below, same QP, ordered after
      } else {
        if (lane_id == 0) {
          printf(
              "moe_ep LL combine: cross-node peer with no IBGDA transport "
              "(rank=%d, dst_rank=%d).\n",
              rank,
              dst_rank);
          trap_kernel();
        }
      }
    }

    __syncthreads();
    if (sub_warp_id == 0 && lane_id == 0) {
      while (ld_volatile_global(atomic_clean_flag) == 0) {
      }

      // Write completion flag to peer's rdma_recv_flag.
      // Hybrid: NVLink IPC store for same-node, IBGDA store for cross-node.
      if (buffer_ptrs[dst_rank] != nullptr) {
        auto* peer_base = reinterpret_cast<uint8_t*>(buffer_ptrs[dst_rank]);
        uintptr_t flag_offset = reinterpret_cast<uintptr_t>(rdma_recv_flag) -
            reinterpret_cast<uintptr_t>(buffer_ptrs[rank]);
        auto* peer_flag = reinterpret_cast<int64_t*>(peer_base + flag_offset);
#ifdef __HIP_PLATFORM_AMD__
        __hip_atomic_store(
            peer_flag + global_expert_idx,
            static_cast<int64_t>(1),
            __ATOMIC_RELEASE,
            __HIP_MEMORY_SCOPE_SYSTEM);
#else
        *(peer_flag + global_expert_idx) = 1;
        __threadfence_system();
#endif
      } else if (
          device_transport != nullptr && peer_remote_recv_flag != nullptr &&
          local_x_buf != nullptr) {
        // Cross-node IBGDA. Use a regular RDMA Write (8B inline) of the
        // flag value (1) instead of atomic-FA. The receiver only checks
        // `flag != 0` — atomicity isn't required, and plain put is the
        // most-validated cross-host RDMA path on AMD.
        auto& peer_transport = device_transport->get(dst_rank);
        std::size_t flag_offset =
            static_cast<std::size_t>(global_expert_idx) * sizeof(int64_t);
        auto remote_slot =
            peer_remote_recv_flag[dst_rank].subBuffer(flag_offset);
        int64_t local_flag = 1;
        comms::prims::IbgdaLocalBuffer local_buf = local_x_buf->subBuffer(0);
        // INVARIANT: local_flag is a device stack variable OUTSIDE the
        // RDMA-registered combine_send_x region; we borrow local_x_buf's lkey
        // only to fill a valid WQE lkey field. This is safe ONLY because a
        // single 8B RDMA write is inline-encoded into the WQE at prepare time,
        // so the backend reads *local_buf.ptr synchronously on this thread and
        // never hands the (unregistered) stack address to the NIC for DMA. A
        // larger payload would spill out of the inline segment and DMA the
        // unregistered address (LOCAL_PROT / garbage), so pin the size here.
        static_assert(
            sizeof(local_flag) == sizeof(int64_t),
            "cross-node combine flag must stay a single 8B inline-encoded word; "
            "a larger payload would DMA the unregistered local_flag stack "
            "address through the borrowed lkey");
        local_buf.ptr = &local_flag;
        auto thr = make_thread_solo();
        // Key on this expert via block_id (same expert QP band as its data
        // puts). The flag is ordered after the data by the synchronous put
        // drain + __syncthreads above, not by same-QP FIFO. 8B INLINE-encoded.
        thr.block_id = static_cast<uint32_t>(local_expert_idx);
        peer_transport.put(
            thr,
            local_buf,
            remote_slot,
            sizeof(int64_t),
            comms::prims::IbgdaRemoteBuffer{});
      } else {
        printf(
            "moe_ep LL combine flag: cross-node peer with no IBGDA "
            "transport (rank=%d, dst_rank=%d).\n",
            rank,
            dst_rank);
        trap_kernel();
      }
      atomicAdd(atomic_clean_flag, -1);
    }
  }

  // ---- RECEIVE PHASE ----
LL_COMBINE_RECV:
  if ((phases & kLLRecvPhase) == 0) {
    return;
  }

  if (responsible_expert_idx < num_experts) {
    if (sub_warp_id == 0 && lane_id == 0) {
      long long start = wall_clock64_compat();
      while (*reinterpret_cast<volatile int64_t*>(
                 rdma_recv_flag + responsible_expert_idx) == 0) {
        if (wall_clock64_compat() - start > NUM_TIMEOUT_CYCLES) {
          printf(
              "moe_ep LL combine recv timeout rank=%d expert=%d\n",
              rank,
              responsible_expert_idx);
          trap_kernel();
        }
      }
      // Zero flag so the next combine starts fresh
      rdma_recv_flag[responsible_expert_idx] = 0;
    }
  }

  grid_barrier_ll_comb(global_atomic_counter, num_sms);

  // Reduce: weighted sum across top-k experts. Threads stride over the hidden
  // dimension so every hidden_bf16_int4 position is written even when it
  // exceeds the block size (blockDim.x).
  for (int token_idx = sm_id; token_idx < num_combined_tokens;
       token_idx += num_sms) {
    int reg_topk_idx[kNumMaxTopk];
    float reg_topk_weights[kNumMaxTopk];
#pragma unroll
    for (int i = 0; i < num_topk; ++i) {
      reg_topk_idx[i] =
          static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
      reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
    }

    for (int h = thread_id; h < static_cast<int>(hidden_bf16_int4);
         h += blockDim.x) {
      float combined_values[kNumElemsPerInt4] = {0.0f};
#pragma unroll
      for (int i = 0; i < num_topk; ++i) {
        if (reg_topk_idx[i] >= 0) {
          auto* row = reinterpret_cast<const uint8_t*>(rdma_recv_x) +
              (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) *
                  num_bytes_per_slot +
              sizeof(int4);
          auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(row) + h);
          const auto* x_bf16 = reinterpret_cast<const uint16_t*>(&x_vec);
#pragma unroll 4
          for (int j = 0; j < kNumElemsPerInt4; ++j) {
            float val = __uint_as_float(static_cast<uint32_t>(x_bf16[j]) << 16);
            combined_values[j] += val * reg_topk_weights[i];
          }
        }
      }

      int4 combined_int4;
      auto* out_bf16 = reinterpret_cast<uint16_t*>(&combined_int4);
#pragma unroll 4
      for (int j = 0; j < kNumElemsPerInt4; ++j) {
        uint32_t bits = __float_as_uint(combined_values[j]);
        out_bf16[j] =
            static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
      }
      (reinterpret_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[h] =
          combined_int4;
    }
  }
}

} // namespace

void low_latency_combine(
    void* combined_x,
    void* rdma_recv_x,
    std::int64_t* rdma_recv_flag,
    void* rdma_send_x,
    const void* x,
    const std::int64_t* topk_idx,
    const float* topk_weights,
    const int* src_info,
    const std::int64_t* layout_range,
    int* global_atomic_counter,
    std::int64_t* next_clean,
    int num_next_clean_int,
    int num_combined_tokens,
    int hidden,
    int num_max_dispatch_tokens_per_rank,
    int num_topk,
    int num_experts,
    int rank,
    int num_ranks,
    void* workspace,
    void** buffer_ptrs,
    comms::prims::MultipeerIbgdaDeviceTransport* device_transport,
    const comms::prims::IbgdaLocalBuffer* local_x_buf,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_x,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_flag,
    int phase,
    bool zero_copy,
    cudaStream_t stream) {
#ifdef __HIP_PLATFORM_AMD__
  constexpr int kNumWarpsPerGroup = 8;
  constexpr int kNumWarpGroups = 2;
#else
  constexpr int kNumWarpsPerGroup = 4;
  constexpr int kNumWarpGroups = 2;
#endif
  constexpr int kNumMaxTopk = 9;

  const int num_warps = kNumWarpGroups * kNumWarpsPerGroup;
  const int num_sms = cell_div(num_experts, kNumWarpGroups);

  auto* atomic_clean_flag = reinterpret_cast<int*>(workspace);

  SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);

#define LL_COMBINE_LAUNCH(H)                                                  \
  LAUNCH_KERNEL_NON_COOPERATIVE(                                              \
      &cfg,                                                                   \
      (ll_combine_kernel<kNumWarpGroups, kNumWarpsPerGroup, H, kNumMaxTopk>), \
      combined_x,                                                             \
      rdma_recv_x,                                                            \
      rdma_recv_flag,                                                         \
      rdma_send_x,                                                            \
      x,                                                                      \
      topk_idx,                                                               \
      topk_weights,                                                           \
      src_info,                                                               \
      layout_range,                                                           \
      global_atomic_counter,                                                  \
      next_clean,                                                             \
      num_next_clean_int,                                                     \
      atomic_clean_flag,                                                      \
      num_combined_tokens,                                                    \
      hidden,                                                                 \
      num_topk,                                                               \
      num_max_dispatch_tokens_per_rank,                                       \
      num_experts,                                                            \
      rank,                                                                   \
      num_ranks,                                                              \
      buffer_ptrs,                                                            \
      device_transport,                                                       \
      local_x_buf,                                                            \
      peer_remote_recv_x,                                                     \
      peer_remote_recv_flag,                                                  \
      phase,                                                                  \
      zero_copy)

  SWITCH_HIDDEN(LL_COMBINE_LAUNCH);
#undef LL_COMBINE_LAUNCH
}

} // namespace comms::prims::moe_ep::kernels
