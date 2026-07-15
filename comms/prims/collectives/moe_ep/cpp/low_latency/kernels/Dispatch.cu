// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// dispatch kernel. NVLink-only (single-node) version — replaces all
// rocSHMEM/NVSHMEM calls with direct peer-mapped IPC buffer writes via
// UNROLLED_WARP_COPY + system-scope atomics, matching the pipes NVLink
// transport primitives.

#include "comms/prims/collectives/moe_ep/cpp/low_latency/kernels/Dispatch.cuh"

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

// Phase flags selecting the send and/or receive stage of the kernel.
constexpr int kLLSendPhase = 1;
constexpr int kLLRecvPhase = 2;

__device__ void grid_barrier_ll(int* global_counter, int num_blocks) {
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

template <int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
__global__
__launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * kWarpSize, 1) void ll_dispatch_kernel(
    void* packed_recv_x,
    float* packed_recv_x_scales,
    int* packed_recv_src_info,
    int64_t* packed_recv_layout_range,
    int* packed_recv_count,
    int* global_atomic_counter,
    void* rdma_recv_x,
    int64_t* rdma_recv_count,
    void* rdma_x,
    const void* x,
    const int64_t* topk_idx,
    int* atomic_counter_per_expert,
    int* atomic_finish_counter_per_expert,
    int64_t* next_clean,
    int num_next_clean_int,
    int num_tokens,
    int num_max_dispatch_tokens_per_rank,
    int num_topk,
    int num_experts,
    int rank,
    int num_ranks,
    void** buffer_ptrs,
    comms::prims::MultipeerIbgdaDeviceTransport* device_transport,
    const comms::prims::IbgdaLocalBuffer* local_rdma_x_buf,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_x,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_count,
    int phases) {
  const int sm_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int warp_id = thread_id / kWarpSize;
  const int lane_id = get_lane_id();
  const int num_sms = static_cast<int>(gridDim.x);
  constexpr int num_warps = kNumWarpGroups * kNumWarpsPerGroup;
  const int num_local_experts = num_experts / num_ranks;
  const int warp_group_id = warp_id / kNumWarpsPerGroup;
  const int sub_warp_id = warp_id % kNumWarpsPerGroup;
  const int responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

  // BF16 mode: hidden bytes = kHidden * 2 (bf16 = 2 bytes)
  constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(uint16_t); // 8
  const size_t hidden_bytes = kHidden * sizeof(uint16_t);
  const size_t hidden_int4 = hidden_bytes / sizeof(int4);

  // Message package: [src_idx (int4)] + [hidden data (bf16)]
  const size_t num_bytes_per_msg = sizeof(int4) + hidden_bytes;
  const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);

  // ---- SEND PHASE ----
  if ((phases & kLLSendPhase) == 0)
    goto LL_DISPATCH_RECV;

  __shared__ int shared_num_tokens_sent_per_expert[kNumWarpGroups];

  if (warp_id < num_warps) {
    constexpr int num_threads = kNumWarpGroups * kNumWarpsPerGroup * kWarpSize;
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

    for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
      const auto x_int4 =
          reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
      auto rdma_x_src_idx = reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
      auto rdma_x_vec = reinterpret_cast<int4*>(
          reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));

      // Read top-k expert index for this warp
      auto dst_expert_idx = warp_id < num_topk
          ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id))
          : -1;
      // Write source token index
      if (thread_id == 0) {
        *rdma_x_src_idx = token_idx;
      }

      // Copy BF16 data to send buffer
#pragma unroll
      for (int i = thread_id; i < static_cast<int>(hidden_bf16_int4);
           i += num_threads) {
        rdma_x_vec[i] = __ldg(x_int4 + i);
      }

      __syncthreads();

      // Send to peer via NVLink (IPC peer-mapped buffer)
      if (dst_expert_idx >= 0) {
        int slot_idx = lane_id == 0
            ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1)
            : 0;
        slot_idx = shfl_sync_compat(slot_idx, 0);
        const int dst_rank = dst_expert_idx / num_local_experts;
        const int dst_expert_local_idx = dst_expert_idx % num_local_experts;

        // Source: local rdma_x send buffer
        const auto* src_int4_ptr =
            reinterpret_cast<const int4*>(rdma_x_src_idx);

        // Hybrid path.
        // Same-node peers (P2P-capable): NVLink IPC fast path via
        // UNROLLED_WARP_COPY into the peer's symmetric buffer.
        // Cross-node peers: RDMA via MultipeerIbgdaTransport.
        if (buffer_ptrs[dst_rank] != nullptr) {
          // Destination: peer's rdma_recv_x buffer via IPC.
          // Compute the offset of rdma_recv_x within the symmetric buffer,
          // then apply the same offset to the peer's buffer base.
          uintptr_t recv_x_off = reinterpret_cast<uintptr_t>(rdma_recv_x) -
              reinterpret_cast<uintptr_t>(buffer_ptrs[rank]);
          auto* dst_ptr = reinterpret_cast<int4*>(
              reinterpret_cast<uint8_t*>(buffer_ptrs[dst_rank]) + recv_x_off +
              dst_expert_local_idx * num_ranks *
                  num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
              rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
              slot_idx * num_bytes_per_msg);

          // NVLink copy (same as intranode UNROLLED_WARP_COPY)
          MOE_EP_UNROLLED_WARP_COPY(
              kIntranodeUnrollFactor,
              lane_id,
              static_cast<int>(num_int4_per_msg),
              dst_ptr,
              src_int4_ptr,
              ld_nc_global,
              st_na_global);
        } else {
          // Cross-node IBGDA RDMA path: warp-collective put of the token
          // message to the peer's recv buffer.
          if (device_transport != nullptr && local_rdma_x_buf != nullptr &&
              peer_remote_recv_x != nullptr) {
            auto& peer_transport = device_transport->get(dst_rank);
            // Local source: the rdma_x staging slot for this token
            auto local_buf = local_rdma_x_buf->subBuffer(
                static_cast<std::size_t>(token_idx) * num_bytes_per_msg);
            // Remote dest: peer's rdma_recv_x slot for [dst_expert_local_idx,
            // src_rank=rank, slot_idx]
            std::size_t remote_offset =
                static_cast<std::size_t>(dst_expert_local_idx) * num_ranks *
                    num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                static_cast<std::size_t>(rank) *
                    num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                static_cast<std::size_t>(slot_idx) * num_bytes_per_msg;
            auto remote_slot =
                peer_remote_recv_x[dst_rank].subBuffer(remote_offset);
            auto warp_grp = make_warp_group();
            // Key each put on the destination expert via block_id (the
            // transport indexes QPs by block_id, giving expert E its own band
            // of qpsPerBlockPerNic QPs per NIC). This spreads QP-spinlock
            // contention across the per-expert QPs — pinning everything to one
            // QP funnels every warp in the grid onto a single GPU spinlock
            // whose holder may be a non-resident wavefront -> deadlock under
            // load. Data-before-count does NOT depend on same-QP FIFO: the
            // synchronous BNXT put drains each token's CQE before its
            // finish-counter bump, and the count put (below) only fires once
            // the counter shows all of this expert's data has landed.
            warp_grp.block_id = static_cast<uint32_t>(dst_expert_local_idx);
            peer_transport.put(
                warp_grp,
                local_buf,
                remote_slot,
                num_bytes_per_msg,
                IbgdaRemoteBuffer{}); // no signal; count put below, same QP
          } else {
            // Should be unreachable in well-formed runs (caller passes
            // either valid IBGDA or all peers same-node). Trap to surface
            // misconfiguration.
            if (lane_id == 0) {
              printf(
                  "moe_ep LL dispatch: cross-node peer with no IBGDA "
                  "transport (rank=%d, dst_rank=%d). Configure with "
                  "MultipeerIbgdaTransport or use single-node mode.\n",
                  rank,
                  dst_rank);
              trap_kernel();
            }
          }
        }

        // Signal completion. The AMD path needs `s_waitcnt` to retire any
        // VMEM stores still in-flight (the IPC `MOE_EP_UNROLLED_WARP_COPY`
        // above is non-coherent and would otherwise still be pending when
        // the next warp/SM reads the destination). The signal-fence pair
        // around it forces the compiler not to reorder unrelated writes
        // across the barrier. NVIDIA doesn't need any of this — its store
        // ordering is already strict enough for this pattern.
        syncwarp();
#ifdef __HIP_PLATFORM_AMD__
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        __builtin_amdgcn_s_waitcnt(0);
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
#else
        memory_fence();
#endif
        if (lane_id == 0) {
          atomicAdd(atomic_finish_counter_per_expert + dst_expert_idx, 1);
        }
      }
    }
  }

  if (warp_id == num_warps - 1) {
    if (sm_id == 0) {
      // Clean next buffer
#pragma unroll
      for (int i = lane_id; i < num_next_clean_int; i += kWarpSize) {
        next_clean[i] = 0;
      }
    }

    // Count tokens per expert
    int expert_count[kNumWarpGroups] = {};
    const int expert_begin_idx = sm_id * kNumWarpGroups;
    const int expert_end_idx =
        min(expert_begin_idx + kNumWarpGroups, num_experts);

#pragma unroll 2
    for (int i = lane_id; i < num_tokens * num_topk; i += kWarpSize) {
      auto idx = static_cast<int>(__ldg(topk_idx + i));
      if (idx >= expert_begin_idx && idx < expert_end_idx) {
        expert_count[idx - expert_begin_idx]++;
      }
    }

#pragma unroll 2
    for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
      auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
      if (lane_id == 0) {
        shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
        atomicAdd(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
      }
    }
  }

  __syncthreads();

  // Issue count sends via NVLink
  if (responsible_expert_idx < num_experts && sub_warp_id == 0 &&
      lane_id == 0) {
    const int dst_rank = responsible_expert_idx / num_local_experts;
    const int dst_expert_local_idx = responsible_expert_idx % num_local_experts;
    const int num_tokens_sent = shared_num_tokens_sent_per_expert
        [responsible_expert_idx - sm_id * kNumWarpGroups];

    // Wait for all local sends to complete
    while (ld_volatile_global(
               atomic_finish_counter_per_expert + responsible_expert_idx) !=
           FINISHED_SUM_TAG) {
    }

    // Write token count to peer's rdma_recv_count.
    // Encode as a single int64: -(count+1) so 0 means "not yet sent".
    // Hybrid path: NVLink IPC store for same-node, IBGDA store for
    // cross-node.
    int64_t encoded = -static_cast<int64_t>(num_tokens_sent) - 1;
    if (buffer_ptrs[dst_rank] != nullptr) {
      auto* peer_recv_count = reinterpret_cast<int64_t*>(
          reinterpret_cast<uint8_t*>(buffer_ptrs[dst_rank]) +
          reinterpret_cast<uintptr_t>(rdma_recv_count) -
          reinterpret_cast<uintptr_t>(buffer_ptrs[rank]));
#ifdef __HIP_PLATFORM_AMD__
      __hip_atomic_store(
          peer_recv_count + dst_expert_local_idx * num_ranks + rank,
          encoded,
          __ATOMIC_RELEASE,
          __HIP_MEMORY_SCOPE_SYSTEM);
#else
      *(peer_recv_count + dst_expert_local_idx * num_ranks + rank) = encoded;
      __threadfence_system();
#endif
    } else if (
        device_transport != nullptr && peer_remote_recv_count != nullptr &&
        local_rdma_x_buf != nullptr) {
      // Cross-node IBGDA. Use a regular RDMA Write (8B inline) of the
      // `encoded` value instead of atomic-FA. Each (src_rank, dst_expert)
      // pair has a unique destination slot — atomicity is not needed.
      // Plain put is the most-validated cross-host RDMA path on AMD; the
      // atomic-FA path is reportedly flaky for AMD+GPU-direct on MI300X.
      auto& peer_transport = device_transport->get(dst_rank);
      std::size_t slot_offset =
          (static_cast<std::size_t>(dst_expert_local_idx) * num_ranks +
           static_cast<std::size_t>(rank)) *
          sizeof(int64_t);
      auto remote_slot =
          peer_remote_recv_count[dst_rank].subBuffer(slot_offset);
      // Stage `encoded` in a local int64. Address is GPU-readable (stack
      // register-spill or local memory). For 8B writes the BNXT/MLX5
      // backends use INLINE encoding — the data is copied into the WQE
      // at prepare time and never DMA-fetched from this address, so the
      // lkey we pass is irrelevant. We borrow `local_rdma_x_buf`'s lkeys
      // to satisfy IbgdaLocalBuffer's structural requirement.
      int64_t local_encoded = encoded;
      comms::prims::IbgdaLocalBuffer local_buf = local_rdma_x_buf->subBuffer(0);
      local_buf.ptr = &local_encoded;
      auto thr = make_thread_solo();
      // Key on this expert via block_id (same expert QP band as its data puts).
      // The count is ordered after all of the expert's data by the finish
      // counter above + the synchronous put drain, not by same-QP FIFO. The 8B
      // value is INLINE-encoded by the BNXT/MLX5 backend (<=16B), so
      // local_buf's lkey is irrelevant.
      thr.block_id = static_cast<uint32_t>(dst_expert_local_idx);
      peer_transport.put(
          thr,
          local_buf,
          remote_slot,
          sizeof(int64_t),
          comms::prims::IbgdaRemoteBuffer{});
    } else {
      printf(
          "moe_ep LL dispatch count signal: cross-node peer with no IBGDA "
          "transport (rank=%d, dst_rank=%d).\n",
          rank,
          dst_rank);
      trap_kernel();
    }

    // Clean workspace
    atomic_counter_per_expert[responsible_expert_idx] = 0;
    atomic_finish_counter_per_expert[responsible_expert_idx] = 0;
    if (dst_rank == 0) {
      packed_recv_count[dst_expert_local_idx] = 0;
    }
  }

  // ---- RECEIVE PHASE ----
LL_DISPATCH_RECV:
  if ((phases & kLLRecvPhase) == 0) {
    return;
  }

  // Grid sync for send-and-recv kernels
  if (phases & kLLSendPhase) {
    grid_barrier_ll(global_atomic_counter, num_sms);
  }

  // Receive and pack
  if (responsible_expert_idx < num_experts) {
    const int src_rank = responsible_expert_idx / num_local_experts;
    const int local_expert_idx = responsible_expert_idx % num_local_experts;
    auto* rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank *
            num_bytes_per_msg +
        src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
    auto* recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank *
            hidden_int4;
    auto* recv_src_info = packed_recv_src_info +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
    auto* recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;

    __shared__ int shared_num_recv_tokens[kNumWarpGroups];
    __shared__ int shared_recv_token_begin_idx[kNumWarpGroups];

    // Wait for tokens to arrive
    int num_recv_tokens = 0, recv_token_begin_idx = 0;
    if (sub_warp_id == 0 && lane_id == 0) {
      long long start_time = wall_clock64_compat();
      int64_t recv_val = 0;
      auto* slot_ptr = reinterpret_cast<volatile int64_t*>(
          rdma_recv_count + local_expert_idx * num_ranks + src_rank);
      while ((recv_val = *slot_ptr) == 0) {
        long long elapsed = wall_clock64_compat() - start_time;
        if (elapsed > NUM_TIMEOUT_CYCLES) {
          printf(
              "moe_ep LL dispatch recv timeout rank=%d expert=%d "
              "slot_ptr=%p slot_val=%lld src_rank=%d\n",
              rank,
              local_expert_idx,
              const_cast<int64_t*>(slot_ptr),
              (long long)recv_val,
              src_rank);
          trap_kernel();
        }
      }
      num_recv_tokens = static_cast<int>(-recv_val - 1);
      // Zero the count so the next dispatch starts fresh
      *(rdma_recv_count + local_expert_idx * num_ranks + src_rank) = 0;
      recv_token_begin_idx =
          atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
      shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
      shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
      // Pack layout range: (begin_idx << 32) | count.
      recv_range[src_rank] =
          (static_cast<int64_t>(recv_token_begin_idx) << 32) |
          static_cast<int64_t>(num_recv_tokens);
    }

    __syncthreads();
    num_recv_tokens = shared_num_recv_tokens[warp_group_id];
    recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

    // Copy received tokens to packed output
    for (int i = sub_warp_id; i < num_recv_tokens; i += kNumWarpsPerGroup) {
      auto* src_src_idx =
          reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
      if (lane_id == 0) {
        recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
      }
      syncwarp();

      // Copy data
      auto* src_data = reinterpret_cast<int4*>(
          reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
      auto* dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
      MOE_EP_UNROLLED_WARP_COPY(
          8,
          lane_id,
          static_cast<int>(hidden_int4),
          dst_data,
          src_data,
          ld_nc_global,
          st_na_global);
    }
  }
}

} // namespace

void low_latency_dispatch(
    void* packed_recv_x,
    float* packed_recv_x_scales,
    int* packed_recv_src_info,
    std::int64_t* packed_recv_layout_range,
    int* packed_recv_count,
    int* global_atomic_counter,
    void* rdma_recv_x,
    std::int64_t* rdma_recv_count,
    void* rdma_x,
    const void* x,
    const std::int64_t* topk_idx,
    int* atomic_counter_per_expert,
    int* atomic_finish_counter_per_expert,
    std::int64_t* next_clean,
    int num_next_clean_int,
    int num_tokens,
    int hidden,
    int num_max_dispatch_tokens_per_rank,
    int num_topk,
    int num_experts,
    int rank,
    int num_ranks,
    bool use_fp8,
    bool round_scale,
    bool use_ue8m0,
    void** buffer_ptrs,
    comms::prims::MultipeerIbgdaDeviceTransport* device_transport,
    const comms::prims::IbgdaLocalBuffer* local_rdma_x_buf,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_x,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_count,
    int phase,
    cudaStream_t stream) {
  if (use_fp8) {
    return;
  }

#ifdef __HIP_PLATFORM_AMD__
  constexpr int kNumWarpsPerGroup = 8;
  constexpr int kNumWarpGroups = 2;
#else
  constexpr int kNumWarpsPerGroup = 4;
  constexpr int kNumWarpGroups = 2;
#endif
  const int num_warps = kNumWarpGroups * kNumWarpsPerGroup;
  const int num_sms = cell_div(num_experts, kNumWarpGroups);

  SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);

#define LL_DISPATCH_LAUNCH(H)                                     \
  LAUNCH_KERNEL_NON_COOPERATIVE(                                  \
      &cfg,                                                       \
      (ll_dispatch_kernel<kNumWarpGroups, kNumWarpsPerGroup, H>), \
      packed_recv_x,                                              \
      packed_recv_x_scales,                                       \
      packed_recv_src_info,                                       \
      packed_recv_layout_range,                                   \
      packed_recv_count,                                          \
      global_atomic_counter,                                      \
      rdma_recv_x,                                                \
      rdma_recv_count,                                            \
      rdma_x,                                                     \
      x,                                                          \
      topk_idx,                                                   \
      atomic_counter_per_expert,                                  \
      atomic_finish_counter_per_expert,                           \
      next_clean,                                                 \
      num_next_clean_int,                                         \
      num_tokens,                                                 \
      num_max_dispatch_tokens_per_rank,                           \
      num_topk,                                                   \
      num_experts,                                                \
      rank,                                                       \
      num_ranks,                                                  \
      buffer_ptrs,                                                \
      device_transport,                                           \
      local_rdma_x_buf,                                           \
      peer_remote_recv_x,                                         \
      peer_remote_recv_count,                                     \
      phase)

  SWITCH_HIDDEN(LL_DISPATCH_LAUNCH);
#undef LL_DISPATCH_LAUNCH
}

} // namespace comms::prims::moe_ep::kernels
