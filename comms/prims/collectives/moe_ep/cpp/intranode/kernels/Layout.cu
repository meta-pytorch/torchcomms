// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// (~150 LOC, no transport involved). Pure compute kernel that derives
// per-rank / per-expert token counts from `topk_idx`. Used as the first
// step of every dispatch operation.

// HipHostCompat aliases `__trap()` -> `abort()` in HIP device-compile pass
// (HIP doesn't expose `__trap()` in host pass like nvcc does). Pulled in
// before the kernel that uses MOE_EP_DEVICE_ASSERT.
#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/HipHostCompat.h" // @manual
#endif

#include "comms/prims/collectives/moe_ep/cpp/intranode/kernels/Layout.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/Config.h"

#include <algorithm>
#include <stdexcept>

namespace comms::prims::moe_ep::kernels {

namespace {

// Trap on bad inputs from the kernel. `__trap()` is only valid in the
// device-compile pass (CUDA: `__CUDA_ARCH__`; HIP: `__HIP_DEVICE_COMPILE__`).
// In host pass we still emit a no-op so the kernel signature compiles.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define MOE_EP_DEVICE_ASSERT(cond) \
  do {                             \
    if (!(cond)) {                 \
      __trap();                    \
    }                              \
  } while (0)
#else
#define MOE_EP_DEVICE_ASSERT(cond) ((void)0)
#endif

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void getDispatchLayoutKernel(
    const topk_idx_t* __restrict__ topk_idx,
    int* __restrict__ num_tokens_per_rank,
    int* __restrict__ num_tokens_per_rdma_rank,
    int* __restrict__ num_tokens_per_expert,
    bool* __restrict__ is_token_in_rank,
    int num_tokens,
    int num_topk,
    int num_ranks,
    int num_experts) {
  const int sm_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);

  // First (num_experts / kNumExpertsPerSM) SMs handle the per-expert count.
  __shared__ int per_expert_partial[kNumThreads][kNumExpertsPerSM];
  const int expert_begin_idx = sm_id * kNumExpertsPerSM;
  const int expert_end_idx =
      min(expert_begin_idx + kNumExpertsPerSM, num_experts);

  if (expert_begin_idx < expert_end_idx) {
#pragma unroll
    for (int i = 0; i < kNumExpertsPerSM; ++i) {
      per_expert_partial[thread_id][i] = 0;
    }
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      const auto* shifted = topk_idx + i * num_topk;
#pragma unroll 4
      for (int j = 0; j < num_topk; ++j) {
        const int expert_idx = static_cast<int>(shifted[j]);
        if (expert_idx >= expert_begin_idx && expert_idx < expert_end_idx) {
          ++per_expert_partial[thread_id][expert_idx - expert_begin_idx];
        }
      }
    }
    __syncthreads();

    static_assert(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
    if (expert_begin_idx + thread_id < expert_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) {
        sum += per_expert_partial[i][thread_id];
      }
      num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
    }
    return;
  }

  if (num_tokens_per_rdma_rank != nullptr) {
    MOE_EP_DEVICE_ASSERT(
        num_ranks % NUM_MAX_NVL_PEERS == 0 && num_ranks > NUM_MAX_NVL_PEERS);
  }

  // Remaining SMs handle per-rank + per-RDMA-rank counts.
  constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
  __shared__ int per_rank_partial[kNumThreads][kNumRanksPerSM];
  __shared__ int per_rdma_partial[kNumThreads][kNumRDMARanksPerSM];

  const int sm_begin_for_rank =
      (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
  const int rank_begin_idx = (sm_id - sm_begin_for_rank) * kNumRanksPerSM;
  const int rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
  const int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS;
  const int rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;

  if (rank_begin_idx < rank_end_idx) {
    const int num_expert_per_rank = num_experts / num_ranks;
    const int expert_begin = rank_begin_idx * num_expert_per_rank;
    const int expert_end = rank_end_idx * num_expert_per_rank;

#pragma unroll
    for (int i = 0; i < kNumRanksPerSM; ++i) {
      per_rank_partial[thread_id][i] = 0;
    }
#pragma unroll
    for (int i = 0; i < kNumRDMARanksPerSM; ++i) {
      per_rdma_partial[thread_id][i] = 0;
    }

    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      const auto* shifted = topk_idx + i * num_topk;
      int is_in_rank[kNumRanksPerSM] = {0};
      int is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll 4
      for (int j = 0; j < num_topk; ++j) {
        const int expert_idx = static_cast<int>(shifted[j]);
        if (expert_idx >= expert_begin && expert_idx < expert_end) {
          const int rank_idx =
              expert_idx / num_expert_per_rank - rank_begin_idx;
          is_in_rank[rank_idx]++;
          is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
        }
      }

      bool* shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
#pragma unroll 4
      for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
        shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
        per_rank_partial[thread_id][j] += (is_in_rank[j] > 0);
      }
#pragma unroll 4
      for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j) {
        per_rdma_partial[thread_id][j] += (is_in_rdma_rank[j] > 0);
      }
    }
    __syncthreads();

    static_assert(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
    if (rank_begin_idx + thread_id < rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) {
        sum += per_rank_partial[i][thread_id];
      }
      num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
    }

    if (num_tokens_per_rdma_rank != nullptr &&
        rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) {
        sum += per_rdma_partial[i][thread_id];
      }
      num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
    }
  }
}

} // namespace

void get_dispatch_layout(
    const topk_idx_t* topk_idx,
    int* num_tokens_per_rank,
    int* num_tokens_per_rdma_rank,
    int* num_tokens_per_expert,
    bool* is_token_in_rank,
    int num_tokens,
    int num_topk,
    int num_ranks,
    int num_experts,
    cudaStream_t stream) {
  constexpr int kNumThreads = 256;
  constexpr int kNumExpertsPerSM = 4;
  constexpr int kNumRanksPerSM = 8;
  static_assert(
      kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0,
      "kNumRanksPerSM must be divisible by NUM_MAX_NVL_PEERS");

  const int num_sms =
      ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
      (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;

  void* args[] = {
      reinterpret_cast<void*>(const_cast<topk_idx_t**>(&topk_idx)),
      reinterpret_cast<void*>(&num_tokens_per_rank),
      reinterpret_cast<void*>(&num_tokens_per_rdma_rank),
      reinterpret_cast<void*>(&num_tokens_per_expert),
      reinterpret_cast<void*>(&is_token_in_rank),
      reinterpret_cast<void*>(&num_tokens),
      reinterpret_cast<void*>(&num_topk),
      reinterpret_cast<void*>(&num_ranks),
      reinterpret_cast<void*>(&num_experts),
  };
  auto err = cudaLaunchKernel(
      reinterpret_cast<const void*>(&getDispatchLayoutKernel<
                                    kNumThreads,
                                    kNumExpertsPerSM,
                                    kNumRanksPerSM>),
      dim3(num_sms),
      dim3(kNumThreads),
      args,
      0,
      stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("get_dispatch_layout: cudaLaunchKernel failed: ") +
        cudaGetErrorString(err));
  }
}

} // namespace comms::prims::moe_ep::kernels
