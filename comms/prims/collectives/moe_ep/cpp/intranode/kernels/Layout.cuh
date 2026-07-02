// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime.h>
#endif

namespace comms::prims::moe_ep::kernels {

// Top-k index type (TOPK_IDX_BITS=64).
using topk_idx_t = int64_t;

/**
 * get_dispatch_layout — pure compute kernel that derives per-rank /
 * per-expert / per-RDMA-rank token counts from `topk_idx`.
 *
 * No transport involvement; pure local compute.
 *
 * @param topk_idx                Per-token expert indices, [num_tokens,
 * num_topk].
 * @param num_tokens_per_rank     Output [num_ranks].
 * @param num_tokens_per_rdma_rank Output [num_rdma_ranks] — may be `nullptr`
 *                                 in intranode-only mode.
 * @param num_tokens_per_expert   Output [num_experts].
 * @param is_token_in_rank        Output [num_tokens, num_ranks] bool.
 * @param num_tokens              Number of input tokens.
 * @param num_topk                Top-k count per token.
 * @param num_ranks               Total ranks (== num_nvl_peers in intranode).
 * @param num_experts             Total experts (must be divisible by
 * num_ranks).
 * @param stream                  CUDA / HIP stream.
 */
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
    cudaStream_t stream);

} // namespace comms::prims::moe_ep::kernels
