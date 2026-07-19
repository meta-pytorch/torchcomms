// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace comms::prims::moe_ep::kernels {

/**
 * Intranode dispatch — chunked NVLink pipelined send across ≤
 * NUM_MAX_NVL_PEERS peers.
 *
 * Per-rank workspace layout — see the comment block at the top of
 * IntranodeDispatch.cu.
 *
 * `topk_idx` / `recv_topk_idx` are int64 (`topk_idx_t`).
 */
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
    int num_scales,
    int scale_token_stride,
    int scale_hidden_stride,
    void** buffer_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens);

} // namespace comms::prims::moe_ep::kernels
