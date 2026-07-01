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
 * Intranode combine — reduces per-rank dispatched tokens back to original
 * source ranks.
 *
 * Real kernel + launch wrapper in a follow-up sub-commit.
 */
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
    int num_recv_buffer_tokens);

} // namespace comms::prims::moe_ep::kernels
