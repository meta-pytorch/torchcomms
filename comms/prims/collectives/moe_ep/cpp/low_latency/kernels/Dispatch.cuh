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
 * Low-latency dispatch — RDMA-direct send of FP8/bf16 quantized tokens to
 * peer's per-(local-expert × src-rank) recv buffer.
 */
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
    int phase,
    cudaStream_t stream);

} // namespace comms::prims::moe_ep::kernels
