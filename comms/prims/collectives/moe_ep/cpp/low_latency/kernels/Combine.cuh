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
 * Low-latency combine — return post-MoE results to source ranks via NVLink
 * peer-mapped IPC buffers.
 */
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
    int phase,
    bool zero_copy,
    cudaStream_t stream);

} // namespace comms::prims::moe_ep::kernels
