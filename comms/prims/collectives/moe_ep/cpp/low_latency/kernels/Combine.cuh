// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace comms::prims {
struct MultipeerIbgdaDeviceTransport;
struct IbgdaRemoteBuffer;
struct IbgdaLocalBuffer;
} // namespace comms::prims

namespace comms::prims::moe_ep::kernels {

/**
 * Low-latency combine — return post-MoE results to source ranks via NVLink
 * peer-mapped IPC buffers (same-node) or IBGDA RDMA (cross-node).
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
    // IBGDA cross-node hybrid path. May be nullptr (single-node mode).
    // local_x_buf: local x tensor as registered IbgdaLocalBuffer.
    // peer_remote_recv_x: each peer's combine_recv_x descriptor.
    // peer_remote_recv_flag: each peer's combine_recv_flag descriptor.
    comms::prims::MultipeerIbgdaDeviceTransport* device_transport,
    const comms::prims::IbgdaLocalBuffer* local_x_buf,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_x,
    const comms::prims::IbgdaRemoteBuffer* peer_remote_recv_flag,
    int phase,
    bool zero_copy,
    cudaStream_t stream);

} // namespace comms::prims::moe_ep::kernels
