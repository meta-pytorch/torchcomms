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
 * Notify pre-dispatch — exchanges per-rank counts via NVL transport and
 * stages the channel head/tail/queue offsets that subsequent dispatch
 * reads.
 *
 * Real implementation lands in a follow-up sub-commit alongside
 * IntranodeDispatch.cu — this header keeps the BUCK target buildable.
 */
void notify_dispatch(
    const int* num_tokens_per_rank,
    int* moe_recv_counter_mapped,
    int num_ranks,
    const int* num_tokens_per_expert,
    int* moe_recv_expert_counter_mapped,
    int num_experts,
    int num_tokens,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix_copy,
    int num_memset_int,
    int expert_alignment,
    void** buffer_ptrs,
    int** task_fifo_ptrs,
    int head,
    int rank,
    cudaStream_t stream,
    int num_channels);

/**
 * `cached_notify_dispatch` — handle-reuse path. Called when the user passes
 * a `handle` from a prior dispatch back into the next one; skips the count
 * exchange and just refreshes the local prefix-matrix region + zeroes the
 * channel state.
 */
void cached_notify_dispatch(
    const int* rank_prefix_matrix,
    int num_memset_int,
    void** buffer_ptrs,
    int** task_fifo_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int head = 0);

/**
 * `cached_notify_combine` — symmetric to cached_notify_dispatch. Barrier +
 * memset + per-channel `send_head` placeholder fill so combine sees
 * already-routed token slots in the right shape.
 */
void cached_notify_combine(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_recv_tokens,
    int num_memset_int,
    int** task_fifo_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int head = 0);

} // namespace comms::prims::moe_ep::kernels
