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
 * `clean_low_latency_buffer` — wipes the low-latency send/recv counters back
 * to 0 and barriers across all ranks.
 *
 * Works alongside the `LowLatencyRuntime` that owns the team-barrier
 * signal slots.
 */
void clean_low_latency_buffer(
    std::int64_t* clean_0,
    int num_clean_int_0,
    std::int64_t* clean_1,
    int num_clean_int_1,
    int rank,
    int num_ranks,
    int* mask_buffer_ptr,
    int* sync_buffer_ptr,
    cudaStream_t stream);

} // namespace comms::prims::moe_ep::kernels
