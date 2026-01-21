// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/dispatch.h"

namespace comms::pipes::test {

// Wrapper to call dispatch from test code
// This ensures proper CUDA compilation for types used in the test
void testDispatch(
    // Outputs
    DeviceSpan<void* const> recvbuffs,
    DeviceSpan<std::size_t> output_chunk_sizes_per_rank,
    // Inputs
    DeviceSpan<Transport> transports,
    int my_rank,
    const void* sendbuff_d,
    DeviceSpan<const std::size_t> input_chunk_sizes,
    const std::size_t* input_chunk_indices_d,
    DeviceSpan<const std::size_t> input_chunk_indices_count_per_rank,
    cudaStream_t stream,
    int num_blocks,
    int num_threads);

} // namespace comms::pipes::test
