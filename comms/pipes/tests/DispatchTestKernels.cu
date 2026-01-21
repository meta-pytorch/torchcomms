// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/tests/DispatchTestKernels.cuh"

#include "comms/pipes/collectives/dispatch.h"

namespace comms::pipes::test {

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
    int num_threads) {
  collectives::dispatch(
      recvbuffs,
      output_chunk_sizes_per_rank,
      transports,
      my_rank,
      sendbuff_d,
      input_chunk_sizes,
      input_chunk_indices_d,
      input_chunk_indices_count_per_rank,
      stream,
      num_blocks,
      num_threads);
}

} // namespace comms::pipes::test
