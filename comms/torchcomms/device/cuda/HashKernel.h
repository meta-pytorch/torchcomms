// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <cstdint>

namespace torch::comms {

struct HashEntry {
  uint64_t user_context;
  uint64_t hash;
};

cudaError_t launchHash(
    cudaStream_t stream,
    const void* buf,
    size_t len_bytes,
    HashEntry* entries,
    size_t max_entries,
    uint64_t* unfilled,
    uint64_t user_context,
    int num_blocks,
    int threads_per_block);

} // namespace torch::comms
