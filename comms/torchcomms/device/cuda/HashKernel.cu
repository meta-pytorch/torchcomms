// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/torchcomms/device/cuda/HashKernel.h"

#include <cstdint>

namespace torch::comms {

__device__ __forceinline__ uint64_t
hashBuffer(const void* buf, size_t len_bytes) {
  const auto* data = static_cast<const uint8_t*>(buf);
  size_t num_words = len_bytes / 8;
  size_t tail_bytes = len_bytes % 8;

  auto tid = threadIdx.x;
  auto stride = blockDim.x;

  uint64_t local_hash = 0;
  const auto* words = reinterpret_cast<const uint64_t*>(data);
  for (size_t i = tid; i < num_words; i += stride) {
    local_hash ^= words[i];
  }

  if (tid == 0 && tail_bytes > 0) {
    uint64_t tail = 0;
    const uint8_t* tail_ptr = data + num_words * 8;
    for (size_t i = 0; i < tail_bytes; ++i) {
      tail |= static_cast<uint64_t>(tail_ptr[i]) << (i * 8);
    }
    local_hash ^= tail;
  }

  return local_hash;
}

__device__ __forceinline__ uint64_t
reduceShared(uint64_t local_hash, uint64_t* shared_hash) {
  shared_hash[threadIdx.x] = local_hash;
  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared_hash[threadIdx.x] ^= shared_hash[threadIdx.x + s];
    }
    __syncthreads();
  }

  return shared_hash[0];
}

__global__ void hashKernel(
    const void* buf,
    size_t len_bytes,
    HashEntry* entries,
    size_t max_entries,
    uint64_t* unfilled,
    uint64_t user_context) {
  extern __shared__ uint64_t shared_hash[];

  uint64_t local_hash = hashBuffer(buf, len_bytes);
  uint64_t block_hash = reduceShared(local_hash, shared_hash);

  if (threadIdx.x == 0) {
    size_t slot = *unfilled;
    HashEntry* entry = &entries[slot % max_entries];
    entry->user_context = user_context;
    entry->hash = block_hash;
    __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned long long*>(unfilled), 1ULL);
  }
}

__attribute__((weak)) cudaError_t launchHash(
    cudaStream_t stream,
    const void* buf,
    size_t len_bytes,
    HashEntry* entries,
    size_t max_entries,
    uint64_t* unfilled,
    uint64_t user_context,
    int /* num_blocks */,
    int threads_per_block) {
  size_t shared_mem = threads_per_block * sizeof(uint64_t);
  hashKernel<<<1, threads_per_block, shared_mem, stream>>>(
      buf, len_bytes, entries, max_entries, unfilled, user_context);
  return cudaGetLastError();
}

} // namespace torch::comms
