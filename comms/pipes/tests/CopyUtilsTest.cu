// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

using namespace comms::pipes;

__global__ void testCopyChunkVectorizedKernel(
    char* dst_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d) {
  auto warp = make_warp_group();

  memcpy_vectorized(dst_d, src_d, chunk_bytes, warp);

  __syncthreads();

  if (warp.is_global_leader()) {
    for (std::size_t i = 0; i < chunk_bytes; i++) {
      if (dst_d[i] != src_d[i]) {
        atomicAdd(errorCount_d, 1);
      }
    }
  }
}

void testCopyChunkVectorized(
    char* dst_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testCopyChunkVectorizedKernel<<<numBlocks, blockSize>>>(
      dst_d, src_d, chunk_bytes, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

__global__ void testCopyChunkVectorizedDualDestKernel(
    char* dst1_d,
    char* dst2_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d) {
  auto warp = make_warp_group();

  std::array<char*, 2> dsts = {{dst1_d, dst2_d}};
  memcpy_vectorized_multi_dest<2>(dsts, src_d, chunk_bytes, warp);

  __syncthreads();

  if (warp.is_global_leader()) {
    for (std::size_t i = 0; i < chunk_bytes; i++) {
      if (dst1_d[i] != src_d[i]) {
        atomicAdd(errorCount_d, 1);
      }
      if (dst2_d[i] != src_d[i]) {
        atomicAdd(errorCount_d, 1);
      }
    }
  }
}

void testCopyChunkVectorizedDualDest(
    char* dst1_d,
    char* dst2_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testCopyChunkVectorizedDualDestKernel<<<numBlocks, blockSize>>>(
      dst1_d, dst2_d, src_d, chunk_bytes, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

__global__ void testCopyChunkVectorizedMultiDest1Kernel(
    char* dst_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d) {
  auto warp = make_warp_group();

  std::array<char*, 1> dsts = {{dst_d}};
  memcpy_vectorized_multi_dest<1>(dsts, src_d, chunk_bytes, warp);

  __syncthreads();

  if (warp.is_global_leader()) {
    for (std::size_t i = 0; i < chunk_bytes; i++) {
      if (dst_d[i] != src_d[i]) {
        atomicAdd(errorCount_d, 1);
      }
    }
  }
}

void testCopyChunkVectorizedMultiDest1(
    char* dst_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testCopyChunkVectorizedMultiDest1Kernel<<<numBlocks, blockSize>>>(
      dst_d, src_d, chunk_bytes, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

__global__ void testCopyChunkVectorizedMultiDest3Kernel(
    char* dst1_d,
    char* dst2_d,
    char* dst3_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d) {
  auto warp = make_warp_group();

  std::array<char*, 3> dsts = {{dst1_d, dst2_d, dst3_d}};
  memcpy_vectorized_multi_dest<3>(dsts, src_d, chunk_bytes, warp);

  __syncthreads();

  if (warp.is_global_leader()) {
    for (std::size_t i = 0; i < chunk_bytes; i++) {
      if (dst1_d[i] != src_d[i]) {
        atomicAdd(errorCount_d, 1);
      }
      if (dst2_d[i] != src_d[i]) {
        atomicAdd(errorCount_d, 1);
      }
      if (dst3_d[i] != src_d[i]) {
        atomicAdd(errorCount_d, 1);
      }
    }
  }
}

void testCopyChunkVectorizedMultiDest3(
    char* dst1_d,
    char* dst2_d,
    char* dst3_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testCopyChunkVectorizedMultiDest3Kernel<<<numBlocks, blockSize>>>(
      dst1_d, dst2_d, dst3_d, src_d, chunk_bytes, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
