// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Shared CUDA device helpers for iterated device API tests.
// These are __device__ functions that can be called from both NCCLx and Pipes
// test kernels.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace torchcomms::device::test {

// Fill a float buffer with a pattern that encodes rank and iteration.
// Pattern value: (rank + 1) * 1000.0f + iteration
// Must match fillPatternValue() in IteratedTestHelpers.hpp.
__device__ inline void
fillPattern(float* buf, size_t count, int rank, int iteration) {
  float val = static_cast<float>((rank + 1) * 1000 + iteration);
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    buf[i] = val;
  }
  __syncthreads();
}

// Verify a float buffer matches the expected fill pattern.
// Writes 1 to *result if verification passes, 0 if any element mismatches.
// Only thread 0 writes the result, but all threads participate in checking.
__device__ inline void verifyPattern(
    const float* buf,
    size_t count,
    int expected_rank,
    int expected_iteration,
    int* result) {
  float expected_val =
      static_cast<float>((expected_rank + 1) * 1000 + expected_iteration);

  // Use shared memory for reduction
  __shared__ int any_mismatch;
  if (threadIdx.x == 0) {
    any_mismatch = 0;
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    float diff = buf[i] - expected_val;
    if (diff > 1e-3f || diff < -1e-3f) {
      atomicExch(&any_mismatch, 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    *result = (any_mismatch == 0) ? 1 : 0;
  }
}

} // namespace torchcomms::device::test
