/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_allreduce_kernels.h"

/**
 * GPU kernel for incremental reduction: output[i] += input[i]
 * Used to add received chunks directly into the buffer.
 */
template <typename T>
__global__ void incrementalAddKernel(T* output, const T* input, size_t count) {
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    output[elemIdx] += input[elemIdx];
  }
}

template <typename T>
void launchIncrementalAddKernel(
    void* output,
    const void* input,
    size_t count,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  incrementalAddKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(output), static_cast<const T*>(input), count);
}

/**
 * GPU kernel for scaling: output[i] = output[i] / divisor
 * Used to compute average after sum reduction (for ncclAvg operation).
 */
template <typename T>
__global__ void scaleKernel(T* data, size_t count, int divisor) {
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    data[elemIdx] = data[elemIdx] / static_cast<T>(divisor);
  }
}

template <typename T>
void launchScaleKernel(
    void* data,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  scaleKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(data), count, divisor);
}

/**
 * GPU kernel for fused incremental add + scale:
 *   output[i] = (output[i] + input[i]) / divisor
 *
 * Combines DISPATCH_INCREMENTAL_ADD + DISPATCH_SCALE into a single HBM pass
 * (read output, read input, write output once instead of twice).  Used by
 * the active rank to merge passthrough relay scratch into recvbuff while
 * applying the AVG divisor in one fused kernel.
 *
 * When divisor == 1, this collapses to a plain incremental add — but the
 * caller should prefer DISPATCH_INCREMENTAL_ADD in that case to avoid the
 * unnecessary divide.
 */
template <typename T>
__global__ void incrementalAddAndScaleKernel(
    T* output,
    const T* input,
    size_t count,
    int divisor) {
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    T sum = output[elemIdx] + input[elemIdx];
    if (divisor > 1) {
      output[elemIdx] = sum / static_cast<T>(divisor);
    } else {
      output[elemIdx] = sum;
    }
  }
}

template <typename T>
void launchIncrementalAddAndScaleKernel(
    void* output,
    const void* input,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  incrementalAddAndScaleKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(output), static_cast<const T*>(input), count, divisor);
}

/**
 * GPU kernel for fused reduction: output[i] = (a[i] + b[i]) / divisor
 * When divisor == 1, this is a simple sum: output[i] = a[i] + b[i]
 * When divisor == 2, this computes the average: output[i] = (a[i] + b[i]) / 2
 *
 * Used by helper ranks to combine data from both active ranks and compute
 * sum or average in a single kernel launch (avoiding separate add + scale).
 */
template <typename T>
__global__ void fusedReduceKernel(
    T* output,
    const T* inputA,
    const T* inputB,
    size_t count,
    int divisor) {
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    T sum = inputA[elemIdx] + inputB[elemIdx];
    if (divisor > 1) {
      output[elemIdx] = sum / static_cast<T>(divisor);
    } else {
      output[elemIdx] = sum;
    }
  }
}

template <typename T>
void launchFusedReduceKernel(
    void* output,
    const void* inputA,
    const void* inputB,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  fusedReduceKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(output),
      static_cast<const T*>(inputA),
      static_cast<const T*>(inputB),
      count,
      divisor);
}

// Explicit template instantiations for every dtype used by the DISPATCH_*
// macros in sharded_relay_allreduce.cc.  Without these, the symbols would
// not exist with external linkage in the device object, and the host TU
// (which only sees `extern template` declarations via the header) would
// fail to link the launchers — leaving the underlying `__global__`
// kernels' host stubs unresolved at runtime.
#define RCCLX_INSTANTIATE_RELAY_KERNELS(T)                                 \
  template void launchIncrementalAddKernel<T>(                             \
      void* output, const void* input, size_t count, cudaStream_t stream); \
  template void launchScaleKernel<T>(                                      \
      void* data, size_t count, int divisor, cudaStream_t stream);         \
  template void launchIncrementalAddAndScaleKernel<T>(                     \
      void* output,                                                        \
      const void* input,                                                   \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);                                                \
  template void launchFusedReduceKernel<T>(                                \
      void* output,                                                        \
      const void* inputA,                                                  \
      const void* inputB,                                                  \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);

RCCLX_INSTANTIATE_RELAY_KERNELS(int8_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(uint8_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(int32_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(uint32_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(int64_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(uint64_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(__half)
RCCLX_INSTANTIATE_RELAY_KERNELS(float)
RCCLX_INSTANTIATE_RELAY_KERNELS(double)
RCCLX_INSTANTIATE_RELAY_KERNELS(__nv_bfloat16)

#undef RCCLX_INSTANTIATE_RELAY_KERNELS
