/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "low_precision_utility.h"

/**
 * GPU kernel for vectorized float-to-FP8 conversion with optimized memory
 * access patterns. Uses multi-level vectorization (8-element, 4-element,
 * scalar) for maximum throughput.
 */
template <typename T>
__global__ void quantizeFloatToFp8Kernel(
    const T* input,
    void* output,
    size_t totalCount,
    size_t chunkStart,
    size_t chunkSize) {
  rccl_float8* fp8_output = static_cast<rccl_float8*>(output);

  // Calculate thread indexing for parallel processing
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  // Use 8-element vectorization for maximum throughput
  constexpr size_t VECTOR_SIZE = 8;
  const size_t vectorizedCount = (chunkSize / VECTOR_SIZE) * VECTOR_SIZE;

  // Main vectorized loop: process 8 elements per thread for optimal throughput
#pragma unroll 8
  for (size_t localIdx = threadId * VECTOR_SIZE;
       localIdx + VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    // Branch prediction optimization for common case
    if (__builtin_expect(
            globalIdx + VECTOR_SIZE <= totalCount &&
                localIdx + VECTOR_SIZE <= chunkSize,
            1)) {
      // Prefetch future data to hide memory latency
      if (localIdx + totalThreads * VECTOR_SIZE * 4 <= chunkSize) {
        __builtin_prefetch(
            &input[globalIdx + totalThreads * VECTOR_SIZE * 4], 0, 1);
      }

      // Use aligned access for better memory bandwidth
      const float* aligned_input = reinterpret_cast<const float*>(
          __builtin_assume_aligned(&input[globalIdx], 64));
      quantize_float8_to_fp8_batch(aligned_input, &fp8_output[globalIdx]);
    }
  }

  // Fallback loop: handle remaining elements with 4-element vectorization
  constexpr size_t FALLBACK_VECTOR_SIZE = 4;
  const size_t fallbackCount =
      ((chunkSize - vectorizedCount) / FALLBACK_VECTOR_SIZE) *
          FALLBACK_VECTOR_SIZE +
      vectorizedCount;

#pragma unroll 2
  for (size_t localIdx = vectorizedCount + threadId * FALLBACK_VECTOR_SIZE;
       localIdx + FALLBACK_VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * FALLBACK_VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(
            globalIdx + FALLBACK_VECTOR_SIZE <= totalCount &&
                localIdx + FALLBACK_VECTOR_SIZE <= chunkSize,
            1)) {
      // Use cache load for better memory efficiency
      float4 input_vec =
          __ldg(reinterpret_cast<const float4*>(&input[globalIdx]));

      quantize_float4_to_fp8_batch(input_vec, &fp8_output[globalIdx]);
    }
  }

  // Scalar loop: handle remaining individual elements
  for (size_t localIdx = fallbackCount + threadId; localIdx < chunkSize;
       localIdx += totalThreads) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(globalIdx < totalCount && localIdx < chunkSize, 1)) {
      fp8_output[globalIdx] = quantize_float_to_fp8_e4m3(input[globalIdx]);
    }
  }
}

/**
 * GPU kernel for high-performance BFloat16-to-FP8 conversion.
 * Takes hip_bfloat16 input array and directly quantizes to FP8 output (1:1
 * mapping). Uses optimized vectorized operations for maximum throughput.
 */
template <typename T>
__global__ void quantizeBF16ToFp8Kernel(
    const T* bf16Input,
    void* fp8Output,
    size_t totalOutputCount,
    size_t chunkStart,
    size_t chunkSize) {
  rccl_float8* fp8_output = static_cast<rccl_float8*>(fp8Output);

  // Calculate thread indexing for parallel processing
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  // Use 8-element vectorization for maximum throughput (8 BF16 -> 8 FP8)
  constexpr size_t VECTOR_SIZE = 8;
  const size_t vectorizedCount = (chunkSize / VECTOR_SIZE) * VECTOR_SIZE;

  // Main vectorized loop: process 8 bfloat16 -> 8 FP8 elements
#pragma unroll 8
  for (size_t localIdx = threadId * VECTOR_SIZE;
       localIdx + VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    // Branch prediction optimization for common case
    if (__builtin_expect(
            globalIdx + VECTOR_SIZE <= totalOutputCount &&
                localIdx + VECTOR_SIZE <= chunkSize,
            1)) {
      // Prefetch future data to hide memory latency
      if (localIdx + totalThreads * VECTOR_SIZE * 4 <= chunkSize) {
        __builtin_prefetch(
            &bf16Input[globalIdx + totalThreads * VECTOR_SIZE * 4], 0, 1);
      }

      // Use aligned access for better memory bandwidth
      const T* aligned_input = reinterpret_cast<const T*>(
          __builtin_assume_aligned(&bf16Input[globalIdx], 64));

      // Convert 8 bfloat16 values to float and quantize to FP8
      float temp_floats[8];
#pragma unroll 8
      for (int i = 0; i < 8; ++i) {
        // Convert bfloat16 to float by shifting and padding
        uint32_t bf16_as_uint =
            *reinterpret_cast<const uint16_t*>(&aligned_input[i]);
        temp_floats[i] = __uint_as_float(bf16_as_uint << 16);
      }

      // Quantize 8 floats to 8 FP8 values
      quantize_float8_to_fp8_batch(temp_floats, &fp8_output[globalIdx]);
    }
  }

  // Fallback loop: handle remaining elements with 4-element vectorization
  // (4 BF16 -> 4 FP8)
  constexpr size_t FALLBACK_VECTOR_SIZE = 4;
  const size_t fallbackCount =
      ((chunkSize - vectorizedCount) / FALLBACK_VECTOR_SIZE) *
          FALLBACK_VECTOR_SIZE +
      vectorizedCount;

#pragma unroll 2
  for (size_t localIdx = vectorizedCount + threadId * FALLBACK_VECTOR_SIZE;
       localIdx + FALLBACK_VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * FALLBACK_VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(
            globalIdx + FALLBACK_VECTOR_SIZE <= totalOutputCount &&
                localIdx + FALLBACK_VECTOR_SIZE <= chunkSize,
            1)) {
      // Convert 4 bfloat16 values to float and quantize to FP8
      float temp_floats[4];
#pragma unroll 4
      for (int i = 0; i < 4; ++i) {
        // Convert bfloat16 to float by shifting and padding
        uint32_t bf16_as_uint =
            *reinterpret_cast<const uint16_t*>(&bf16Input[globalIdx + i]);
        temp_floats[i] = __uint_as_float(bf16_as_uint << 16);
      }

      // Use float4 for vectorized quantization
      float4 temp_vec = {
          temp_floats[0], temp_floats[1], temp_floats[2], temp_floats[3]};
      quantize_float4_to_fp8_batch(temp_vec, &fp8_output[globalIdx]);
    }
  }

  // Scalar loop: handle remaining individual bfloat16 elements (1:1 mapping)
  for (size_t localIdx = fallbackCount + threadId; localIdx < chunkSize;
       localIdx += totalThreads) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(
            globalIdx < totalOutputCount && localIdx < chunkSize, 1)) {
      // Convert single bfloat16 to float and quantize to FP8
      uint32_t bf16_as_uint =
          *reinterpret_cast<const uint16_t*>(&bf16Input[globalIdx]);
      float temp_float = __uint_as_float(bf16_as_uint << 16);

      // Quantize to FP8
      fp8_output[globalIdx] = quantize_float_to_fp8_e4m3(temp_float);
    }
  }
}

/**
 * GPU kernel for high-performance float-to-BF16 conversion.
 * Takes count float input and converts to count bfloat16 output with 1:1
 * mapping. Uses optimized vectorized operations for maximum throughput.
 */
template <typename T>
__global__ void dequantizeFloatToBF16Kernel(
    const T* floatInput,
    uint16_t* bf16Output,
    size_t totalFloatCount,
    size_t chunkStart,
    size_t chunkSize) {
  // Calculate thread indexing for parallel processing
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  // Use 8-element vectorization for maximum throughput
  constexpr size_t VECTOR_SIZE = 8;
  const size_t vectorizedCount = (chunkSize / VECTOR_SIZE) * VECTOR_SIZE;

  // Main vectorized loop: process 8 float elements -> 8 bfloat16 elements
#pragma unroll 4
  for (size_t localIdx = threadId * VECTOR_SIZE;
       localIdx + VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    // Branch prediction optimization for common case
    if (__builtin_expect(
            globalIdx + VECTOR_SIZE <= totalFloatCount &&
                localIdx + VECTOR_SIZE <= chunkSize,
            1)) {
      // Prefetch future data to hide memory latency
      if (localIdx + totalThreads * VECTOR_SIZE * 4 <= chunkSize) {
        __builtin_prefetch(
            &floatInput[globalIdx + totalThreads * VECTOR_SIZE * 4], 0, 1);
      }

      // Use aligned access for better memory bandwidth
      const float* aligned_input = reinterpret_cast<const float*>(
          __builtin_assume_aligned(&floatInput[globalIdx], 64));

      // Convert 8 float values directly to 8 bfloat16 values
#pragma unroll 8
      for (int i = 0; i < 8; ++i) {
        // Convert float to bfloat16 by truncating mantissa
        uint16_t bf16_value = __float_as_uint(aligned_input[i]) >> 16;
        bf16Output[globalIdx + i] = bf16_value;
      }
    }
  }

  // Fallback loop: handle remaining elements with 4-element vectorization
  constexpr size_t FALLBACK_VECTOR_SIZE = 4;
  const size_t fallbackCount =
      ((chunkSize - vectorizedCount) / FALLBACK_VECTOR_SIZE) *
          FALLBACK_VECTOR_SIZE +
      vectorizedCount;

#pragma unroll 2
  for (size_t localIdx = vectorizedCount + threadId * FALLBACK_VECTOR_SIZE;
       localIdx + FALLBACK_VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * FALLBACK_VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(
            globalIdx + FALLBACK_VECTOR_SIZE <= totalFloatCount &&
                localIdx + FALLBACK_VECTOR_SIZE <= chunkSize,
            1)) {
      // Convert 4 float values directly to 4 bfloat16 values
#pragma unroll 4
      for (int i = 0; i < 4; ++i) {
        // Convert float to bfloat16 by truncating mantissa
        uint16_t bf16_value = __float_as_uint(floatInput[globalIdx + i]) >> 16;
        bf16Output[globalIdx + i] = bf16_value;
      }
    }
  }

  // Scalar loop: handle remaining elements one by one
  for (size_t localIdx = fallbackCount + threadId; localIdx < chunkSize;
       localIdx += totalThreads) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(
            globalIdx < totalFloatCount && localIdx < chunkSize, 1)) {
      // Process single element
      float temp_float = floatInput[globalIdx];
      uint16_t bf16_value = __float_as_uint(temp_float) >> 16;
      bf16Output[globalIdx] = bf16_value;
    }
  }
}

/**
 * GPU kernel for multi-rank local reduction with FP8 dequantization and
 * prefetching. Performs element-wise reduction across multiple rank
 * contributions with optimized access patterns.
 */
template <typename T>
__global__ void localReductionKernel(
    const void* fp8Input,
    T* floatOutput,
    size_t totalCount,
    size_t chunkStart,
    size_t chunkSize,
    int nRanks,
    int myRank) {
  const rccl_float8* fp8_input = static_cast<const rccl_float8*>(fp8Input);

  // Calculate thread indexing and grid configuration
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;
  size_t uniformChunkSize = chunkSize;

  // Main reduction loop: each thread processes elements with stride across
  // chunk
  for (size_t elemIdx = threadId; elemIdx < uniformChunkSize;
       elemIdx += totalThreads) {
    float accumulator = 0.0f;

    // Inner loop: sum contributions from all ranks for current element
#pragma unroll 8
    for (int contributorRank = 0; contributorRank < nRanks; contributorRank++) {
      size_t sourceOffset =
          static_cast<size_t>(contributorRank) * uniformChunkSize + elemIdx;

      if (__builtin_expect(sourceOffset < totalCount, 1)) {
        // Prefetch next rank's data to improve cache performance
        if (contributorRank + 1 < nRanks) {
          size_t nextOffset =
              static_cast<size_t>(contributorRank + 1) * uniformChunkSize +
              elemIdx;
          __builtin_prefetch(&fp8_input[nextOffset], 0, 3);
        }

        // Dequantize FP8 value and accumulate for final sum
        float fp8Val = dequantize_fp8_e4m3_to_float(fp8_input[sourceOffset]);
        accumulator += fp8Val;
      }
    }

    // Write reduced result to output buffer
    if (__builtin_expect(elemIdx < uniformChunkSize, 1)) {
      floatOutput[elemIdx] = accumulator;
    }
  }
}

/**
 * GPU kernel for high-performance FP8-to-float conversion.
 * Uses multi-level vectorization and improved grid utilization for maximum
 * throughput.
 */
template <typename T>
__global__ void dequantizeFp8ToFloatKernel(
    const void* fp8Input,
    T* floatOutput,
    size_t totalCount,
    size_t chunkStart,
    size_t chunkSize) {
  const rccl_float8* fp8_input = static_cast<const rccl_float8*>(fp8Input);

  // Calculate thread indexing for parallel processing
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  // Use 8-element vectorization for maximum throughput
  constexpr size_t VECTOR_SIZE = 8;
  const size_t vectorizedCount = (chunkSize / VECTOR_SIZE) * VECTOR_SIZE;

  // Main vectorized loop: process 8 elements per thread for optimal throughput
#pragma unroll 8
  for (size_t localIdx = threadId * VECTOR_SIZE;
       localIdx + VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    // Branch prediction optimization for common case
    if (__builtin_expect(
            globalIdx + VECTOR_SIZE <= totalCount &&
                localIdx + VECTOR_SIZE <= chunkSize,
            1)) {
      // Prefetch future data to hide memory latency
      if (localIdx + totalThreads * VECTOR_SIZE * 4 <= chunkSize) {
        __builtin_prefetch(
            &fp8_input[globalIdx + totalThreads * VECTOR_SIZE * 4], 0, 1);
      }

      // Use aligned access for better memory bandwidth
      const rccl_float8* aligned_input = reinterpret_cast<const rccl_float8*>(
          __builtin_assume_aligned(&fp8_input[globalIdx], 64));

      // Batch dequantize 8 FP8 values to 8 float values
      dequantize_fp8_batch_to_float8(aligned_input, &floatOutput[globalIdx]);
    }
  }

  // Fallback loop: handle remaining elements with 4-element vectorization
  constexpr size_t FALLBACK_VECTOR_SIZE = 4;
  const size_t fallbackCount =
      ((chunkSize - vectorizedCount) / FALLBACK_VECTOR_SIZE) *
          FALLBACK_VECTOR_SIZE +
      vectorizedCount;

#pragma unroll 2
  for (size_t localIdx = vectorizedCount + threadId * FALLBACK_VECTOR_SIZE;
       localIdx + FALLBACK_VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * FALLBACK_VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(
            globalIdx + FALLBACK_VECTOR_SIZE <= totalCount &&
                localIdx + FALLBACK_VECTOR_SIZE <= chunkSize,
            1)) {
      // Use cache load for better memory efficiency
      float4 output_vec = dequantize_fp8_batch_to_float4(&fp8_input[globalIdx]);

      *reinterpret_cast<float4*>(&floatOutput[globalIdx]) = output_vec;
    }
  }

  // Scalar loop: handle remaining individual elements
  for (size_t localIdx = fallbackCount + threadId; localIdx < chunkSize;
       localIdx += totalThreads) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(globalIdx < totalCount && localIdx < chunkSize, 1)) {
      floatOutput[globalIdx] =
          dequantize_fp8_e4m3_to_float(fp8_input[globalIdx]);
    }
  }
}

/**
 * GPU kernel for high-performance FP8-to-BFloat16 conversion.
 * Converts FP8 input to bfloat16 output with 1:1 mapping using MI300
 * optimizations and vectorized operations for maximum throughput.
 */
template <typename T>
__global__ void dequantizeFp8ToBF16Kernel(
    const void* fp8Input,
    T* bf16Output,
    size_t totalCount,
    size_t chunkStart,
    size_t chunkSize) {
  const rccl_float8* fp8_input = static_cast<const rccl_float8*>(fp8Input);

  // Calculate thread indexing for parallel processing
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  // Use 8-element vectorization for maximum throughput (8 FP8 -> 8 BF16)
  constexpr size_t VECTOR_SIZE = 8;
  const size_t vectorizedCount = (chunkSize / VECTOR_SIZE) * VECTOR_SIZE;

  // Main vectorized loop: process 8 FP8 elements -> 8 BF16 elements (1:1
  // mapping)
#pragma unroll 8
  for (size_t localIdx = threadId * VECTOR_SIZE;
       localIdx + VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    // Branch prediction optimization for common case
    if (__builtin_expect(
            globalIdx + VECTOR_SIZE <= totalCount &&
                localIdx + VECTOR_SIZE <= chunkSize,
            1)) {
      // Prefetch future data to hide memory latency
      if (localIdx + totalThreads * VECTOR_SIZE * 4 <= chunkSize) {
        __builtin_prefetch(
            &fp8_input[globalIdx + totalThreads * VECTOR_SIZE * 4], 0, 1);
      }

      // Use aligned access for better memory bandwidth
      const rccl_float8* aligned_input = reinterpret_cast<const rccl_float8*>(
          __builtin_assume_aligned(&fp8_input[globalIdx], 64));

      // Batch dequantize 8 FP8 values to 8 float values, then convert to BF16
      float temp_floats[8];
      dequantize_fp8_batch_to_float8(aligned_input, temp_floats);

      // Convert 8 float values directly to 8 BF16 values (1:1 mapping)
#pragma unroll 8
      for (int i = 0; i < 8; ++i) {
        // Convert float to bfloat16 by truncating mantissa
        uint16_t bf16_value = __float_as_uint(temp_floats[i]) >> 16;
        bf16Output[globalIdx + i] = *reinterpret_cast<const T*>(&bf16_value);
      }
    }
  }

  // Fallback loop: handle remaining elements with 4-element vectorization
  // (4 FP8 -> 4 BF16)
  constexpr size_t FALLBACK_VECTOR_SIZE = 4;
  const size_t fallbackCount =
      ((chunkSize - vectorizedCount) / FALLBACK_VECTOR_SIZE) *
          FALLBACK_VECTOR_SIZE +
      vectorizedCount;

#pragma unroll 2
  for (size_t localIdx = vectorizedCount + threadId * FALLBACK_VECTOR_SIZE;
       localIdx + FALLBACK_VECTOR_SIZE <= chunkSize;
       localIdx += totalThreads * FALLBACK_VECTOR_SIZE) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(
            globalIdx + FALLBACK_VECTOR_SIZE <= totalCount &&
                localIdx + FALLBACK_VECTOR_SIZE <= chunkSize,
            1)) {
      // Use cache load for better memory efficiency
      float4 temp_vec = dequantize_fp8_batch_to_float4(&fp8_input[globalIdx]);

      // Convert 4 float values directly to 4 BF16 values (1:1 mapping)
      float temp_array[4] = {temp_vec.x, temp_vec.y, temp_vec.z, temp_vec.w};
#pragma unroll 4
      for (int i = 0; i < 4; ++i) {
        // Convert float to bfloat16 by truncating mantissa
        uint16_t bf16_value = __float_as_uint(temp_array[i]) >> 16;
        bf16Output[globalIdx + i] = *reinterpret_cast<const T*>(&bf16_value);
      }
    }
  }

  // Scalar loop: handle remaining individual elements (1:1 mapping)
  for (size_t localIdx = fallbackCount + threadId; localIdx < chunkSize;
       localIdx += totalThreads) {
    size_t globalIdx = chunkStart + localIdx;

    if (__builtin_expect(globalIdx < totalCount && localIdx < chunkSize, 1)) {
      // Dequantize single FP8 to float, then convert to BF16
      float temp_float = dequantize_fp8_e4m3_to_float(fp8_input[globalIdx]);
      uint16_t bf16_value = __float_as_uint(temp_float) >> 16;
      bf16Output[globalIdx] = *reinterpret_cast<const T*>(&bf16_value);
    }
  }
}
