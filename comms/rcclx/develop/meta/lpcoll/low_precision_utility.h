/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>
#include "collectives.h"
#include "rccl_float8.h"

#define ALWAYS_INLINE __forceinline__ __host__ __device__
#define HOT __attribute__((hot))
#define PURE __attribute__((pure))
#define RESTRICT __restrict__

#define DEFAULT_BLOCK_SIZE 1024
#define DEFAULT_MAX_BLOCKS 304

// Cache line size for optimal memory access alignment in low precision
// collectives
#define CACHE_LINE_SIZE 128UL

// Refer to MI300X CDNA3 ISA manual for more details on the HW intrinsics

/**
 * Converts a single float value to FP8 E4M3 format using AMD GPU instruction.
 */
__device__ __forceinline__ rccl_float8 quantize_float_to_fp8_e4m3(float value) {
  uint32_t ival = 0;
  asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2"
               : "=v"(ival)
               : "v"(value), "v"(value));
  rccl_float8* packed_fp8 = reinterpret_cast<rccl_float8*>(&ival);
  return packed_fp8[0];
}

/**
 * Converts a single FP8 E4M3 value back to float using AMD GPU instruction.
 */
ALWAYS_INLINE PURE float dequantize_fp8_e4m3_to_float(rccl_float8 fp8_value) {
  float result;
  uint32_t fp8_bits = *reinterpret_cast<const uint8_t*>(&fp8_value);
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0"
               : "=v"(result)
               : "v"(fp8_bits));
  return result;
}

/**
 * Converts 4 float values to FP8 E4M3 format using vectorized operations.
 */
__device__ __forceinline__ void quantize_float4_to_fp8_batch(
    const float4& input,
    void* output) {
  rccl_float8* fp8_output = static_cast<rccl_float8*>(output);
  uint32_t packed_result = 0;

  asm volatile(
      "v_cvt_pk_fp8_f32 %0, %1, %2\n"
      "v_cvt_pk_fp8_f32 %0, %3, %4, op_sel:[0, 0, 1]\n"
      : "=v"(packed_result)
      : "v"(input.x), "v"(input.y), "v"(input.z), "v"(input.w));

  rccl_float8* packed_fp8 = reinterpret_cast<rccl_float8*>(&packed_result);
  fp8_output[0] = packed_fp8[0];
  fp8_output[1] = packed_fp8[1];
  fp8_output[2] = packed_fp8[2];
  fp8_output[3] = packed_fp8[3];
}

/**
 * Converts 4 FP8 E4M3 values back to float4 using vectorized operations.
 */
__device__ __forceinline__ float4
dequantize_fp8_batch_to_float4(const void* input) {
  const rccl_float8* fp8_input = static_cast<const rccl_float8*>(input);
  float4 result;
  uint32_t packed_input = *reinterpret_cast<const uint32_t*>(fp8_input);

  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0"
               : "=v"(result.x)
               : "v"(packed_input));
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_1"
               : "=v"(result.y)
               : "v"(packed_input));
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_2"
               : "=v"(result.z)
               : "v"(packed_input));
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_3"
               : "=v"(result.w)
               : "v"(packed_input));

  return result;
}

/**
 * Converts 8 float values to FP8 E4M3 format using dual 4-element vectorized
 * operations.
 */
__device__ __forceinline__ void quantize_float8_to_fp8_batch(
    const float* input,
    void* output) {
  rccl_float8* fp8_output = static_cast<rccl_float8*>(output);
  uint32_t packed_result_0 = 0;
  uint32_t packed_result_1 = 0;

  asm volatile(
      "v_cvt_pk_fp8_f32 %0, %1, %2\n"
      "v_cvt_pk_fp8_f32 %0, %3, %4, op_sel:[0, 0, 1]\n"
      : "=v"(packed_result_0)
      : "v"(input[0]), "v"(input[1]), "v"(input[2]), "v"(input[3]));

  asm volatile(
      "v_cvt_pk_fp8_f32 %0, %1, %2\n"
      "v_cvt_pk_fp8_f32 %0, %3, %4, op_sel:[0, 0, 1]\n"
      : "=v"(packed_result_1)
      : "v"(input[4]), "v"(input[5]), "v"(input[6]), "v"(input[7]));

  rccl_float8* packed_fp8_0 = reinterpret_cast<rccl_float8*>(&packed_result_0);
  rccl_float8* packed_fp8_1 = reinterpret_cast<rccl_float8*>(&packed_result_1);

  fp8_output[0] = packed_fp8_0[0];
  fp8_output[1] = packed_fp8_0[1];
  fp8_output[2] = packed_fp8_0[2];
  fp8_output[3] = packed_fp8_0[3];
  fp8_output[4] = packed_fp8_1[0];
  fp8_output[5] = packed_fp8_1[1];
  fp8_output[6] = packed_fp8_1[2];
  fp8_output[7] = packed_fp8_1[3];
}

/**
 * Converts 8 FP8 E4M3 values back to 8 float values using dual vectorized
 * operations.
 */
__device__ __forceinline__ void dequantize_fp8_batch_to_float8(
    const void* input,
    float* output) {
  const rccl_float8* fp8_input = static_cast<const rccl_float8*>(input);
  uint32_t packed_input_0 = *reinterpret_cast<const uint32_t*>(&fp8_input[0]);
  uint32_t packed_input_1 = *reinterpret_cast<const uint32_t*>(&fp8_input[4]);

  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0"
               : "=v"(output[0])
               : "v"(packed_input_0));
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_1"
               : "=v"(output[1])
               : "v"(packed_input_0));
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_2"
               : "=v"(output[2])
               : "v"(packed_input_0));
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_3"
               : "=v"(output[3])
               : "v"(packed_input_0));

  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0"
               : "=v"(output[4])
               : "v"(packed_input_1));
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_1"
               : "=v"(output[5])
               : "v"(packed_input_1));
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_2"
               : "=v"(output[6])
               : "v"(packed_input_1));
  asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_3"
               : "=v"(output[7])
               : "v"(packed_input_1));
}

/**
 * Data structure to manage buffer layout and addressing for multi-phase
 * collective operations. Provides utilities for chunk partitioning, offset
 * calculation, and address mapping.
 */
struct PhaseBufferLayout {
  enum LayoutType { CONTIGUOUS, INTERLEAVED };

  LayoutType type;
  size_t totalElements;
  int nRanks;
  int myRank;
  size_t uniformChunkSize;

  // Calculate memory offset for a specific rank's chunk in the buffer layout
  __host__ __device__ __forceinline__ size_t getChunkOffset(int rank) const {
    return static_cast<size_t>(rank) * uniformChunkSize;
  }

  // Get the size of a chunk for the specified rank (uniform for all ranks)
  __host__ __device__ __forceinline__ size_t getChunkSize(int rank) const {
    return uniformChunkSize;
  }

  // Calculate memory offset for data contributions from a specific source rank
  __host__ __device__ __forceinline__ size_t
  getContributionOffset(int sourceRank) const {
    return static_cast<size_t>(sourceRank) * uniformChunkSize;
  }

  // Check if a global index belongs to the specified target rank's chunk
  __host__ __device__ __forceinline__ bool isValidGlobalIndex(
      size_t globalIdx,
      int targetRank) const {
    size_t chunkStart = getChunkOffset(targetRank);
    return (globalIdx >= chunkStart) &&
        (globalIdx < chunkStart + uniformChunkSize) &&
        (globalIdx < totalElements);
  }

  // Determine which rank owns a specific global index based on chunk
  // partitioning
  __host__ __device__ __forceinline__ int getOwnerRank(size_t globalIdx) const {
    return static_cast<int>(globalIdx / uniformChunkSize);
  }

  // Convert a global index to its corresponding local index within a chunk
  __host__ __device__ __forceinline__ size_t
  getLocalIndex(size_t globalIdx) const {
    return globalIdx % uniformChunkSize;
  }

  // Constructor to initialize buffer layout parameters for multi-phase
  // allreduce
  PhaseBufferLayout(
      LayoutType t,
      size_t total,
      int ranks,
      int rank,
      size_t uniformSize)
      : type(t),
        totalElements(total),
        nRanks(ranks),
        myRank(rank),
        uniformChunkSize(uniformSize) {}

  // Debug method to print buffer layout information (currently empty
  // implementation)
  void print(const char* label) const {}
};
