/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "low_precision_utility.h"

// The `__global__` kernel definitions for this header now live in
// `low_precision_kernels.cu`. Keeping the bodies out of the header is required
// for the `rcclx-dev` build path, where every `.cc` translation unit that
// includes this header is compiled with `--offload-host-only` (see
// `HOST_ONLY_HIP_FLAGS` in `fbcode/comms/rcclx/rccl_build_config.bzl`). With
// `--offload-host-only` the compiler drops `__global__` kernel bodies before
// device codegen runs, which would otherwise leave `librcclx-dev.a` with host
// stubs but no device code for any of these kernels — the first launch from
// `RCCL_LOW_PRECISION_ENABLE=1` consumers would then SIGABRT with
// `hip_global.cpp:117 :  Cannot find Symbol with name: _Z*Kernel*`.
//
// The dedicated `.cu` is compiled in a separate `cpp_library`
// (`rccl_lpcoll_kernels_obj{suffix}`) with monolithic non-RDC HIP
// (`-fno-gpu-rdc`, no `--offload-host-only`), and merged into the final
// archive via `additional_archives`. Per-`(kernel, T)`-pair explicit template
// instantiations in that `.cu` emit the external-linkage symbols that the
// `<<<...>>>` host stubs generated for these declarations resolve against.

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
    size_t chunkSize);

extern template __global__ void quantizeFloatToFp8Kernel<float>(
    const float* input,
    void* output,
    size_t totalCount,
    size_t chunkStart,
    size_t chunkSize);

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
    size_t chunkSize);

extern template __global__ void quantizeBF16ToFp8Kernel<uint16_t>(
    const uint16_t* bf16Input,
    void* fp8Output,
    size_t totalOutputCount,
    size_t chunkStart,
    size_t chunkSize);

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
    size_t chunkSize);

extern template __global__ void dequantizeFloatToBF16Kernel<float>(
    const float* floatInput,
    uint16_t* bf16Output,
    size_t totalFloatCount,
    size_t chunkStart,
    size_t chunkSize);

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
    int myRank);

extern template __global__ void localReductionKernel<float>(
    const void* fp8Input,
    float* floatOutput,
    size_t totalCount,
    size_t chunkStart,
    size_t chunkSize,
    int nRanks,
    int myRank);

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
    size_t chunkSize);

extern template __global__ void dequantizeFp8ToFloatKernel<float>(
    const void* fp8Input,
    float* floatOutput,
    size_t totalCount,
    size_t chunkStart,
    size_t chunkSize);

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
    size_t chunkSize);

extern template __global__ void dequantizeFp8ToBF16Kernel<uint16_t>(
    const void* fp8Input,
    uint16_t* bf16Output,
    size_t totalCount,
    size_t chunkStart,
    size_t chunkSize);
