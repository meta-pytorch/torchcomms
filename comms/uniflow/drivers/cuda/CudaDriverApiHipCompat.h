// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// CUDA driver spellings that the hipified CudaDriverApi.h and its consumers
// need on AMD. This header is deliberately not hipified, so each alias keeps
// its CUDA name whichever hipify tool translates the including file:
// - CUmemRangeHandleType, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD: hipify-perl
//   from ROCm 7.2 and TheRock maps them to HIP names, ROCm 7.0's only with
//   -experimental, torch hipify not at all. A hipified caller sees either the
//   HIP name (from HIP's headers) or the CUDA name (from here).
// - CU_STREAM_WRITE_VALUE_DEFAULT: no hipify-perl maps it without
//   -experimental. Torch hipify maps it to hipStreamWriteValueDefault, which
//   only the TheRock headers define (ROCm 7.0 and 7.2 do not), so on AMD only
//   hipify-perl-translated code may use it.
// Hipifying this file would turn the aliases into redefinitions of HIP names.

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime_api.h>

using CUmemRangeHandleType = hipMemRangeHandleType;
inline constexpr hipMemRangeHandleType CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD =
    hipMemRangeHandleTypeDmaBufFd;
// 0 is CUDA's default, which fences prior memory operations before the write.
// HIP 7.0 to 7.16 document the flags argument as reserved and ignored; the
// TheRock (7.16) headers also define hipStreamWriteValueDefault as 0.
inline constexpr unsigned int CU_STREAM_WRITE_VALUE_DEFAULT = 0u;
#endif
