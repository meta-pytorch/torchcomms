// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
// Avoid host-pass parsing of warp sync builtins from hip_bf16.h.
#if !defined(__HIP_DEVICE_COMPILE__) && defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
#undef HIP_ENABLE_WARP_SYNC_BUILTINS
#endif

#include <hip/hip_bf16.h>

#ifndef __CUDA_BF16_TYPES_EXIST__
#define __CUDA_BF16_TYPES_EXIST__
#endif

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#endif
