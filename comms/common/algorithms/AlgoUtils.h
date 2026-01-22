// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
using bf16 = hip_bfloat16;
#else
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using bf16 = __nv_bfloat16;
#endif
#include <stdexcept>
#include "comms/utils/commSpecs.h"

namespace meta {
namespace comms {

#define ASSIGN_FUNC_NRANKS(func, templ, nranks, hasAcc) \
  do {                                                  \
    switch ((nranks)) {                                 \
      case 2:                                           \
        func = (const void*)&templ<T, 2, hasAcc>;       \
        break;                                          \
                                                        \
      case 4:                                           \
        func = (const void*)&templ<T, 4, hasAcc>;       \
        break;                                          \
                                                        \
      case 8:                                           \
        func = (const void*)&templ<T, 8, hasAcc>;       \
        break;                                          \
                                                        \
      case 16:                                          \
        func = (const void*)&templ<T, 16, hasAcc>;      \
        break;                                          \
                                                        \
      default:                                          \
        throw std::runtime_error("Unsupported nranks"); \
    }                                                   \
  } while (0)

#define TYPED_CALL(commDataType, func, ...)                       \
  ({                                                              \
    do {                                                          \
      switch (commDataType) {                                     \
        case commFloat: {                                         \
          func<float>(__VA_ARGS__);                               \
          break;                                                  \
        }                                                         \
        case commFloat16: {                                       \
          func<half>(__VA_ARGS__);                                \
          break;                                                  \
        }                                                         \
        case commBfloat16: {                                      \
          func<bf16>(__VA_ARGS__);                                \
          break;                                                  \
        }                                                         \
        default: {                                                \
          throw std::runtime_error("Invalid type in TYPED_CALL"); \
        }                                                         \
      }                                                           \
    } while (0);                                                  \
  })

// determine the optimal grid/block size to launch kernel func
std::pair<dim3, dim3>
getGridAndBlockDims(size_t count, commDataType_t datatype, size_t maxBlocks);

} // namespace comms
} // namespace meta
