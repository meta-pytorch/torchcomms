// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <utility>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#else
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#endif

#include "comms/prims/collectives/moe_ep/cpp/shared/Config.h"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Exception.cuh"

//
// Provides launch-config helpers + dispatch-by-num-ranks switch macros.
//
// Two notable simplifications:
//
//  1) `LAUNCH_KERNEL_NON_COOPERATIVE` collapses to a plain triple-bracket
//     launch on both NVIDIA and AMD, avoiding `cudaLaunchKernelEx` and its
//     cooperative-launch attributes. We don't use cooperative launches in
//     Phase 1 (notify_dispatch is the only place
//     that touches `task_fifo_ptrs`, and we replace that with pipes
//     `barrier_all` which doesn't require cooperative launch).
//
//  2) `SWITCH_RANKS` is restricted to the values D3 actually instantiates
//     (2 / 4 / 8). D5 (internode) extends with rdma-rank variants.

namespace comms::prims::moe_ep::kernels {

#ifdef __HIP_PLATFORM_AMD__
#define MOE_EP_GPU_R_16BF HIP_R_16BF
#define MOE_EP_GPU_R_32F HIP_R_32F
using gpu_bfloat16_t = hip_bfloat16;
#else
#define MOE_EP_GPU_R_16BF CUDA_R_16BF
#define MOE_EP_GPU_R_32F CUDA_R_32F
using gpu_bfloat16_t = nv_bfloat16;
#endif

// Light-weight launch config that works on both NVIDIA and AMD without
// needing cudaLaunchKernelEx / cooperative launch.
struct LaunchConfig {
  dim3 num_sms;
  dim3 num_threads;
  unsigned int shared_mem_bytes{0};
  cudaStream_t stream{nullptr};
};

} // namespace comms::prims::moe_ep::kernels

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
  ::comms::prims::moe_ep::kernels::LaunchConfig cfg = {   \
      static_cast<dim3>(num_sms),                         \
      static_cast<dim3>(num_threads),                     \
      0u,                                                 \
      (stream)}
#endif

#ifndef LAUNCH_KERNEL_NON_COOPERATIVE
#define LAUNCH_KERNEL_NON_COOPERATIVE(cfg_ptr, kernel, ...) \
  do {                                                      \
    auto* _cfg = (cfg_ptr);                                 \
    (kernel)<<<                                             \
        _cfg->num_sms,                                      \
        _cfg->num_threads,                                  \
        _cfg->shared_mem_bytes,                             \
        _cfg->stream>>>(__VA_ARGS__);                       \
  } while (0)
#endif

#ifndef SWITCH_RANKS
#define SWITCH_RANKS(case_macro)                                     \
  switch (num_ranks) {                                               \
    case 2:                                                          \
      case_macro(2);                                                 \
      break;                                                         \
    case 4:                                                          \
      case_macro(4);                                                 \
      break;                                                         \
    case 8:                                                          \
      case_macro(8);                                                 \
      break;                                                         \
    default:                                                         \
      EP_HOST_ASSERT(false && "Unsupported num_ranks (need 2/4/8)"); \
  }
#endif

#ifndef SWITCH_RANKS_WITH_DTYPE
#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)                   \
  switch (num_ranks) {                                               \
    case 2:                                                          \
      case_macro(dtype, 2);                                          \
      break;                                                         \
    case 4:                                                          \
      case_macro(dtype, 4);                                          \
      break;                                                         \
    case 8:                                                          \
      case_macro(dtype, 8);                                          \
      break;                                                         \
    default:                                                         \
      EP_HOST_ASSERT(false && "Unsupported num_ranks (need 2/4/8)"); \
  }
#endif

#ifndef SWITCH_TYPES
#define SWITCH_TYPES(case_macro)                                    \
  switch (type) {                                                   \
    case MOE_EP_GPU_R_16BF:                                         \
      case_macro(::comms::prims::moe_ep::kernels::gpu_bfloat16_t);  \
      break;                                                        \
    case MOE_EP_GPU_R_32F:                                          \
      case_macro(float);                                            \
      break;                                                        \
    default:                                                        \
      EP_HOST_ASSERT(false && "Unsupported dtype (need bf16/f32)"); \
  }
#endif

#ifndef SWITCH_HIDDEN
#define SWITCH_HIDDEN(case_macro)                    \
  switch (hidden) {                                  \
    case 2048:                                       \
      case_macro(2048);                              \
      break;                                         \
    case 2560:                                       \
      case_macro(2560);                              \
      break;                                         \
    case 4096:                                       \
      case_macro(4096);                              \
      break;                                         \
    case 5120:                                       \
      case_macro(5120);                              \
      break;                                         \
    case 7168:                                       \
      case_macro(7168);                              \
      break;                                         \
    default:                                         \
      EP_HOST_ASSERT(false && "Unsupported hidden"); \
  }
#endif
