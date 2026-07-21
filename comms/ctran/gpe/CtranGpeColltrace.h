// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <mutex>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::gpe {

// In-kernel colltrace (the collective kernel publishing its own start/end
// timestamps into the colltrace ring) requires sm_90+ for the ring's 128b
// atomic device write — the same hardware requirement as the GPE device ring,
// but gated independently of it via NCCL_COLLTRACE_IN_KERNEL. The compute
// capability is queried once per process (call_once); NCCL pins one device per
// process, so a single cached value is correct.
inline bool inKernelColltraceSupported() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return false;
#else
  if (!NCCL_COLLTRACE_IN_KERNEL) {
    return false;
  }
  static std::once_flag onceFlag;
  static int ccMajor = -1; // -1 = unknown/query failed
  std::call_once(onceFlag, [] {
    int dev = 0;
    int major = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess) {
      ccMajor = major;
    }
  });
  return ccMajor >= 9;
#endif
}

} // namespace ctran::gpe
