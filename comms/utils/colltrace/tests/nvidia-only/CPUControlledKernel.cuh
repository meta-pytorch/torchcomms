// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>

enum KernelFlag : int32_t {
  // Indicate a free flag
  KERNEL_FLAG_UNSET = 0,
  // Indicate the kernel has been scheduled and the flag is inuse
  KERNEL_FLAG_SCHEDULED = 1,
  // Indicate kernel has started, so that GPE thread can start
  KERNEL_FLAG_STARTED = 2,
  // Indicate GPE thread has finished, so that kernel can terminate
  KERNEL_FLAG_TERMINATE = 3,
};

inline __device__ int32_t loadInt(int32_t* ptr) {
  int32_t v;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

inline __device__ void storeInt(int32_t* ptr, int32_t val) {
  asm volatile("st.volatile.global.s32 [%0], %1;" ::"l"(ptr), "r"(val));
}

inline __device__ void kernelSignalStart(KernelFlag* flag) {
  storeInt(reinterpret_cast<int32_t*>(flag), KERNEL_FLAG_STARTED);
}

inline __device__ void kernelWaitCPUSignal(KernelFlag* flag) {
  KernelFlag flagVal = KERNEL_FLAG_STARTED;
  do {
    flagVal =
        static_cast<KernelFlag>(loadInt(reinterpret_cast<int32_t*>(flag)));
  } while (flagVal != KERNEL_FLAG_TERMINATE);

  // Mark the flag as unset for reclaim
  storeInt(reinterpret_cast<int32_t*>(flag), KERNEL_FLAG_UNSET);
}

__global__ void waitCPUKernel(KernelFlag* flag);
