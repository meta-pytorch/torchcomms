// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/testinfra/TestsCuUtils.h"

// Test version: Set dynamic shared memory size and launch kernel with it
#define CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_TEST(                              \
    kernel, grid, block, args, dynamicShmSize, stream)                         \
  do {                                                                         \
    CUDACHECK_TEST(cudaFuncSetAttribute(                                       \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicShmSize)); \
    CUDACHECK_TEST(                                                            \
        cudaLaunchKernel(kernel, grid, block, args, dynamicShmSize, stream));  \
  } while (0)
