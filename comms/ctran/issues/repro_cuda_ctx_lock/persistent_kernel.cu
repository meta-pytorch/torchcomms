// Copyright (c) Meta Platforms, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

__global__ void persistentKernel(
    volatile int* shutdown,
    volatile int* started) {
  if (threadIdx.x == 0) {
    *started = 1;
    __threadfence_system();
  }
  __syncthreads();
  while (*shutdown == 0) {
  }
}
