// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include "comms/ctran/algos/common/ColltraceEventScope.cuh"
#include "comms/utils/colltrace/ColltraceDeviceHandle.h"

// Every thread constructs the scope; ColltraceEventScope elects a single writer
// (block 0, thread 0) internally, emitting the start on construction and the
// end on destruction into the armed handle's ring.
__global__ void colltraceScopeKernel(
    meta::comms::colltrace::ColltraceDeviceHandle handle) {
  ctran::device::ColltraceEventScope scope(handle);
}

void launchColltraceScopeKernel(
    const meta::comms::colltrace::ColltraceDeviceHandle& handle,
    int blocks,
    int threads,
    cudaStream_t stream) {
  colltraceScopeKernel<<<blocks, threads, 0, stream>>>(handle);
}
