// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

#include <cuda.h>

#include "nccl.h" // @manual

struct ncclComm;

namespace ncclx::nvls {

CUresult multicastBindMemWithWatchdog(
    const ncclComm* comm,
    size_t inputSize,
    size_t ucsize,
    size_t mcsize,
    CUmemGenericAllocationHandle mcHandle,
    CUmemGenericAllocationHandle ucHandle,
    // Non-zero on 2.32+, where NVLS buffers are partitions of a larger
    // multicast group rather than owning the whole group.
    size_t mcOffset = 0,
    size_t memOffset = 0);

// cuMemMap of a multicast object under the same watchdog. On 2.32+ this map,
// not the bind, is where a rank waits for every local rank to join the group.
// Reports a failure as CUCHECKGOTO does and returns ncclUnhandledCudaError.
ncclResult_t multicastMapWithWatchdog(
    const ncclComm* comm,
    CUdeviceptr base,
    size_t size,
    CUmemGenericAllocationHandle mcHandle);

} // namespace ncclx::nvls
