// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

#include <cuda.h>

struct ncclComm;

namespace ncclx::nvls {

CUresult multicastBindMemWithWatchdog(
    const ncclComm* comm,
    size_t inputSize,
    size_t ucsize,
    size_t mcsize,
    CUmemGenericAllocationHandle mcHandle,
    CUmemGenericAllocationHandle ucHandle);

} // namespace ncclx::nvls
