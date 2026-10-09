// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include "nccl.h"
#include "nccl_device.h"

// Each rank puts `bytes` from its local-only source window into its slot of the
// next rank's collective window, then waits for the put from the previous rank.
cudaError_t launchPutFromLocalWindow(
    ncclWindow_t srcWin,
    ncclWindow_t dstWin,
    size_t bytes,
    const ncclDevComm& devComm,
    cudaStream_t stream);
