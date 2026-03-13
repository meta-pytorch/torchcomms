// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::benchmark {

using namespace comms::pipes;

__global__ void copyKernel(
    char* dst,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope);

__global__ void sequentialCopyKernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope);

__global__ void dualDestCopyKernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope);

__global__ void sequentialTriCopyKernel(
    char* dst1,
    char* dst2,
    char* dst3,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope);

__global__ void triDestCopyKernel(
    char* dst1,
    char* dst2,
    char* dst3,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope);

} // namespace comms::pipes::benchmark
