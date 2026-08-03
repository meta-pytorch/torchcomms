// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "comms/common/fault_tolerance/AbortDevice.cuh"

namespace comms::fault_tolerance::benchmark {

cudaError_t
launchDeviceLoadLoop(int* flag, int* sink, int iterations, cudaStream_t stream);

cudaError_t launchManyBlockDeviceLoadLoop(
    int* flag,
    int* sink,
    int blocks,
    int threads,
    int iterations,
    cudaStream_t stream);

cudaError_t
launchDeviceStoreLoop(int* flag, int iterations, cudaStream_t stream);

cudaError_t launchDeviceToHostRoundTrip(
    int* request,
    int* response,
    int* ready,
    int iterations,
    cudaStream_t stream);

cudaError_t launchDeviceToDevicePingPong(
    int* request,
    int* response,
    int* ready,
    int* start,
    int* observed,
    int iterations,
    int maxPolls,
    cudaStream_t stream);

cudaError_t launchAbortDeviceDefaultTimeoutLoadLoop(
    AbortDevice abort,
    int64_t* sink,
    int iterations,
    cudaStream_t stream);

} // namespace comms::fault_tolerance::benchmark
