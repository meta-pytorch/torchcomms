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
    int* observed,
    int iterations,
    uint64_t maxWaitCycles,
    cudaStream_t stream);

// `observed` must point at kPingPongBlocks elements: each block reports its own
// status into its own slot, so a late timeout in one block cannot be masked by
// the other block's success.
inline constexpr int kPingPongBlocks = 2;

cudaError_t launchDeviceToDevicePingPong(
    int* request,
    int* response,
    int* ready,
    int* start,
    int* observed,
    int iterations,
    uint64_t maxWaitCycles,
    cudaStream_t stream);

cudaError_t launchAbortDeviceDefaultTimeoutLoadLoop(
    AbortDevice abort,
    int64_t* sink,
    int iterations,
    cudaStream_t stream);

cudaError_t launchAbortDeviceIsAbortedLoadLoop(
    AbortDevice abort,
    int* sink,
    int iterations,
    bool startTimeout,
    cudaStream_t stream);

} // namespace comms::fault_tolerance::benchmark
