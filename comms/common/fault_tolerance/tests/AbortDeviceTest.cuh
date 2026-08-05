// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "comms/common/fault_tolerance/AbortDevice.cuh"

namespace comms::fault_tolerance::testing {

cudaError_t launchDeviceSetAbort(
    AbortDevice abort,
    AbortReason reason,
    cudaStream_t stream);

cudaError_t launchDeviceReadAbort(
    AbortDevice abort,
    int* observed,
    int* observedMode,
    cudaStream_t stream);

cudaError_t launchDeviceWaitForAbort(
    AbortDevice abort,
    int* observed,
    int* observedMode,
    int maxIterations,
    cudaStream_t stream);

cudaError_t launchDeviceReadDefaultTimeoutMs(
    AbortDevice abort,
    int64_t* observedTimeoutMs,
    cudaStream_t stream);

cudaError_t launchDeviceReadAbortPredicate(
    AbortDevice abort,
    int* observedIsAborted,
    int* observedReason,
    cudaStream_t stream);

cudaError_t launchDeviceReadCheckExpired(
    AbortDevice abort,
    int* observedCheckExpired,
    int* observedReason,
    cudaStream_t stream);

cudaError_t launchDeviceWaitForTimeout(
    AbortDevice abort,
    int* observedMode,
    int* observedIsAborted,
    int maxIterations,
    cudaStream_t stream);

cudaError_t launchDeviceWaitForTimeoutStartAlias(
    AbortDevice abort,
    int* observedMode,
    int* observedCheckExpired,
    int maxIterations,
    cudaStream_t stream);

cudaError_t launchDeviceCancelAndRestartTimeout(
    AbortDevice abort,
    int* observedAfterCancel,
    int* observedMode,
    int maxIterations,
    cudaStream_t stream);

} // namespace comms::fault_tolerance::testing
