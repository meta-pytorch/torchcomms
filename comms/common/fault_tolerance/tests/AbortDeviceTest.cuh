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

cudaError_t launchDeviceSetAbortWithContext(
    AbortDevice abort,
    AbortReason reason,
    bool useContext,
    int* observedWinner,
    cudaStream_t stream);

cudaError_t launchAbortFlagSetAbort(
    AbortDevice abort,
    AbortReason reason,
    int* observedWinner,
    int* observedContextReady,
    cudaStream_t stream);

cudaError_t launchDevicePublishReasonWithoutContext(
    AbortDevice abort,
    AbortReason reason,
    int* observedWinner,
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

cudaError_t launchDeviceReadCheckResult(
    AbortDevice abort,
    int* observedCheckResult,
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

// FT_ABORT_* macro coverage. Each kernel runs a bounded spin loop that can only
// end through the macro under test, and reports how many iterations it took, so
// a macro that fails to terminate shows up as the loop bound rather than a
// hang.
cudaError_t launchMacroBreakLoop(
    AbortDevice abort,
    int* observedIterations,
    int maxIterations,
    cudaStream_t stream);

// Same loop, but the kernel arms the deadline itself so the timeout is measured
// against the device clock the checks read.
cudaError_t launchMacroTimeoutLoop(
    AbortDevice abort,
    int* observedIterations,
    int maxIterations,
    cudaStream_t stream);

cudaError_t launchMacroCheckLoop(
    AbortDevice abort,
    int* observedIterations,
    int* observedStop,
    int maxIterations,
    cudaStream_t stream);

cudaError_t launchMacroReturnValue(
    AbortDevice abort,
    int* observedReturn,
    cudaStream_t stream);

// Runs FT_ABORT_BREAK as the unbraced body of an `if` that has an `else`, so a
// macro whose `if` swallows that `else` shows up as `observedFallback == 1`
// rather than as a silent behavior change at some future call site.
cudaError_t launchMacroBreakInIfElse(
    AbortDevice abort,
    int* observedIterations,
    int* observedFallback,
    int maxIterations,
    cudaStream_t stream);

} // namespace comms::fault_tolerance::testing
