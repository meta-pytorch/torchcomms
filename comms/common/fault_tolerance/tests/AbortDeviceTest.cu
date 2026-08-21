// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/fault_tolerance/tests/AbortDeviceTest.cuh"

#include "comms/common/fault_tolerance/AbortMacros.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace comms::fault_tolerance::testing {
namespace {

__global__ void deviceSetAbortKernel(AbortDevice abort, AbortReason reason) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    abort.setAbort(reason);
  }
}

__global__ void
deviceReadAbortKernel(AbortDevice abort, int* observed, int* observedMode) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    auto reason = abort.reason();
    *observedMode = static_cast<int>(reason);
    *observed = reason != AbortReason::NONE ? 1 : 0;
  }
}

__global__ void deviceWaitForAbortKernel(
    AbortDevice abort,
    int* observed,
    int* observedMode,
    int maxIterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  for (int i = 0; i < maxIterations; ++i) {
    auto reason = abort.reason();
    if (reason != AbortReason::NONE) {
      *observedMode = static_cast<int>(reason);
      *observed = 1;
      return;
    }
    __nanosleep(64);
  }

  *observedMode = static_cast<int>(AbortReason::NONE);
  *observed = 0;
}

__global__ void deviceReadDefaultTimeoutMsKernel(
    AbortDevice abort,
    int64_t* observedTimeoutMs) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *observedTimeoutMs = abort.getTimeoutMs();
  }
}

__global__ void deviceReadAbortPredicateKernel(
    AbortDevice abort,
    int* observedIsAborted,
    int* observedReason) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *observedIsAborted = abort.isAborted() ? 1 : 0;
    *observedReason = static_cast<int>(abort.reason());
  }
}

__global__ void deviceReadCheckExpiredKernel(
    AbortDevice abort,
    int* observedCheckExpired,
    int* observedReason) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *observedCheckExpired = abort.checkExpired() ? 1 : 0;
    *observedReason = static_cast<int>(abort.reason());
  }
}

__global__ void deviceReadCheckResultKernel(
    AbortDevice abort,
    int* observedCheckResult) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *observedCheckResult = static_cast<int>(abort.check());
  }
}

__global__ void deviceWaitForTimeoutKernel(
    AbortDevice abort,
    int* observedMode,
    int* observedIsAborted,
    int maxIterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  abort.startTimeout();
  for (int i = 0; i < maxIterations; ++i) {
    if (abort.isAborted()) {
      *observedMode = static_cast<int>(abort.reason());
      *observedIsAborted = 1;
      return;
    }
    __nanosleep(64);
  }

  *observedMode = static_cast<int>(AbortReason::NONE);
  *observedIsAborted = 0;
}

__global__ void deviceWaitForTimeoutStartAliasKernel(
    AbortDevice abort,
    int* observedMode,
    int* observedCheckExpired,
    int maxIterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  abort.start();
  for (int i = 0; i < maxIterations; ++i) {
    if (abort.checkExpired()) {
      *observedMode = static_cast<int>(abort.reason());
      *observedCheckExpired = 1;
      return;
    }
    __nanosleep(64);
  }

  *observedMode = static_cast<int>(AbortReason::NONE);
  *observedCheckExpired = 0;
}

__global__ void deviceCancelAndRestartTimeoutKernel(
    AbortDevice abort,
    int* observedAfterCancel,
    int* observedMode,
    int maxIterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  abort.startTimeout();
  abort.cancelTimeout();
  for (int i = 0; i < maxIterations; ++i) {
    if (abort.isAborted()) {
      *observedAfterCancel = 1;
      *observedMode = static_cast<int>(abort.reason());
      return;
    }
    __nanosleep(64);
  }

  *observedAfterCancel = 0;
  abort.startTimeout();
  for (int i = 0; i < maxIterations; ++i) {
    if (abort.isAborted()) {
      *observedMode = static_cast<int>(abort.reason());
      return;
    }
    __nanosleep(64);
  }

  *observedMode = static_cast<int>(AbortReason::NONE);
}

} // namespace

cudaError_t launchDeviceSetAbort(
    AbortDevice abort,
    AbortReason reason,
    cudaStream_t stream) {
  deviceSetAbortKernel<<<1, 1, 0, stream>>>(abort, reason);
  return cudaGetLastError();
}

cudaError_t launchDeviceReadAbort(
    AbortDevice abort,
    int* observed,
    int* observedMode,
    cudaStream_t stream) {
  deviceReadAbortKernel<<<1, 1, 0, stream>>>(abort, observed, observedMode);
  return cudaGetLastError();
}

cudaError_t launchDeviceWaitForAbort(
    AbortDevice abort,
    int* observed,
    int* observedMode,
    int maxIterations,
    cudaStream_t stream) {
  deviceWaitForAbortKernel<<<1, 1, 0, stream>>>(
      abort, observed, observedMode, maxIterations);
  return cudaGetLastError();
}

cudaError_t launchDeviceReadDefaultTimeoutMs(
    AbortDevice abort,
    int64_t* observedTimeoutMs,
    cudaStream_t stream) {
  deviceReadDefaultTimeoutMsKernel<<<1, 1, 0, stream>>>(
      abort, observedTimeoutMs);
  return cudaGetLastError();
}

cudaError_t launchDeviceReadAbortPredicate(
    AbortDevice abort,
    int* observedIsAborted,
    int* observedReason,
    cudaStream_t stream) {
  deviceReadAbortPredicateKernel<<<1, 1, 0, stream>>>(
      abort, observedIsAborted, observedReason);
  return cudaGetLastError();
}

cudaError_t launchDeviceReadCheckExpired(
    AbortDevice abort,
    int* observedCheckExpired,
    int* observedReason,
    cudaStream_t stream) {
  deviceReadCheckExpiredKernel<<<1, 1, 0, stream>>>(
      abort, observedCheckExpired, observedReason);
  return cudaGetLastError();
}

cudaError_t launchDeviceReadCheckResult(
    AbortDevice abort,
    int* observedCheckResult,
    cudaStream_t stream) {
  deviceReadCheckResultKernel<<<1, 1, 0, stream>>>(abort, observedCheckResult);
  return cudaGetLastError();
}

cudaError_t launchDeviceWaitForTimeout(
    AbortDevice abort,
    int* observedMode,
    int* observedIsAborted,
    int maxIterations,
    cudaStream_t stream) {
  deviceWaitForTimeoutKernel<<<1, 1, 0, stream>>>(
      abort, observedMode, observedIsAborted, maxIterations);
  return cudaGetLastError();
}

cudaError_t launchDeviceWaitForTimeoutStartAlias(
    AbortDevice abort,
    int* observedMode,
    int* observedCheckExpired,
    int maxIterations,
    cudaStream_t stream) {
  deviceWaitForTimeoutStartAliasKernel<<<1, 1, 0, stream>>>(
      abort, observedMode, observedCheckExpired, maxIterations);
  return cudaGetLastError();
}

cudaError_t launchDeviceCancelAndRestartTimeout(
    AbortDevice abort,
    int* observedAfterCancel,
    int* observedMode,
    int maxIterations,
    cudaStream_t stream) {
  deviceCancelAndRestartTimeoutKernel<<<1, 1, 0, stream>>>(
      abort, observedAfterCancel, observedMode, maxIterations);
  return cudaGetLastError();
}

// --- FT_ABORT_* macro coverage -------------------------------------------
//
// Every loop below is bounded by maxIterations so a macro that never
// terminates fails as "ran to the bound" instead of hanging the suite.

namespace {

__global__ void macroBreakLoopKernel(
    AbortDevice abort,
    int* observedIterations,
    int maxIterations) {
  int iterations = 0;
  while (iterations < maxIterations) {
    ++iterations;
    FT_ABORT_BREAK(abort, "macroBreakLoop iteration %d", iterations);
  }
  *observedIterations = iterations;
}

// Guards the `else`-binding of FT_ABORT_BREAK. The macro expands to an `if`,
// so without the trailing `else (void)0` the caller's `else` below would bind
// to the macro instead, and `fallback` would run on the healthy path -- exactly
// when the caller wrote it not to. Braces are omitted deliberately: this is the
// shape that misbinds, so the test is only meaningful unbraced.
__global__ void macroBreakInIfElseKernel(
    AbortDevice abort,
    int* observedIterations,
    int* observedFallback,
    int maxIterations) {
  int iterations = 0;
  int fallback = 0;
  while (iterations < maxIterations) {
    ++iterations;
    if (iterations > 0)
      FT_ABORT_BREAK(abort, "macroBreakInIfElse iteration %d", iterations);
    else
      fallback = 1;
  }
  *observedIterations = iterations;
  *observedFallback = fallback;
}

__global__ void macroCheckLoopKernel(
    AbortDevice abort,
    int* observedIterations,
    int* observedStop,
    int maxIterations) {
  int iterations = 0;
  bool stop = false;
  while (iterations < maxIterations) {
    ++iterations;
    // The CHECK form: the caller decides what to do with the result.
    stop = FT_ABORT_CHECK(abort, "macroCheckLoop iteration %d", iterations);
    if (stop) {
      break;
    }
  }
  *observedIterations = iterations;
  *observedStop = stop ? 1 : 0;
}

// Arms the deadline on the device. `startTimeout()` is `__device__`-only and
// reads the device clock, so arming it from the host would derive the deadline
// from a clock the kernel never sees and the very first check would report it
// expired -- the loop would end for the wrong reason.
__global__ void macroTimeoutLoopKernel(
    AbortDevice abort,
    int* observedIterations,
    int maxIterations) {
  abort.startTimeout();
  int iterations = 0;
  while (iterations < maxIterations) {
    ++iterations;
    FT_ABORT_BREAK(abort, "macroTimeoutLoop iteration %d", iterations);
    // Paces the loop so the bound is reached well after the deadline rather
    // than spinning past it.
    __nanosleep(1000);
  }
  *observedIterations = iterations;
}

__global__ void macroReturnValueKernel(AbortDevice abort, int* observedReturn) {
  // Wrapped so RETURN has a function to leave; -1 marks "aborted".
  auto body = [&]() -> int {
    FT_ABORT_RETURN(abort, -1, "macroReturnValue");
    return 7;
  };
  *observedReturn = body();
}

} // namespace

cudaError_t launchMacroBreakLoop(
    AbortDevice abort,
    int* observedIterations,
    int maxIterations,
    cudaStream_t stream) {
  macroBreakLoopKernel<<<1, 1, 0, stream>>>(
      abort, observedIterations, maxIterations);
  return cudaGetLastError();
}

cudaError_t launchMacroCheckLoop(
    AbortDevice abort,
    int* observedIterations,
    int* observedStop,
    int maxIterations,
    cudaStream_t stream) {
  macroCheckLoopKernel<<<1, 1, 0, stream>>>(
      abort, observedIterations, observedStop, maxIterations);
  return cudaGetLastError();
}

cudaError_t launchMacroTimeoutLoop(
    AbortDevice abort,
    int* observedIterations,
    int maxIterations,
    cudaStream_t stream) {
  macroTimeoutLoopKernel<<<1, 1, 0, stream>>>(
      abort, observedIterations, maxIterations);
  return cudaGetLastError();
}

cudaError_t launchMacroReturnValue(
    AbortDevice abort,
    int* observedReturn,
    cudaStream_t stream) {
  macroReturnValueKernel<<<1, 1, 0, stream>>>(abort, observedReturn);
  return cudaGetLastError();
}

cudaError_t launchMacroBreakInIfElse(
    AbortDevice abort,
    int* observedIterations,
    int* observedFallback,
    int maxIterations,
    cudaStream_t stream) {
  macroBreakInIfElseKernel<<<1, 1, 0, stream>>>(
      abort, observedIterations, observedFallback, maxIterations);
  return cudaGetLastError();
}

} // namespace comms::fault_tolerance::testing
