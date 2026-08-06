// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/fault_tolerance/tests/AbortDeviceTest.cuh"

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

} // namespace comms::fault_tolerance::testing
