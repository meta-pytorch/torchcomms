// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/CudaRAII.h"
#include "comms/utils/checks.h"

namespace meta::comms {

StreamCaptureModeGuard::StreamCaptureModeGuard(
    cudaStreamCaptureMode desiredMode)
    : prevMode_(desiredMode) {
  CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&prevMode_));
}

void StreamCaptureModeGuard::init() {
  FB_CUDACHECKTHROW(exchangeFn_(ctx_, &prevMode_));
}

StreamCaptureModeGuard::~StreamCaptureModeGuard() {
  if (exchangeFn_) {
    CUDA_CHECK_WITH_IGNORE(
        exchangeFn_(ctx_, &prevMode_),
        cudaErrorCudartUnloading,
        cudaErrorContextIsDestroyed);
  } else {
    CUDA_CHECK_WITH_IGNORE(
        cudaThreadExchangeStreamCaptureMode(&prevMode_),
        cudaErrorCudartUnloading,
        cudaErrorContextIsDestroyed);
  }
}

} // namespace meta::comms
