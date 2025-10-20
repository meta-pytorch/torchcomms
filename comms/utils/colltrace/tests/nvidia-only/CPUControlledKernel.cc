// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/tests/nvidia-only/CPUControlledKernel.h"

#include <folly/stop_watch.h>

#include "comms/utils/checks.h"

namespace meta::comms::colltrace {

CPUControlledKernel::CPUControlledKernel(cudaStream_t stream)
    : stream_(stream) {
  CUDA_CHECK(cudaMallocManaged(&sharedFlag_, sizeof(KernelFlag)));
}

CPUControlledKernel::~CPUControlledKernel() {
  if (sharedFlag_) {
    endKernel();
    // Free the shared flag regardless of whether the kernel is terminated
    // successfully after the timeout
    CUDA_CHECK(cudaFree(const_cast<KernelFlag*>(sharedFlag_)));
    sharedFlag_ = nullptr;
  }
}

void CPUControlledKernel::launch() {
  if (kernelLaunched_) {
    return;
  }
  *sharedFlag_ = KERNEL_FLAG_SCHEDULED;
  void* args[] = {&sharedFlag_};
  CUDA_CHECK(cudaLaunchKernel(
      (void*)waitCPUKernel, dim3(1), dim3(1), args, 0, stream_));
  kernelLaunched_ = true;
}

KernelFlag CPUControlledKernel::getKernelFlagValue() const {
  return *sharedFlag_;
}

bool CPUControlledKernel::waitKernelFlagValue(
    KernelFlag value,
    std::chrono::milliseconds timeout) const {
  folly::stop_watch<std::chrono::milliseconds> timer;
  while (*sharedFlag_ != value && timer.elapsed() < timeout) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return *sharedFlag_ == value;
}

bool CPUControlledKernel::endKernel(std::chrono::milliseconds timeout) {
  // Can't end a kernel before it is launched.
  if (!kernelLaunched_) {
    return false;
  }
  // Return if the kernel is already terminated
  if (*sharedFlag_ == KERNEL_FLAG_UNSET) {
    return true;
  }
  folly::stop_watch<std::chrono::milliseconds> timer;

  // If the kernel has not signaled start, wait for it to do so before settting
  // the flag to terminate
  if (*sharedFlag_ == KERNEL_FLAG_SCHEDULED) {
    auto valueGet =
        waitKernelFlagValue(KERNEL_FLAG_STARTED, timeout - timer.elapsed());
    if (!valueGet) {
      // If the kernel has not signaled start, return due to timeout
      return false;
    }
  }
  if (*sharedFlag_ == KERNEL_FLAG_STARTED) {
    // Signal the kernel to terminate if started
    *sharedFlag_ = KERNEL_FLAG_TERMINATE;
  }
  return waitKernelFlagValue(KERNEL_FLAG_UNSET, timeout - timer.elapsed());
}

} // namespace meta::comms::colltrace
