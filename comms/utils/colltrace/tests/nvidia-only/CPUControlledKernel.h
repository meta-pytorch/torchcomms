// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <chrono>

#include "comms/utils/colltrace/tests/nvidia-only/CPUControlledKernel.cuh"

namespace meta::comms::colltrace {

class CPUControlledKernel {
 public:
  CPUControlledKernel(cudaStream_t stream);
  ~CPUControlledKernel();

  KernelFlag getKernelFlagValue() const;

  void launch();

  bool waitKernelFlagValue(
      KernelFlag value,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{
          1000}) const;

  bool endKernel(
      std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});

 private:
  cudaStream_t stream_{nullptr};
  bool kernelLaunched_{false};
  volatile KernelFlag* sharedFlag_{nullptr};
};

} // namespace meta::comms::colltrace
