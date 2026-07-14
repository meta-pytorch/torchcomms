// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"

namespace uniflow {

// Convert an integer device address to the CUDA driver pointer type. On NVIDIA
// CUdeviceptr is an integer handle, so the address converts with static_cast.
// After hipification CUdeviceptr is hipDeviceptr_t (a pointer), which requires
// reinterpret_cast from the integer address. Shared by the RDMA copy/dma-buf
// paths so the platform-divergent cast lives in exactly one place.
inline CUdeviceptr toDevicePtr(uint64_t addr) {
#if defined(__HIP_PLATFORM_AMD__)
  return reinterpret_cast<CUdeviceptr>(addr);
#else
  return static_cast<CUdeviceptr>(addr);
#endif
}

} // namespace uniflow
