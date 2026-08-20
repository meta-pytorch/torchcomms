// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <cstdio>

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/common/fault_tolerance/AbortMacros.cuh"
#include "comms/prims/transport/amd/HipHostCompat.h"

namespace comms::prims {

using AbortDevice = comms::fault_tolerance::AbortDevice;

__device__ __forceinline__ uint64_t gpu_clock64() {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  return wall_clock64();
#elif defined(__CUDA_ARCH__)
  return clock64();
#else
  return 0;
#endif
}

} // namespace comms::prims

// The FT_ABORT_* checks live in comms/common/fault_tolerance/AbortMacros.cuh
// and are re-exported here so Prims device code keeps a single include for the
// abort handle, the device clock, and the checks.
