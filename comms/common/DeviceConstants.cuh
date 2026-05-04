// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::device {

// Statically define the warp size (unlike warpSize, this can be used in
// constexpr expressions).
// Note: Uses __HIP_PLATFORM_AMD__ (not __HIP_DEVICE_COMPILE__) because this
// constant is needed in both host and device code (e.g., buffer allocation).
#if defined(__HIP_PLATFORM_AMD__)
constexpr uint32_t kWarpSize = 64; // AMD wavefront size
#else
constexpr uint32_t kWarpSize = 32; // NVIDIA warp size
#endif

} // namespace comms::device
