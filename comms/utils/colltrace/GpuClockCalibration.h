// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>

namespace meta::comms::colltrace {

// One-time calibration point mapping globaltimer nanoseconds to system_clock.
// Thread-safe; lazily initialized on first call.
struct GlobaltimerCalibration {
  uint64_t device_ns{};
  std::chrono::system_clock::time_point host_time;

  // Convert a device globaltimer nanosecond value to a system_clock time_point.
  std::chrono::system_clock::time_point toWallClock(uint64_t gpu_ns) const;

  // Get the process-global calibration singleton.
  static const GlobaltimerCalibration& get();
};

// Launch a single-thread kernel that writes globaltimer() to *out.
cudaError_t launchReadGlobaltimer(cudaStream_t stream, uint64_t* out);

} // namespace meta::comms::colltrace
