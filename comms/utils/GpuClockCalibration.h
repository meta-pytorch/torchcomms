// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#if not defined(__CUDACC__) and not defined(__HIPCC__)
#include <chrono>
#endif // not defined(__CUDACC__) and not defined(__HIPCC__)

// Globaltimer (ns) is reduced to ~1024ns ticks (`>> shift`) to fit in 32
// bits when packed into HRDWEntry. 32 bits of 1024ns ticks wraps every
// ~73 minutes; the host reader reconstructs the full value via
// GlobaltimerCalibration::toWallClock(uint32_t).
#define US_TICK_TIMESTAMP_SHIFT 10

namespace meta::comms::colltrace {

// One-time calibration point mapping globaltimer nanoseconds to system_clock.
// Thread-safe; lazily initialized on first call.
#if not defined(__CUDACC__) and not defined(__HIPCC__)
struct GlobaltimerCalibration {
  uint64_t device_ns{};
  std::chrono::system_clock::time_point host_time;

  // Convert a device globaltimer nanosecond value to a system_clock time_point.
  std::chrono::system_clock::time_point toWallClock(uint64_t gpu_ns) const;

  // Convert a packed 32-bit timestamp (globaltimer_ns >> 10, i.e. ~1024ns
  // ticks, as written by HRDWRingBuffer) into a system_clock time_point.
  // Reconstructs the high 32 bits against the current wall-clock-derived
  // "now" device time. Events are assumed to be in the past; a small
  // future window (~10s) is allowed for host/device clock skew, and
  // anything beyond is treated as a past wrap. Correct as long as the
  // event is within ~73 minutes in the past of "now".
  std::chrono::system_clock::time_point toWallClock(
      uint32_t gpu_ticks_low32) const;

  // Get the process-global calibration singleton.
  static const GlobaltimerCalibration& get();
};

// Launch a single-thread kernel that writes globaltimer() to *out.
cudaError_t launchReadGlobaltimer(cudaStream_t stream, uint64_t* out);
#endif // not defined(__CUDACC__) and not defined(__HIPCC__)

#if defined(__CUDACC__) || defined(__HIPCC__)
// Device-side globaltimer read. Returns nanoseconds since device boot.
__device__ __forceinline__ uint64_t readGlobaltimer() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return wall_clock64();
#else
  uint64_t timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
#endif
}
#endif

} // namespace meta::comms::colltrace
