// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/GpuClockCalibration.h"

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// simple fprintf-based check for host code.
#define CUDA_CHECK_CC(cmd)             \
  do {                                 \
    auto _err = (cmd);                 \
    if (_err != cudaSuccess) {         \
      fprintf(                         \
          stderr,                      \
          "CUDA error %s:%d %s: %s\n", \
          __FILE__,                    \
          __LINE__,                    \
          #cmd,                        \
          cudaGetErrorString(_err));   \
      abort();                         \
    }                                  \
  } while (0)

namespace meta::comms::colltrace {

std::chrono::system_clock::time_point GlobaltimerCalibration::toWallClock(
    uint64_t gpu_ns) const {
  auto delta_ns =
      static_cast<int64_t>(gpu_ns) - static_cast<int64_t>(device_ns);
  return host_time + std::chrono::nanoseconds(delta_ns);
}

std::chrono::system_clock::time_point GlobaltimerCalibration::toWallClock(
    uint32_t gpu_ticks_low32) const {
  auto host_now = std::chrono::system_clock::now();
  int64_t elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(host_now - host_time)
          .count();
  uint64_t now_device_ns = device_ns + static_cast<uint64_t>(elapsed_ns);
  uint64_t now_ticks = now_device_ns >> US_TICK_TIMESTAMP_SHIFT;

  // Events are always written before they're read, so the delta is
  // semantically non-positive. Allow a small future window for clock
  // skew (host/device drift, or race between event write and `now`
  // read); anything beyond it is interpreted as a past wrap. This
  // gives ~71 min - kFutureToleranceTicks of past lookback instead of
  // ~36 min from a symmetric ±2^31 split.
  constexpr uint32_t kFutureToleranceTicks =
      (10ULL * 1'000'000'000ULL) >> US_TICK_TIMESTAMP_SHIFT; // ~10s
  uint32_t udelta =
      gpu_ticks_low32 - static_cast<uint32_t>(now_ticks); // mod 2^32
  int64_t delta = udelta <= kFutureToleranceTicks
      ? static_cast<int64_t>(udelta)
      : static_cast<int64_t>(udelta) - (int64_t{1} << 32);
  uint64_t full_ticks = now_ticks + static_cast<uint64_t>(delta);

  int64_t delta_ns =
      static_cast<int64_t>(full_ticks << US_TICK_TIMESTAMP_SHIFT) -
      static_cast<int64_t>(device_ns);
  return host_time + std::chrono::nanoseconds(delta_ns);
}

/* static */ const GlobaltimerCalibration& GlobaltimerCalibration::get() {
  static const GlobaltimerCalibration instance = [] {
    GlobaltimerCalibration cal{};

    uint64_t* mapped_ptr = nullptr;
    CUDA_CHECK_CC(cudaHostAlloc(
        reinterpret_cast<void**>(&mapped_ptr),
        sizeof(uint64_t),
        cudaHostAllocDefault));
    *mapped_ptr = 0;

    (void)launchReadGlobaltimer(nullptr, mapped_ptr);
    CUDA_CHECK_CC(cudaDeviceSynchronize());

    cal.device_ns = *mapped_ptr;
    cal.host_time = std::chrono::system_clock::now();

    CUDA_CHECK_CC(cudaFreeHost(mapped_ptr));
    return cal;
  }();
  return instance;
}

#undef CUDA_CHECK_CC

} // namespace meta::comms::colltrace
