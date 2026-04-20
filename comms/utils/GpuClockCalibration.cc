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

/* static */ const GlobaltimerCalibration& GlobaltimerCalibration::get() {
  static const GlobaltimerCalibration instance = [] {
    GlobaltimerCalibration cal{};

    uint64_t* mapped_ptr = nullptr;
    CUDA_CHECK_CC(cudaHostAlloc(
        reinterpret_cast<void**>(&mapped_ptr),
        sizeof(uint64_t),
        cudaHostAllocDefault));
    *mapped_ptr = 0;

    launchReadGlobaltimer(nullptr, mapped_ptr);
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
