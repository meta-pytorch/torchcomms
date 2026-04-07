// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>

#include "comms/utils/colltrace/GpuClockCalibration.h"

// simple fprintf-based check for this .cu file.
#define CUDA_CHECK_CU(cmd)             \
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

namespace {

__global__ void readGlobaltimerKernel(uint64_t* out) {
  *out = readGlobaltimer();
}

} // namespace

cudaError_t launchReadGlobaltimer(cudaStream_t stream, uint64_t* out) {
  readGlobaltimerKernel<<<1, 1, 0, stream>>>(out);
  return cudaGetLastError();
}

std::chrono::system_clock::time_point GlobaltimerCalibration::toWallClock(
    uint64_t gpu_ns) const {
  // globaltimer values are in nanoseconds. Compute signed delta to handle
  // timestamps both before and after the calibration point.
  auto delta_ns =
      static_cast<int64_t>(gpu_ns) - static_cast<int64_t>(device_ns);
  return host_time + std::chrono::nanoseconds(delta_ns);
}

/* static */ const GlobaltimerCalibration& GlobaltimerCalibration::get() {
  static const GlobaltimerCalibration instance = [] {
    GlobaltimerCalibration cal{};

    // Allocate a single uint64_t in mapped pinned memory for the kernel to
    // write the globaltimer value.
    uint64_t* mapped_ptr = nullptr;
    CUDA_CHECK_CU(cudaHostAlloc(
        reinterpret_cast<void**>(&mapped_ptr),
        sizeof(uint64_t),
        cudaHostAllocDefault));
    *mapped_ptr = 0;

    // Launch a calibration kernel on the default stream and synchronize.
    launchReadGlobaltimer(nullptr, mapped_ptr);
    CUDA_CHECK_CU(cudaDeviceSynchronize());

    cal.device_ns = *mapped_ptr;
    cal.host_time = std::chrono::system_clock::now();

    CUDA_CHECK_CU(cudaFreeHost(mapped_ptr));
    return cal;
  }();
  return instance;
}

#undef CUDA_CHECK_CU

} // namespace meta::comms::colltrace
