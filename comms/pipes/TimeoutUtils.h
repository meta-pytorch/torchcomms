// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

/**
 * getGpuCyclesPerMs - Get the GPU clock rate in cycles per millisecond
 *
 * Query the GPU's shader clock rate for use with timeout-enabled wait methods.
 *
 * @param device CUDA device ID (default: 0)
 * @return Clock cycles per millisecond
 * @throws std::runtime_error if cudaDeviceGetAttribute fails
 */
inline uint64_t getGpuCyclesPerMs(int device = 0) {
  int clock_rate_khz;
  cudaError_t err =
      cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get GPU clock rate: " +
        std::string(cudaGetErrorString(err)));
  }
  // cudaDevAttrClockRate returns kHz, which equals cycles per millisecond
  return static_cast<uint64_t>(clock_rate_khz);
}

/**
 * makeTimeout - Create a Timeout configuration for the specified device
 *
 * Convenience function that queries the GPU clock rate and creates a
 * Timeout object ready to pass to kernels. Precomputes timeout_cycles
 * on the host side for efficient GPU-side checking.
 *
 * @param timeout_ms Timeout in milliseconds (0 = infinite wait)
 * @param device CUDA device ID (default: 0)
 * @return Timeout configuration with precomputed timeout_cycles
 * @throws std::runtime_error if cudaDeviceGetAttribute fails
 *
 * Example usage:
 *   auto timeout = makeTimeout(1000, deviceId);  // 1 second timeout
 *   myKernel<<<...>>>(timeout, ...);
 */
inline Timeout makeTimeout(uint32_t timeout_ms, int device = 0) {
  if (timeout_ms == 0) {
    return Timeout(); // No timeout
  }
  uint64_t cycles_per_ms = getGpuCyclesPerMs(device);
  uint64_t timeout_cycles = static_cast<uint64_t>(timeout_ms) * cycles_per_ms;
  return Timeout(timeout_cycles);
}

} // namespace comms::pipes
