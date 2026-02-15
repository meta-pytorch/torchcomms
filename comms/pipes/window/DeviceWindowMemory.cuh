// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/window/DeviceWindowSignal.cuh"

namespace comms::pipes {

/**
 * DeviceWindowMemory - Device-side bundle of synchronization primitives
 *
 * Bundles DeviceWindowSignal (and later on, DeviceBarrier) into a single
 * device-side object for convenient passing to CUDA kernels.
 *
 * Returned by WindowMemory::getDeviceWindowMemory() after exchange().
 *
 * This is a lightweight handle (contains only DeviceWindowSignal and eventually
 * DeviceBarrier as well) that can be passed by value to CUDA kernels.
 */
class DeviceWindowMemory {
 public:
  __host__ __device__ DeviceWindowMemory() = default;

  __host__ __device__ explicit DeviceWindowMemory(DeviceWindowSignal signal)
      : signal_(signal) {}

  __host__ __device__ __forceinline__ DeviceWindowSignal& signal() {
    return signal_;
  }

  __host__ __device__ __forceinline__ const DeviceWindowSignal& signal() const {
    return signal_;
  }

 private:
  DeviceWindowSignal signal_;
};

} // namespace comms::pipes
