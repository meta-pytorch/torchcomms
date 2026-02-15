// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/window/DeviceWindowBarrier.cuh"
#include "comms/pipes/window/DeviceWindowSignal.cuh"

namespace comms::pipes {

/**
 * DeviceWindowMemory - Device-side bundle of synchronization primitives
 *
 * Bundles DeviceWindowSignal and DeviceWindowBarrier into a single device-side
 * object for convenient passing to CUDA kernels.
 *
 * Returned by WindowMemory::getDeviceWindowMemory() after exchange().
 *
 * This is a lightweight handle (contains only DeviceWindowSignal +
 * DeviceWindowBarrier) that can be passed by value to CUDA kernels.
 */
class DeviceWindowMemory {
 public:
  __host__ __device__ DeviceWindowMemory() = default;

  __host__ __device__ explicit DeviceWindowMemory(
      DeviceWindowSignal signal,
      DeviceWindowBarrier barrier)
      : signal_(signal), barrier_(barrier) {}

  __host__ __device__ __forceinline__ DeviceWindowSignal& signal() {
    return signal_;
  }

  __host__ __device__ __forceinline__ const DeviceWindowSignal& signal() const {
    return signal_;
  }

  __host__ __device__ __forceinline__ DeviceWindowBarrier& barrier() {
    return barrier_;
  }

  __host__ __device__ __forceinline__ const DeviceWindowBarrier& barrier()
      const {
    return barrier_;
  }

 private:
  DeviceWindowSignal signal_;
  DeviceWindowBarrier barrier_;
};

} // namespace comms::pipes
