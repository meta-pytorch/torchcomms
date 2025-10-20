// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace meta::comms {

// RAII helper for device buffer pointers
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size);
  ~DeviceBuffer();

  // delete copy constructor
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  // default move constructor

  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void* get() const;

 private:
  void* ptr_{nullptr};
  std::size_t size_{0};
};

// RAII helper for cuda stream
class CudaStream {
 public:
  CudaStream();
  ~CudaStream();

  // delete copy constructor
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  // default move constructor
  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;

  cudaStream_t get() const;

 private:
  cudaStream_t stream_{nullptr};
};

// RAII helper for cuda event
class CudaEvent {
 public:
  CudaEvent();

  ~CudaEvent();

  // delete copy constructor
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  // custom move constructor due to raw pointers won't be automatically moved
  CudaEvent(CudaEvent&& other) noexcept;
  CudaEvent& operator=(CudaEvent&& other) noexcept;

  cudaEvent_t get() const;

 private:
  cudaEvent_t event_{nullptr};
};

} // namespace meta::comms
