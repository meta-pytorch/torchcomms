// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/CudaRAII.h"
#include "comms/utils/checks.h"

namespace meta::comms {

DeviceBuffer::DeviceBuffer(std::size_t size) : size_(size) {
  CUDA_CHECK(cudaMalloc(&ptr_, size));
}

DeviceBuffer::~DeviceBuffer() {
  if (ptr_) {
    CUDA_CHECK(cudaFree(ptr_));
  }
}

void* DeviceBuffer::get() const {
  return ptr_;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
  other.size_ = 0;
  other.ptr_ = nullptr;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  ptr_ = other.ptr_;
  size_ = other.size_;
  other.ptr_ = nullptr;
  other.size_ = 0;
  return *this;
}

CudaStream::CudaStream(unsigned int flags) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

CudaStream::~CudaStream() {
  if (stream_) {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  }
}

cudaStream_t CudaStream::get() const {
  return stream_;
}

CudaStream::CudaStream(CudaStream&& other) noexcept {
  stream_ = other.stream_;
  other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    stream_ = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

CudaEvent::CudaEvent() {
  CUDA_CHECK(cudaEventCreate(&event_));
}

CudaEvent::~CudaEvent() {
  if (event_ == nullptr) {
    return;
  }
  // Don't throw an error if the process is exiting (cudaErrorCudartUnloading)
  // This is due to the fact that sometimes we might hold events till global
  // destruction. And it is totally fine to ignore the error in this case.
  CUDA_CHECK_WITH_IGNORE(
      cudaEventDestroy(event_),
      cudaErrorCudartUnloading,
      cudaErrorContextIsDestroyed);
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept {
  event_ = other.event_;
  other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
  if (this != &other) {
    if (event_ != nullptr) {
      CUDA_CHECK(cudaEventDestroy(event_));
    }
    event_ = other.event_;
    other.event_ = nullptr;
  }
  return *this;
}

cudaEvent_t CudaEvent::get() const {
  return event_;
}

} // namespace meta::comms
