// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <glog/logging.h>
#include <hip/hip_runtime.h> // @manual=third-party//rocm:amdhip64-lazy

namespace torch::comms {

#define HIP_CHECK(cuda_api, call, err_str)                                \
  do {                                                                    \
    hipError_t status = call;                                             \
    if (status != hipSuccess) {                                           \
      std::stringstream ss;                                               \
      ss << err_str << ": " << cuda_api->getErrorString(status) << " at " \
         << __FILE__ << ":" << __LINE__;                                  \
      throw std::runtime_error(ss.str());                                 \
    }                                                                     \
  } while (0)

// Ignore variant for use in destructors - logs errors instead of throwing
#define HIP_CHECK_IGNORE(hip_api, call, err_str)                          \
  do {                                                                    \
    hipError_t status = call;                                             \
    if (status != hipSuccess) {                                           \
      LOG(ERROR) << "[TC] " << err_str << ": "                            \
                 << hip_api->getErrorString(status) << " at " << __FILE__ \
                 << ":" << __LINE__;                                      \
    }                                                                     \
  } while (0)

/**
 * Abstract interface for HIP API operations.
 * This allows for dependency injection and testing by providing
 * a way to override HIP API calls.
 */
class HipApi {
 public:
  virtual ~HipApi() = default;

  // Device management
  virtual hipError_t setDevice(int device) = 0;
  virtual hipError_t getDeviceProperties(hipDeviceProp_t* prop, int device) = 0;
  virtual hipError_t memGetInfo(size_t* free, size_t* total) = 0;
  virtual hipError_t getDeviceCount(int* count) = 0;

  // Stream management
  virtual hipError_t getStreamPriorityRange(
      int* leastPriority,
      int* greatestPriority) = 0;
  virtual hipError_t streamCreateWithPriority(
      hipStream_t* pStream,
      unsigned int flags,
      int priority) = 0;
  virtual hipError_t streamDestroy(hipStream_t stream) = 0;
  virtual hipError_t
  streamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) = 0;
  virtual hipStream_t getCurrentHIPStreamMasqueradingAsCUDA(
      int device_index) = 0;
  virtual hipError_t streamSynchronize(hipStream_t stream) = 0;
  virtual hipError_t threadExchangeStreamCaptureMode(
      enum hipStreamCaptureMode* mode) = 0;

  // Memory management
  virtual hipError_t malloc(void** devPtr, size_t size) = 0;
  virtual hipError_t free(void* devPtr) = 0;
  virtual hipError_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      hipMemcpyKind kind,
      hipStream_t stream) = 0;

  // Event management
  virtual hipError_t eventCreate(hipEvent_t* event) = 0;
  virtual hipError_t eventDestroy(hipEvent_t event) = 0;
  virtual hipError_t eventRecord(hipEvent_t event, hipStream_t stream) = 0;
  virtual hipError_t eventQuery(hipEvent_t event) = 0;

  // Error handling
  virtual const char* getErrorString(hipError_t error) = 0;
};

/**
 * Default implementation that calls the underlying CUDA APIs directly.
 */
class DefaultHipApi : public HipApi {
 public:
  ~DefaultHipApi() override = default;

  // Device management
  hipError_t setDevice(int device) override;
  hipError_t getDeviceProperties(hipDeviceProp_t* prop, int device) override;
  hipError_t memGetInfo(size_t* free, size_t* total) override;
  hipError_t getDeviceCount(int* count) override;

  // Stream management
  virtual hipError_t getStreamPriorityRange(
      int* leastPriority,
      int* greatestPriority) override;
  virtual hipError_t streamCreateWithPriority(
      hipStream_t* pStream,
      unsigned int flags,
      int priority) override;
  hipError_t streamDestroy(hipStream_t stream) override;
  hipError_t streamWaitEvent(
      hipStream_t stream,
      hipEvent_t event,
      unsigned int flags) override;
  hipStream_t getCurrentHIPStreamMasqueradingAsCUDA(int device_index) override;
  hipError_t streamSynchronize(hipStream_t stream) override;
  hipError_t threadExchangeStreamCaptureMode(
      enum hipStreamCaptureMode* mode) override;

  // Memory management
  hipError_t malloc(void** devPtr, size_t size) override;
  hipError_t free(void* devPtr) override;
  hipError_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      hipMemcpyKind kind,
      hipStream_t stream) override;

  // Event management
  hipError_t eventCreate(hipEvent_t* event) override;
  hipError_t eventDestroy(hipEvent_t event) override;
  hipError_t eventRecord(hipEvent_t event, hipStream_t stream) override;
  hipError_t eventQuery(hipEvent_t event) override;

  // Error handling
  const char* getErrorString(hipError_t error) override;
};

} // namespace torch::comms
