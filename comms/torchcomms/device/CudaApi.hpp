// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

namespace torch {
namespace comms {

#define CUDA_CHECK(cuda_api, call, err_str)                               \
  do {                                                                    \
    cudaError_t status = call;                                            \
    if (status != cudaSuccess) {                                          \
      std::stringstream ss;                                               \
      ss << err_str << ": " << cuda_api->getErrorString(status) << " at " \
         << __FILE__ << ":" << __LINE__;                                  \
      throw std::runtime_error(ss.str());                                 \
    }                                                                     \
  } while (0)

/**
 * Abstract interface for CUDA API operations.
 * This allows for dependency injection and testing by providing
 * a way to override CUDA API calls.
 */
class CudaApi {
 public:
  virtual ~CudaApi() = default;

  // Device management
  virtual cudaError_t setDevice(int device) = 0;
  virtual cudaError_t getDeviceProperties(cudaDeviceProp* prop, int device) = 0;
  virtual cudaError_t memGetInfo(size_t* free, size_t* total) = 0;
  virtual cudaError_t getDeviceCount(int* count) = 0;

  // Stream management
  virtual cudaError_t getStreamPriorityRange(
      int* leastPriority,
      int* greatestPriority) = 0;
  virtual cudaError_t streamCreateWithPriority(
      cudaStream_t* pStream,
      unsigned int flags,
      int priority) = 0;
  virtual cudaError_t streamDestroy(cudaStream_t stream) = 0;
  virtual cudaError_t streamWaitEvent(
      cudaStream_t stream,
      cudaEvent_t event,
      unsigned int flags) = 0;
  virtual cudaStream_t getCurrentCUDAStream(int device_index) = 0;
  virtual cudaError_t streamSynchronize(cudaStream_t stream) = 0;
  virtual cudaError_t streamIsCapturing(
      cudaStream_t stream,
      cudaStreamCaptureStatus* pCaptureStatus) = 0;
  virtual cudaError_t streamGetCaptureInfo(
      cudaStream_t stream,
      cudaStreamCaptureStatus* pCaptureStatus,
      unsigned long long* pId) = 0;

  // CUDA Graph and User Object management
  virtual cudaError_t userObjectCreate(
      cudaUserObject_t* object_out,
      void* ptr,
      cudaHostFn_t destroy,
      unsigned int initialRefcount,
      unsigned int flags) = 0;
  virtual cudaError_t graphRetainUserObject(
      cudaGraph_t graph,
      cudaUserObject_t object,
      unsigned int count,
      unsigned int flags) = 0;
  virtual cudaError_t streamGetCaptureInfo_v2(
      cudaStream_t stream,
      cudaStreamCaptureStatus* captureStatus_out,
      unsigned long long* id_out,
      cudaGraph_t* graph_out,
      const cudaGraphNode_t** dependencies_out,
      size_t* numDependencies_out) = 0;
  virtual cudaError_t threadExchangeStreamCaptureMode(
      enum cudaStreamCaptureMode* mode) = 0;

  // Memory management
  virtual cudaError_t malloc(void** devPtr, size_t size) = 0;
  virtual cudaError_t free(void* devPtr) = 0;
  virtual cudaError_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      cudaMemcpyKind kind,
      cudaStream_t stream) = 0;

  // Event management
  virtual cudaError_t eventCreate(cudaEvent_t* event) = 0;
  virtual cudaError_t eventCreateWithFlags(
      cudaEvent_t* event,
      unsigned int flags) = 0;
  virtual cudaError_t eventDestroy(cudaEvent_t event) = 0;
  virtual cudaError_t eventRecord(cudaEvent_t event, cudaStream_t stream) = 0;
  virtual cudaError_t eventQuery(cudaEvent_t event) = 0;

  // Error handling
  virtual const char* getErrorString(cudaError_t error) = 0;
};

/**
 * Default implementation that calls the underlying CUDA APIs directly.
 */
class DefaultCudaApi : public CudaApi {
 public:
  ~DefaultCudaApi() override = default;

  // Device management
  cudaError_t setDevice(int device) override;
  cudaError_t getDeviceProperties(cudaDeviceProp* prop, int device) override;
  cudaError_t memGetInfo(size_t* free, size_t* total) override;
  cudaError_t getDeviceCount(int* count) override;

  // Stream management
  virtual cudaError_t getStreamPriorityRange(
      int* leastPriority,
      int* greatestPriority) override;
  virtual cudaError_t streamCreateWithPriority(
      cudaStream_t* pStream,
      unsigned int flags,
      int priority) override;
  cudaError_t streamDestroy(cudaStream_t stream) override;
  cudaError_t streamWaitEvent(
      cudaStream_t stream,
      cudaEvent_t event,
      unsigned int flags) override;
  cudaStream_t getCurrentCUDAStream(int device_index) override;
  cudaError_t streamSynchronize(cudaStream_t stream) override;
  cudaError_t streamIsCapturing(
      cudaStream_t stream,
      cudaStreamCaptureStatus* pCaptureStatus) override;
  cudaError_t streamGetCaptureInfo(
      cudaStream_t stream,
      cudaStreamCaptureStatus* pCaptureStatus,
      unsigned long long* pId) override;

  // CUDA Graph and User Object management
  cudaError_t userObjectCreate(
      cudaUserObject_t* object_out,
      void* ptr,
      cudaHostFn_t destroy,
      unsigned int initialRefcount,
      unsigned int flags) override;
  cudaError_t graphRetainUserObject(
      cudaGraph_t graph,
      cudaUserObject_t object,
      unsigned int count,
      unsigned int flags) override;
  cudaError_t streamGetCaptureInfo_v2(
      cudaStream_t stream,
      cudaStreamCaptureStatus* captureStatus_out,
      unsigned long long* id_out,
      cudaGraph_t* graph_out,
      const cudaGraphNode_t** dependencies_out,
      size_t* numDependencies_out) override;
  cudaError_t threadExchangeStreamCaptureMode(
      enum cudaStreamCaptureMode* mode) override;

  // Memory management
  cudaError_t malloc(void** devPtr, size_t size) override;
  cudaError_t free(void* devPtr) override;
  cudaError_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      cudaMemcpyKind kind,
      cudaStream_t stream) override;

  // Event management
  cudaError_t eventCreate(cudaEvent_t* event) override;
  cudaError_t eventCreateWithFlags(cudaEvent_t* event, unsigned int flags)
      override;
  cudaError_t eventDestroy(cudaEvent_t event) override;
  cudaError_t eventRecord(cudaEvent_t event, cudaStream_t stream) override;
  cudaError_t eventQuery(cudaEvent_t event) override;

  // Error handling
  const char* getErrorString(cudaError_t error) override;
};

bool deviceSupportsMulticast(int device_idx);

} // namespace comms
} // namespace torch
