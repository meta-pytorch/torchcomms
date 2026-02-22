// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <gmock/gmock.h>
#include "comms/torchcomms/device/cuda/CudaApi.hpp"

namespace torch::comms::test {

/**
 * Mock implementation of CudaApi using Google Mock.
 * This class provides mock implementations of all CUDA API operations
 * for testing purposes without requiring actual CUDA hardware.
 */
class CudaMock : public CudaApi {
 public:
  ~CudaMock() override = default;

  // Device management
  MOCK_METHOD(cudaError_t, setDevice, (int device), (override));
  MOCK_METHOD(
      cudaError_t,
      getDeviceProperties,
      (cudaDeviceProp * prop, int device),
      (override));
  MOCK_METHOD(
      cudaError_t,
      memGetInfo,
      (size_t* free, size_t* total),
      (override));
  MOCK_METHOD(cudaError_t, getDeviceCount, (int* count), (override));

  // Stream management
  MOCK_METHOD(
      cudaError_t,
      getStreamPriorityRange,
      (int* leastPriority, int* greatestPriority),
      (override));
  MOCK_METHOD(
      cudaError_t,
      streamCreateWithPriority,
      (cudaStream_t * pStream, unsigned int flags, int priority),
      (override));
  MOCK_METHOD(cudaError_t, streamDestroy, (cudaStream_t stream), (override));
  MOCK_METHOD(
      cudaError_t,
      streamWaitEvent,
      (cudaStream_t stream, cudaEvent_t event, unsigned int flags),
      (override));
  MOCK_METHOD(
      cudaStream_t,
      getCurrentCUDAStream,
      (int device_index),
      (override));
  MOCK_METHOD(
      cudaError_t,
      streamSynchronize,
      (cudaStream_t stream),
      (override));
  MOCK_METHOD(
      cudaError_t,
      streamIsCapturing,
      (cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus),
      (override));
  MOCK_METHOD(
      cudaError_t,
      streamGetCaptureInfo,
      (cudaStream_t stream,
       cudaStreamCaptureStatus* pCaptureStatus,
       unsigned long long* pId),
      (override));

  // CUDA Graph and User Object management
  MOCK_METHOD(
      cudaError_t,
      userObjectCreate,
      (cudaUserObject_t * object_out,
       void* ptr,
       cudaHostFn_t destroy,
       unsigned int initialRefcount,
       unsigned int flags),
      (override));
  MOCK_METHOD(
      cudaError_t,
      graphRetainUserObject,
      (cudaGraph_t graph,
       cudaUserObject_t object,
       unsigned int count,
       unsigned int flags),
      (override));
  MOCK_METHOD(
      cudaError_t,
      userObjectRelease,
      (cudaUserObject_t object, unsigned int count),
      (override));
  MOCK_METHOD(
      cudaError_t,
      launchHostFunc,
      (cudaStream_t stream, cudaHostFn_t fn, void* userData),
      (override));
  MOCK_METHOD(
      cudaError_t,
      streamGetCaptureInfo_v2,
      (cudaStream_t stream,
       cudaStreamCaptureStatus* captureStatus_out,
       unsigned long long* id_out,
       cudaGraph_t* graph_out,
       const cudaGraphNode_t** dependencies_out,
       size_t* numDependencies_out),
      (override));
  MOCK_METHOD(
      cudaError_t,
      threadExchangeStreamCaptureMode,
      (enum cudaStreamCaptureMode * mode),
      (override));

  // Memory management
  MOCK_METHOD(cudaError_t, malloc, (void** devPtr, size_t size), (override));
  MOCK_METHOD(cudaError_t, free, (void* devPtr), (override));
  MOCK_METHOD(
      cudaError_t,
      memcpy,
      (void* dst, const void* src, size_t count, cudaMemcpyKind kind),
      (override));
  MOCK_METHOD(
      cudaError_t,
      memcpyAsync,
      (void* dst,
       const void* src,
       size_t count,
       cudaMemcpyKind kind,
       cudaStream_t stream),
      (override));

  // Event management
  MOCK_METHOD(cudaError_t, eventCreate, (cudaEvent_t * event), (override));
  MOCK_METHOD(
      cudaError_t,
      eventCreateWithFlags,
      (cudaEvent_t * event, unsigned int flags),
      (override));
  MOCK_METHOD(cudaError_t, eventDestroy, (cudaEvent_t event), (override));
  MOCK_METHOD(
      cudaError_t,
      eventRecord,
      (cudaEvent_t event, cudaStream_t stream),
      (override));
  MOCK_METHOD(
      cudaError_t,
      eventRecordWithFlags,
      (cudaEvent_t event, cudaStream_t stream, unsigned int flags),
      (override));
  MOCK_METHOD(cudaError_t, eventQuery, (cudaEvent_t event), (override));

  // Error handling
  MOCK_METHOD(const char*, getErrorString, (cudaError_t error), (override));

  /**
   * Set up default behaviors for common CUDA operations.
   * This method configures the mock to return success for most operations
   * and provides reasonable default values for queries.
   */
  void setupDefaultBehaviors();

  /**
   * Reset all mock expectations and call counts.
   */
  void reset();
};

} // namespace torch::comms::test
