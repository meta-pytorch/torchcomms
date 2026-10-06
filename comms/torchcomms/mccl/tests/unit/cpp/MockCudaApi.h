// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <gmock/gmock.h>

#include "comms/torchcomms/device/cuda/CudaApi.hpp"

namespace torch::comms::test {

// Mock CudaApi so unit tests don't require a real GPU.
// NiceMock returns default values (cudaSuccess / nullptr) for all methods.
class MockCudaApi : public CudaApi {
 public:
  MOCK_METHOD(cudaError_t, setDevice, (int), (override));
  MOCK_METHOD(cudaError_t, getDevice, (int*), (override));
  MOCK_METHOD(
      cudaError_t,
      getDeviceProperties,
      (cudaDeviceProp*, int),
      (override));
  MOCK_METHOD(cudaError_t, memGetInfo, (size_t*, size_t*), (override));
  MOCK_METHOD(cudaError_t, getDeviceCount, (int*), (override));
  MOCK_METHOD(cudaError_t, getStreamPriorityRange, (int*, int*), (override));
  MOCK_METHOD(
      cudaError_t,
      streamCreateWithPriority,
      (cudaStream_t*, unsigned int, int),
      (override));
  MOCK_METHOD(cudaError_t, streamDestroy, (cudaStream_t), (override));
  MOCK_METHOD(
      cudaError_t,
      streamWaitEvent,
      (cudaStream_t, cudaEvent_t, unsigned int),
      (override));
  MOCK_METHOD(cudaStream_t, getCurrentCUDAStream, (int), (override));
  MOCK_METHOD(cudaError_t, streamSynchronize, (cudaStream_t), (override));
  MOCK_METHOD(
      cudaError_t,
      streamIsCapturing,
      (cudaStream_t, cudaStreamCaptureStatus*),
      (override));
  MOCK_METHOD(
      cudaError_t,
      streamGetCaptureInfo,
      (cudaStream_t, cudaStreamCaptureStatus*, unsigned long long*),
      (override));
  MOCK_METHOD(
      cudaError_t,
      userObjectCreate,
      (cudaUserObject_t*, void*, cudaHostFn_t, unsigned int, unsigned int),
      (override));
  MOCK_METHOD(
      cudaError_t,
      graphRetainUserObject,
      (cudaGraph_t, cudaUserObject_t, unsigned int, unsigned int),
      (override));
  MOCK_METHOD(
      cudaError_t,
      userObjectRelease,
      (cudaUserObject_t, unsigned int),
      (override));
  MOCK_METHOD(
      cudaError_t,
      launchHostFunc,
      (cudaStream_t, cudaHostFn_t, void*),
      (override));
  MOCK_METHOD(
      cudaError_t,
      streamGetCaptureInfo_v2,
      (cudaStream_t,
       cudaStreamCaptureStatus*,
       unsigned long long*,
       cudaGraph_t*,
       const cudaGraphNode_t**,
       size_t*),
      (override));
  MOCK_METHOD(
      cudaError_t,
      threadExchangeStreamCaptureMode,
      (enum cudaStreamCaptureMode*),
      (override));
  MOCK_METHOD(
      cudaError_t,
      hostAlloc,
      (void**, size_t, unsigned int),
      (override));
  MOCK_METHOD(cudaError_t, hostFree, (void*), (override));
  MOCK_METHOD(cudaError_t, malloc, (void**, size_t), (override));
  MOCK_METHOD(cudaError_t, free, (void*), (override));
  MOCK_METHOD(
      cudaError_t,
      memcpy,
      (void*, const void*, size_t, cudaMemcpyKind),
      (override));
  MOCK_METHOD(
      cudaError_t,
      memcpyAsync,
      (void*, const void*, size_t, cudaMemcpyKind, cudaStream_t),
      (override));
  MOCK_METHOD(cudaError_t, eventCreate, (cudaEvent_t*), (override));
  MOCK_METHOD(
      cudaError_t,
      eventCreateWithFlags,
      (cudaEvent_t*, unsigned int),
      (override));
  MOCK_METHOD(cudaError_t, eventDestroy, (cudaEvent_t), (override));
  MOCK_METHOD(
      cudaError_t,
      eventRecord,
      (cudaEvent_t, cudaStream_t),
      (override));
  MOCK_METHOD(
      cudaError_t,
      eventRecordWithFlags,
      (cudaEvent_t, cudaStream_t, unsigned int),
      (override));
  MOCK_METHOD(cudaError_t, eventQuery, (cudaEvent_t), (override));
  MOCK_METHOD(const char*, getErrorString, (cudaError_t), (override));
};

} // namespace torch::comms::test
