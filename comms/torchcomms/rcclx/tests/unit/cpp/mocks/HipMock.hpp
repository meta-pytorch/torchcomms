// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include <hip/hip_runtime.h>

#include "comms/torchcomms/rcclx/HipApi.hpp"

namespace torch {
namespace comms {
namespace test {

class HipMock : public HipApi {
 public:
  MOCK_METHOD(hipError_t, setDevice, (int device), (override));
  MOCK_METHOD(hipError_t, getDeviceCount, (int* count), (override));
  MOCK_METHOD(
      hipError_t,
      getDeviceProperties,
      (hipDeviceProp_t * prop, int device),
      (override));
  MOCK_METHOD(
      hipError_t,
      memGetInfo,
      (size_t* free, size_t* total),
      (override));
  MOCK_METHOD(
      hipError_t,
      streamCreateWithPriority,
      (hipStream_t * stream, unsigned int flags, int priority),
      (override));
  MOCK_METHOD(hipError_t, streamDestroy, (hipStream_t stream), (override));
  MOCK_METHOD(
      hipError_t,
      streamWaitEvent,
      (hipStream_t stream, hipEvent_t event, unsigned int flags),
      (override));
  MOCK_METHOD(
      hipStream_t,
      getCurrentHIPStreamMasqueradingAsCUDA,
      (int device_index),
      (override));
  MOCK_METHOD(hipError_t, streamSynchronize, (hipStream_t stream), (override));
  MOCK_METHOD(
      hipError_t,
      threadExchangeStreamCaptureMode,
      (enum hipStreamCaptureMode * mode),
      (override));
  MOCK_METHOD(
      hipError_t,
      getStreamPriorityRange,
      (int* leastPriority, int* greatestPriority),
      (override));
  MOCK_METHOD(hipError_t, eventCreate, (hipEvent_t * event), (override));

  MOCK_METHOD(hipError_t, eventDestroy, (hipEvent_t event), (override));
  MOCK_METHOD(
      hipError_t,
      eventRecord,
      (hipEvent_t event, hipStream_t stream),
      (override));
  MOCK_METHOD(hipError_t, eventQuery, (hipEvent_t event), (override));

  MOCK_METHOD(hipError_t, malloc, (void** devPtr, size_t size), (override));
  MOCK_METHOD(hipError_t, free, (void* devPtr), (override));

  MOCK_METHOD(
      hipError_t,
      memcpyAsync,
      (void* dst,
       const void* src,
       size_t count,
       hipMemcpyKind kind,
       hipStream_t stream),
      (override));

  MOCK_METHOD(const char*, getErrorString, (hipError_t error), (override));

  // Helper method to set up default behaviors for common operations
  void setupDefaultBehaviors();
};

} // namespace test
} // namespace comms
} // namespace torch
