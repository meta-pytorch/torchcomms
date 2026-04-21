// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>

#include "comms/uniflow/drivers/cuda/CudaApi.h"

namespace uniflow {

/// gmock-based mock for CudaApi.
/// All virtual methods are mocked. Use ON_CALL / EXPECT_CALL to configure
/// behavior. Wrap with testing::NiceMock to suppress warnings for unconfigured
/// methods.
class MockCudaApi : public CudaApi {
 public:
  MOCK_METHOD(Status, setDevice, (int device), (override));
  MOCK_METHOD(Result<int>, getDevice, (), (override));
  MOCK_METHOD(
      Result<bool>,
      deviceCanAccessPeer,
      (int device, int peerDevice),
      (override));
  MOCK_METHOD(Status, deviceEnablePeerAccess, (int peerDevice), (override));
  MOCK_METHOD(Result<int>, getDeviceCount, (), (override));
  MOCK_METHOD(
      Status,
      getDevicePCIBusId,
      (char* pciBusId, int len, int device),
      (override));

  MOCK_METHOD(
      Status,
      memcpyAsync,
      (void* dst,
       const void* src,
       size_t count,
       cudaMemcpyKind kind,
       cudaStream_t stream),
      (override));
#if CUDART_VERSION >= 12080
  MOCK_METHOD(
      Status,
      memcpyBatchAsync,
      (void** dsts,
       void** srcs,
       size_t* sizes,
       size_t count,
       cudaStream_t stream),
      (override));
#endif
  MOCK_METHOD(
      Status,
      memcpyPeerAsync,
      (void* dst,
       int dstDevice,
       const void* src,
       int srcDevice,
       size_t count,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(Status, streamSynchronize, (cudaStream_t stream), (override));

  MOCK_METHOD(Status, eventCreate, (cudaEvent_t * event), (override));
  MOCK_METHOD(
      Status,
      eventRecord,
      (cudaEvent_t event, cudaStream_t stream),
      (override));
  MOCK_METHOD(Result<bool>, eventQuery, (cudaEvent_t event), (override));
  MOCK_METHOD(Status, eventDestroy, (cudaEvent_t event), (override));
};

} // namespace uniflow
