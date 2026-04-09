// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>

#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"

namespace uniflow {

/// gmock-based mock for CudaDriverApi.
/// All virtual methods are mocked. Use ON_CALL / EXPECT_CALL to configure
/// behavior. Wrap with testing::NiceMock to suppress warnings for unconfigured
/// methods.
class MockCudaDriverApi : public CudaDriverApi {
 public:
  MOCK_METHOD(Status, init, (), (override));

  // --- Device ---
  MOCK_METHOD(
      Status,
      cuDeviceGet,
      (CUdevice * device, int ordinal),
      (override));
  MOCK_METHOD(
      Status,
      cuDeviceGetAttribute,
      (int* pi, CUdevice_attribute attrib, CUdevice dev),
      (override));

  // --- Error ---
  MOCK_METHOD(
      Status,
      cuGetErrorString,
      (CUresult error, const char** pStr),
      (override));
  MOCK_METHOD(
      Status,
      cuGetErrorName,
      (CUresult error, const char** pStr),
      (override));

  // --- cuMem VMM ---
  MOCK_METHOD(
      Status,
      cuMemCreate,
      (CUmemGenericAllocationHandle * handle,
       size_t size,
       const CUmemAllocationProp* prop,
       unsigned long long flags),
      (override));
  MOCK_METHOD(
      Status,
      cuMemRelease,
      (CUmemGenericAllocationHandle handle),
      (override));
  MOCK_METHOD(
      Status,
      cuMemAddressReserve,
      (CUdeviceptr * ptr,
       size_t size,
       size_t alignment,
       CUdeviceptr addr,
       unsigned long long flags),
      (override));
  MOCK_METHOD(
      Status,
      cuMemAddressFree,
      (CUdeviceptr ptr, size_t size),
      (override));
  MOCK_METHOD(
      Status,
      cuMemMap,
      (CUdeviceptr ptr,
       size_t size,
       size_t offset,
       CUmemGenericAllocationHandle handle,
       unsigned long long flags),
      (override));
  MOCK_METHOD(Status, cuMemUnmap, (CUdeviceptr ptr, size_t size), (override));
  MOCK_METHOD(
      Status,
      cuMemSetAccess,
      (CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count),
      (override));
  MOCK_METHOD(
      Status,
      cuMemGetAllocationGranularity,
      (size_t* granularity,
       const CUmemAllocationProp* prop,
       CUmemAllocationGranularity_flags option),
      (override));
  MOCK_METHOD(
      Status,
      cuMemExportToShareableHandle,
      (void* shareableHandle,
       CUmemGenericAllocationHandle handle,
       CUmemAllocationHandleType handleType,
       unsigned long long flags),
      (override));
  MOCK_METHOD(
      Status,
      cuMemImportFromShareableHandle,
      (CUmemGenericAllocationHandle * handle,
       void* osHandle,
       CUmemAllocationHandleType shHandleType),
      (override));
  MOCK_METHOD(
      Status,
      cuMemGetHandleForAddressRange,
      (void* handle,
       CUdeviceptr dptr,
       size_t size,
       CUmemRangeHandleType handleType,
       unsigned long long flags),
      (override));
  MOCK_METHOD(
      Status,
      cuMemRetainAllocationHandle,
      (CUmemGenericAllocationHandle * handle, void* addr),
      (override));
  MOCK_METHOD(
      Status,
      cuMemGetAddressRange_v2,
      (CUdeviceptr * pbase, size_t* psize, CUdeviceptr dptr),
      (override));

  // --- supported APIs ---
  MOCK_METHOD(Result<bool>, isDmaBufSupported, (int cudaDev), (override));
  MOCK_METHOD(Result<bool>, isCuMemSupported, (), (override));
  MOCK_METHOD(
      Result<CUmemAllocationHandleType>,
      getCuMemHandleType,
      (),
      (override));
};

} // namespace uniflow
