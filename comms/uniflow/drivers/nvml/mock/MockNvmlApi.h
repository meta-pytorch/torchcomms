// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>

#include "comms/uniflow/drivers/nvml/NvmlApi.h"

namespace uniflow {

/// gmock-based mock for NvmlApi.
/// All virtual methods are mocked. Use ON_CALL / EXPECT_CALL to configure
/// behavior. Wrap with testing::NiceMock to suppress warnings for unconfigured
/// methods.
class MockNvmlApi : public NvmlApi {
 public:
  MOCK_METHOD(Result<int>, deviceCount, (), (override));
  MOCK_METHOD(Result<DeviceInfo>, deviceInfo, (int dev), (override));
  MOCK_METHOD(
      Result<DevicePairInfo>,
      devicePairInfo,
      (int a, int b),
      (override));

  MOCK_METHOD(Status, nvmlInit, (), (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetHandleByPciBusId,
      (const char* pciBusId, nvmlDevice_t* device),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetHandleByIndex,
      (unsigned int index, nvmlDevice_t* device),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetIndex,
      (nvmlDevice_t device, unsigned int* index),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetNvLinkState,
      (nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetNvLinkRemotePciInfo,
      (nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetNvLinkCapability,
      (nvmlDevice_t device,
       unsigned int link,
       nvmlNvLinkCapability_t capability,
       unsigned int* capResult),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetCudaComputeCapability,
      (nvmlDevice_t device, int* major, int* minor),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetP2PStatus,
      (nvmlDevice_t device1,
       nvmlDevice_t device2,
       nvmlGpuP2PCapsIndex_t p2pIndex,
       nvmlGpuP2PStatus_t* p2pStatus),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetFieldValues,
      (nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetGpuFabricInfoV,
      (nvmlDevice_t device, nvmlGpuFabricInfoV_t* info),
      (override));

  MOCK_METHOD(
      Status,
      nvmlDeviceGetPlatformInfo,
      (nvmlDevice_t device, nvmlPlatformInfo_t* platformInfo),
      (override));

  MOCK_METHOD(
      Status,
      nvmlSystemGetConfComputeStatus,
      (NvmlCCStatus * state),
      (override));
};

} // namespace uniflow
