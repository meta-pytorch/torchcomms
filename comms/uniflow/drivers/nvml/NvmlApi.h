// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/Result.h"
#include "comms/uniflow/drivers/nvml/NvmlCore.h"

namespace uniflow {

/// Each method calls nvmlInit() implicitly to ensure the library is loaded.
/// Thread-safe: all calls are serialized via an internal mutex
class NvmlApi {
 public:
  /// Cached per-device info, populated during nvmlInit().
  struct DeviceInfo {
    nvmlDevice_t handle{nullptr};
    int computeCapabilityMajor{-1};
    int computeCapabilityMinor{-1};
  };

  /// Cached P2P status for a pair of devices, populated during nvmlInit().
  struct DevicePairInfo {
    nvmlGpuP2PStatus_t p2pStatusRead{NVML_P2P_STATUS_UNKNOWN};
    nvmlGpuP2PStatus_t p2pStatusWrite{NVML_P2P_STATUS_UNKNOWN};
  };

  /// Confidential compute status information
  struct NvmlCCStatus {
    bool CCEnabled{false};
    bool multiGpuProtectedPCIE{false};
    bool multiGpuNVLE{false};
  };

  virtual ~NvmlApi() = default;

  /// Returns the number of NVML devices on the system.
  virtual Result<int> deviceCount();
  /// Returns cached info (handle, compute capability) for device index @p dev.
  virtual Result<DeviceInfo> deviceInfo(int dev);
  /// Returns cached P2P read/write status between devices @p a and @p b.
  virtual Result<DevicePairInfo> devicePairInfo(int a, int b);

  virtual Status nvmlInit();

  virtual Status nvmlDeviceGetHandleByPciBusId(
      const char* pciBusId,
      nvmlDevice_t* device);

  virtual Status nvmlDeviceGetHandleByIndex(
      unsigned int index,
      nvmlDevice_t* device);

  virtual Status nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index);

  virtual Status nvmlDeviceGetNvLinkState(
      nvmlDevice_t device,
      unsigned int link,
      nvmlEnableState_t* isActive);

  virtual Status nvmlDeviceGetNvLinkRemotePciInfo(
      nvmlDevice_t device,
      unsigned int link,
      nvmlPciInfo_t* pci);

  virtual Status nvmlDeviceGetNvLinkCapability(
      nvmlDevice_t device,
      unsigned int link,
      nvmlNvLinkCapability_t capability,
      unsigned int* capResult);

  virtual Status nvmlDeviceGetCudaComputeCapability(
      nvmlDevice_t device,
      int* major,
      int* minor);

  virtual Status nvmlDeviceGetP2PStatus(
      nvmlDevice_t device1,
      nvmlDevice_t device2,
      nvmlGpuP2PCapsIndex_t p2pIndex,
      nvmlGpuP2PStatus_t* p2pStatus);

  virtual Status nvmlDeviceGetFieldValues(
      nvmlDevice_t device,
      int valuesCount,
      nvmlFieldValue_t* values);

  virtual Status nvmlDeviceGetGpuFabricInfoV(
      nvmlDevice_t device,
      nvmlGpuFabricInfoV_t* info);

  virtual Status nvmlDeviceGetPlatformInfo(
      nvmlDevice_t device,
      nvmlPlatformInfo_t* platformInfo);

  /// Get the confidential compute status of the system
  virtual Status nvmlSystemGetConfComputeStatus(NvmlCCStatus* state);
};

} // namespace uniflow
