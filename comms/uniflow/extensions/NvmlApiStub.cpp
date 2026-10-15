// Copyright (c) Meta Platforms, Inc. and affiliates.

// AMD/ROCm no-op implementation of NvmlApi. NVML is NVIDIA-only and has no HIP
// analog, so this stub reports every query as unsupported and never dlopen's
// libnvidia-ml. The build selects this translation unit instead of NvmlApi.cpp
// on AMD (see drivers/nvml/BUCK), mirroring createCudaApi()'s per-platform TU
// split. Topology discovery degrades gracefully: callers skip NVLink/C2C edges
// when these calls fail, leaving NIC/PCIe + GPU P2P edges (correct for AMD).
//
// A real amdsmi-backed implementation can replace this stub later.

#include "comms/uniflow/drivers/nvml/NvmlApi.h"

namespace uniflow {
namespace {

Err unsupported() {
  return Err(
      ErrCode::NotImplemented, "NVML is not supported on AMD/ROCm builds");
}

} // namespace

Result<int> NvmlApi::deviceCount() {
  return unsupported();
}

Result<NvmlApi::DeviceInfo> NvmlApi::deviceInfo(int) {
  return unsupported();
}

Result<NvmlApi::DevicePairInfo> NvmlApi::devicePairInfo(int, int) {
  return unsupported();
}

Status NvmlApi::nvmlInit() {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetHandleByPciBusId(const char*, nvmlDevice_t*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetHandleByIndex(unsigned int, nvmlDevice_t*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetIndex(nvmlDevice_t, unsigned int*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetNvLinkState(
    nvmlDevice_t,
    unsigned int,
    nvmlEnableState_t*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetNvLinkRemotePciInfo(
    nvmlDevice_t,
    unsigned int,
    nvmlPciInfo_t*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetNvLinkCapability(
    nvmlDevice_t,
    unsigned int,
    nvmlNvLinkCapability_t,
    unsigned int*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetCudaComputeCapability(nvmlDevice_t, int*, int*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetP2PStatus(
    nvmlDevice_t,
    nvmlDevice_t,
    nvmlGpuP2PCapsIndex_t,
    nvmlGpuP2PStatus_t*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetFieldValues(nvmlDevice_t, int, nvmlFieldValue_t*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetGpuFabricInfoV(
    nvmlDevice_t,
    nvmlGpuFabricInfoV_t*) {
  return unsupported();
}

Status NvmlApi::nvmlDeviceGetPlatformInfo(nvmlDevice_t, nvmlPlatformInfo_t*) {
  return unsupported();
}

Status NvmlApi::nvmlSystemGetConfComputeStatus(NvmlCCStatus*) {
  return unsupported();
}

std::shared_ptr<NvmlApi> createNvmlApi() {
  return std::make_shared<NvmlApi>();
}

} // namespace uniflow
