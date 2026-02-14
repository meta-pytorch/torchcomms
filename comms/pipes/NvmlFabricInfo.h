// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstring>

#include <dlfcn.h>

#include <cuda_runtime.h>

namespace comms::pipes {

/**
 * MNNVL fabric info for a single GPU.
 *
 * On GB200 NVLink racks, GPUs in the same NVLink domain share a clusterUuid
 * and cliqueId.  Two GPUs are NVLink-connected if and only if both have
 * valid fabric info and their clusterUuid + cliqueId match.
 *
 * On H100 or when NVML is unavailable, @c available will be false.
 */
struct NvmlFabricInfo {
  static constexpr int kUuidLen = 16;

  char clusterUuid[kUuidLen]{};
  unsigned int cliqueId{0};
  bool available{false};
};

namespace detail {

// ---------------------------------------------------------------------------
// Lightweight NVML wrapper loaded via dlopen.
// Only the symbols needed for fabric-info queries are resolved.  Gracefully
// degrades when libnvidia-ml.so.1 is unavailable (H100 without NVML).
// This follows the same dlopen pattern as NCCL (see nvmlwrap.cc).
// ---------------------------------------------------------------------------

constexpr int kNvmlBusIdLen = 80;

constexpr const char* kNvmlLibName = "libnvidia-ml.so.1";
constexpr const char* kNvmlInitSym = "nvmlInit_v2";
constexpr const char* kNvmlGetHandleByPciBusIdSym =
    "nvmlDeviceGetHandleByPciBusId_v2";
constexpr const char* kNvmlGetGpuFabricInfoVSym = "nvmlDeviceGetGpuFabricInfoV";

enum NvmlReturn { NVML_SUCCESS = 0 };
enum NvmlGpuFabricState {
  NVML_GPU_FABRIC_STATE_NOT_SUPPORTED = 0,
  NVML_GPU_FABRIC_STATE_NOT_STARTED = 1,
  NVML_GPU_FABRIC_STATE_IN_PROGRESS = 2,
  NVML_GPU_FABRIC_STATE_COMPLETED = 3,
};

struct NvmlGpuFabricInfoV {
  unsigned int version;
  char clusterUuid[NvmlFabricInfo::kUuidLen];
  unsigned int cliqueId;
  unsigned int state; // NvmlGpuFabricState
  unsigned int healthMask;
};

using NvmlDevice = void*;
using NvmlInitFn = int (*)();
using NvmlGetHandleByPciBusIdFn = int (*)(const char*, NvmlDevice*);
using NvmlGetGpuFabricInfoVFn = int (*)(NvmlDevice, NvmlGpuFabricInfoV*);

struct NvmlFabricApi {
  NvmlInitFn init{nullptr};
  NvmlGetHandleByPciBusIdFn getHandleByPciBusId{nullptr};
  NvmlGetGpuFabricInfoVFn getGpuFabricInfoV{nullptr};
  bool available{false};
};

inline NvmlFabricApi load_nvml_fabric_api() {
  NvmlFabricApi api;
  void* lib = dlopen(kNvmlLibName, RTLD_NOW);
  if (!lib) {
    return api;
  }
  api.init = reinterpret_cast<NvmlInitFn>(dlsym(lib, kNvmlInitSym));
  api.getHandleByPciBusId = reinterpret_cast<NvmlGetHandleByPciBusIdFn>(
      dlsym(lib, kNvmlGetHandleByPciBusIdSym));
  api.getGpuFabricInfoV = reinterpret_cast<NvmlGetGpuFabricInfoVFn>(
      dlsym(lib, kNvmlGetGpuFabricInfoVSym));
  if (api.init && api.getHandleByPciBusId && api.getGpuFabricInfoV) {
    if (api.init() == NVML_SUCCESS) {
      api.available = true;
    }
  }
  // Intentionally not calling dlclose — keep library in memory (same as NCCL).
  return api;
}

} // namespace detail

/**
 * Query MNNVL fabric info for the given CUDA device ordinal.
 *
 * Opens libnvidia-ml.so.1 via dlopen (once, on first call) and queries
 * nvmlDeviceGetGpuFabricInfoV.  If NVML is unavailable or the GPU does
 * not participate in an NVLink fabric, the returned struct has
 * @c available == false.
 *
 * @param busId       PCI bus ID string for the device (from
 *                    cudaDeviceGetPCIBusId()).  Must be a null-terminated
 *                    string of at least @c detail::kNvmlBusIdLen bytes.
 * @return NvmlFabricInfo with clusterUuid/cliqueId if available.
 */
inline NvmlFabricInfo query_nvml_fabric_info(const char* busId) {
  NvmlFabricInfo info;

  static detail::NvmlFabricApi nvmlApi = detail::load_nvml_fabric_api();
  if (!nvmlApi.available) {
    return info;
  }

  detail::NvmlDevice nvmlDev = nullptr;
  if (nvmlApi.getHandleByPciBusId(busId, &nvmlDev) != detail::NVML_SUCCESS) {
    return info;
  }

  detail::NvmlGpuFabricInfoV fabricInfo{};
  if (nvmlApi.getGpuFabricInfoV(nvmlDev, &fabricInfo) == detail::NVML_SUCCESS &&
      fabricInfo.state == detail::NVML_GPU_FABRIC_STATE_COMPLETED) {
    std::memcpy(
        info.clusterUuid, fabricInfo.clusterUuid, NvmlFabricInfo::kUuidLen);
    info.cliqueId = fabricInfo.cliqueId;
    info.available = true;
  }

  return info;
}

} // namespace comms::pipes
