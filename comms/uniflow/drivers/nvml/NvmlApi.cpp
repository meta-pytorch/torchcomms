// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/nvml/NvmlApi.h"
#include "comms/uniflow/drivers/Constants.h"

#if !UNIFLOW_NVML_DIRECT
#include <dlfcn.h>
#endif

#include <array>
#include <mutex>
#include <string>

namespace uniflow {

// ---------------------------------------------------------------------------
// Function pointer declarations
// ---------------------------------------------------------------------------

#if UNIFLOW_NVML_DIRECT
// Static link: constexpr function pointers resolve directly to NVML symbols.
#define UNIFLOW_NVML_FN(name, rettype, arglist) \
  constexpr rettype(*pfn_##name) arglist = ::name;
#else
// Dynamic link: function pointers filled in by dlsym during init.
#define UNIFLOW_NVML_FN(name, rettype, arglist) \
  rettype(*pfn_##name) arglist = nullptr;
#endif

namespace {

// NOLINTBEGIN(facebook-avoid-non-const-global-variables)
UNIFLOW_NVML_FN(nvmlInit, nvmlReturn_t, ())
UNIFLOW_NVML_FN(nvmlInit_v2, nvmlReturn_t, ())
UNIFLOW_NVML_FN(nvmlShutdown, nvmlReturn_t, ())
UNIFLOW_NVML_FN(nvmlDeviceGetCount, nvmlReturn_t, (unsigned int*))
UNIFLOW_NVML_FN(nvmlDeviceGetCount_v2, nvmlReturn_t, (unsigned int*))
UNIFLOW_NVML_FN(
    nvmlDeviceGetHandleByPciBusId,
    nvmlReturn_t,
    (const char* pciBusId, nvmlDevice_t* device))
UNIFLOW_NVML_FN(
    nvmlDeviceGetHandleByIndex,
    nvmlReturn_t,
    (unsigned int index, nvmlDevice_t* device))
UNIFLOW_NVML_FN(
    nvmlDeviceGetIndex,
    nvmlReturn_t,
    (nvmlDevice_t device, unsigned int* index))
UNIFLOW_NVML_FN(nvmlErrorString, const char*, (nvmlReturn_t r))
UNIFLOW_NVML_FN(
    nvmlDeviceGetNvLinkState,
    nvmlReturn_t,
    (nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive))
UNIFLOW_NVML_FN(
    nvmlDeviceGetNvLinkRemotePciInfo,
    nvmlReturn_t,
    (nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci))
UNIFLOW_NVML_FN(
    nvmlDeviceGetNvLinkCapability,
    nvmlReturn_t,
    (nvmlDevice_t device,
     unsigned int link,
     nvmlNvLinkCapability_t capability,
     unsigned int* capResult))
UNIFLOW_NVML_FN(
    nvmlDeviceGetCudaComputeCapability,
    nvmlReturn_t,
    (nvmlDevice_t device, int* major, int* minor))
UNIFLOW_NVML_FN(
    nvmlDeviceGetP2PStatus,
    nvmlReturn_t,
    (nvmlDevice_t device1,
     nvmlDevice_t device2,
     nvmlGpuP2PCapsIndex_t p2pIndex,
     nvmlGpuP2PStatus_t* p2pStatus))
UNIFLOW_NVML_FN(
    nvmlDeviceGetFieldValues,
    nvmlReturn_t,
    (nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values))
// MNNVL support
UNIFLOW_NVML_FN(
    nvmlDeviceGetGpuFabricInfoV,
    nvmlReturn_t,
    (nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo))
UNIFLOW_NVML_FN(
    nvmlDeviceGetPlatformInfo,
    nvmlReturn_t,
    (nvmlDevice_t device, nvmlPlatformInfo_t* platformInfo))
// CC support
UNIFLOW_NVML_FN(
    nvmlSystemGetConfComputeState,
    nvmlReturn_t,
    (nvmlConfComputeSystemState_t * state))
UNIFLOW_NVML_FN(
    nvmlSystemGetConfComputeSettings,
    nvmlReturn_t,
    (nvmlSystemConfComputeSettings_t * settings))

#undef UNIFLOW_NVML_FN

using DeviceInfo = NvmlApi::DeviceInfo;
using DevicePairInfo = NvmlApi::DevicePairInfo;

// Device tables, populated during nvmlInit().
// Read-only after initialization; no locking needed for access.
int g_deviceCount{-1};
std::array<DeviceInfo, kMaxDevices> g_devices{};
std::array<std::array<DevicePairInfo, kMaxDevices>, kMaxDevices>
    g_devicePairs{};

// Serialize NVML calls — following ncclx precedent
// (nvmlwrap.cc: "NVML has had some thread safety bugs").
std::mutex g_mutex;

std::once_flag g_initFlag;
Status g_initStatus{Ok()};
// NOLINTEND(facebook-avoid-non-const-global-variables)

union nvmlCCInfoInternal {
  nvmlConfComputeSystemState_t settingV12020;
  nvmlSystemConfComputeSettings_t settingV12040;
};

/// Convert an nvmlReturn_t to Status, using nvmlErrorString for the message.
Status nvmlRetToStatus(nvmlReturn_t ret, const char* funcName) {
  if (ret == NVML_SUCCESS) {
    return Ok();
  }
  std::string msg = funcName;
  msg += "() failed, ret = ";
  msg += std::to_string(ret);
  if (pfn_nvmlErrorString != nullptr) {
    const char* errStr = pfn_nvmlErrorString(ret);
    if (errStr != nullptr) {
      msg += ": ";
      msg += errStr;
    }
  }
  return Err(ErrCode::InvalidArgument, std::move(msg));
}

/// Call pfn_##name(...) during init; on failure set g_initStatus and return.
#define NVML_INIT_CALL(name, ...)                \
  do {                                           \
    nvmlReturn_t _r = pfn_##name(__VA_ARGS__);   \
    if (_r != NVML_SUCCESS) {                    \
      g_initStatus = nvmlRetToStatus(_r, #name); \
      return;                                    \
    }                                            \
  } while (0)

/// Like NVML_INIT_CALL but for v2-fallback calls where the function pointer
/// and name are runtime expressions (e.g. ternary between v1 and v2 variants).
/// First arg is the name expression, remaining args form the call expression.
#define NVML_INIT_CALL_V2(name, ...)              \
  do {                                            \
    nvmlReturn_t _r = (__VA_ARGS__);              \
    if (_r != NVML_SUCCESS) {                     \
      g_initStatus = nvmlRetToStatus(_r, (name)); \
      return;                                     \
    }                                             \
  } while (0)

void doInit() {
#if !UNIFLOW_NVML_DIRECT
  void* libHandle = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (libHandle == nullptr) {
    g_initStatus =
        Err(ErrCode::DriverError, "Failed to open libnvidia-ml.so.1");
    return;
  }

  // Resolve all symbols at once (missing optional symbols are left as nullptr).
  struct Symbol {
    void** ppfn;
    const char* name;
  };
  Symbol symbols[] = {
      {reinterpret_cast<void**>(&pfn_nvmlInit), "nvmlInit"},
      {reinterpret_cast<void**>(&pfn_nvmlInit_v2), "nvmlInit_v2"},
      {reinterpret_cast<void**>(&pfn_nvmlShutdown), "nvmlShutdown"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetCount), "nvmlDeviceGetCount"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetCount_v2),
       "nvmlDeviceGetCount_v2"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetHandleByPciBusId),
       "nvmlDeviceGetHandleByPciBusId"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetHandleByIndex),
       "nvmlDeviceGetHandleByIndex"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetIndex), "nvmlDeviceGetIndex"},
      {reinterpret_cast<void**>(&pfn_nvmlErrorString), "nvmlErrorString"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetNvLinkState),
       "nvmlDeviceGetNvLinkState"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetNvLinkRemotePciInfo),
       "nvmlDeviceGetNvLinkRemotePciInfo"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetNvLinkCapability),
       "nvmlDeviceGetNvLinkCapability"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetCudaComputeCapability),
       "nvmlDeviceGetCudaComputeCapability"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetP2PStatus),
       "nvmlDeviceGetP2PStatus"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetFieldValues),
       "nvmlDeviceGetFieldValues"},
      // MNNVL support
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetGpuFabricInfoV),
       "nvmlDeviceGetGpuFabricInfoV"},
      {reinterpret_cast<void**>(&pfn_nvmlDeviceGetPlatformInfo),
       "nvmlDeviceGetPlatformInfo"},
      // CC support
      {reinterpret_cast<void**>(&pfn_nvmlSystemGetConfComputeState),
       "nvmlSystemGetConfComputeState"},
      {reinterpret_cast<void**>(&pfn_nvmlSystemGetConfComputeSettings),
       "nvmlSystemGetConfComputeSettings"},
  };
  for (auto& sym : symbols) {
    *sym.ppfn = dlsym(libHandle, sym.name);
  }
  // Deliberately never dlclose — we want the loaded object to remain in
  // memory until the process terminates.
#endif

  // Prefer nvmlInit_v2, fall back to nvmlInit.
#if UNIFLOW_NVML_DIRECT
  bool have_v2 = true;
#else
  // Guard against GCC warning about comparing constexpr to nullptr.
  bool have_v2 = pfn_nvmlInit_v2 != nullptr;
#endif

  NVML_INIT_CALL_V2(
      (have_v2 ? "nvmlInit_v2" : "nvmlInit"),
      (have_v2 ? pfn_nvmlInit_v2 : pfn_nvmlInit)());

  // ---------------------------------------------------------------------------
  // Cache device info (following ncclx ncclNvmlEnsureInitialized pattern).
  // ---------------------------------------------------------------------------

  // Get device count.
  unsigned int ndev = 0;
  NVML_INIT_CALL_V2(
      (have_v2 ? "nvmlDeviceGetCount_v2" : "nvmlDeviceGetCount"),
      (have_v2 ? pfn_nvmlDeviceGetCount_v2 : pfn_nvmlDeviceGetCount)(&ndev));

  g_deviceCount = static_cast<int>(ndev);
  if (g_deviceCount > kMaxDevices) {
    g_initStatus = Err(
        ErrCode::InvalidArgument,
        "nvmlDeviceGetCount() reported " + std::to_string(g_deviceCount) +
            " devices, exceeding kMaxDevices=" + std::to_string(kMaxDevices));
    return;
  }

  // Get handles and compute capabilities for each device.
  for (int a = 0; a < g_deviceCount; a++) {
    NVML_INIT_CALL(nvmlDeviceGetHandleByIndex, a, &g_devices[a].handle);

    NVML_INIT_CALL(
        nvmlDeviceGetCudaComputeCapability,
        g_devices[a].handle,
        &g_devices[a].computeCapabilityMajor,
        &g_devices[a].computeCapabilityMinor);
  }

  // Get P2P read/write status for all device pairs.
  for (int a = 0; a < g_deviceCount; a++) {
    for (int b = 0; b < g_deviceCount; b++) {
      nvmlDevice_t da = g_devices[a].handle;
      nvmlDevice_t db = g_devices[b].handle;

      NVML_INIT_CALL(
          nvmlDeviceGetP2PStatus,
          da,
          db,
          NVML_P2P_CAPS_INDEX_READ,
          &g_devicePairs[a][b].p2pStatusRead);

      NVML_INIT_CALL(
          nvmlDeviceGetP2PStatus,
          da,
          db,
          NVML_P2P_CAPS_INDEX_WRITE,
          &g_devicePairs[a][b].p2pStatusWrite);
    }
  }
}

#undef NVML_INIT_CALL
#undef NVML_INIT_CALL_V2

} // namespace

// ---------------------------------------------------------------------------
// NvmlApi implementation
// ---------------------------------------------------------------------------

#define NVML_ENSURE_INIT()          \
  do {                              \
    auto _s = nvmlInit();           \
    if (_s.hasError()) {            \
      return std::move(_s).error(); \
    }                               \
  } while (0)

#define NVML_CHECK(name, ...)            \
  do {                                   \
    auto _s = pfn_##name(__VA_ARGS__);   \
    if (_s != NVML_SUCCESS) {            \
      return nvmlRetToStatus(_s, #name); \
    }                                    \
  } while (0)

#if UNIFLOW_NVML_DIRECT
#define NVML_CALL(name, ...)                                \
  do {                                                      \
    NVML_ENSURE_INIT();                                     \
    std::lock_guard<std::mutex> _lock(g_mutex);             \
    return nvmlRetToStatus(pfn_##name(__VA_ARGS__), #name); \
  } while (0)
#else
#define NVML_CALL(name, ...)                                       \
  do {                                                             \
    NVML_ENSURE_INIT();                                            \
    if (pfn_##name == nullptr) {                                   \
      return Err(ErrCode::DriverError, #name " symbol not found"); \
    }                                                              \
    std::lock_guard<std::mutex> _lock(g_mutex);                    \
    return nvmlRetToStatus(pfn_##name(__VA_ARGS__), #name);        \
  } while (0)
#endif

Status NvmlApi::nvmlInit() {
  std::call_once(g_initFlag, doInit);
  return g_initStatus;
}

Result<int> NvmlApi::deviceCount() {
  NVML_ENSURE_INIT();
  return g_deviceCount;
}

Result<DeviceInfo> NvmlApi::deviceInfo(int dev) {
  NVML_ENSURE_INIT();
  if (dev < 0 || dev >= g_deviceCount) {
    return Err(
        ErrCode::InvalidArgument,
        "NvmlApi::deviceInfo: dev=" + std::to_string(dev) + " out of range");
  }
  return g_devices[dev];
}

Result<DevicePairInfo> NvmlApi::devicePairInfo(int a, int b) {
  NVML_ENSURE_INIT();
  if (a < 0 || a >= g_deviceCount || b < 0 || b >= g_deviceCount) {
    return Err(
        ErrCode::InvalidArgument,
        "NvmlApi::devicePairInfo: a=" + std::to_string(a) +
            ", b=" + std::to_string(b) + " out of range");
  }
  return g_devicePairs[a][b];
}

Status NvmlApi::nvmlDeviceGetHandleByPciBusId(
    const char* pciBusId,
    nvmlDevice_t* device) {
  NVML_CALL(nvmlDeviceGetHandleByPciBusId, pciBusId, device);
}

Status NvmlApi::nvmlDeviceGetHandleByIndex(
    unsigned int index,
    nvmlDevice_t* device) {
  NVML_ENSURE_INIT();
  if (index >= static_cast<unsigned int>(g_deviceCount)) {
    return Err(
        ErrCode::InvalidArgument,
        "nvmlDeviceGetHandleByIndex: index out of range");
  }
  *device = g_devices[index].handle;
  return Ok();
}

Status NvmlApi::nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) {
  NVML_ENSURE_INIT();
  for (int d = 0; d < g_deviceCount; d++) {
    if (g_devices[d].handle == device) {
      *index = d;
      return Ok();
    }
  }
  return Err(ErrCode::InvalidArgument, "nvmlDeviceGetIndex: device not found");
}

Status NvmlApi::nvmlDeviceGetNvLinkState(
    nvmlDevice_t device,
    unsigned int link,
    nvmlEnableState_t* isActive) {
  NVML_CALL(nvmlDeviceGetNvLinkState, device, link, isActive);
}

Status NvmlApi::nvmlDeviceGetNvLinkRemotePciInfo(
    nvmlDevice_t device,
    unsigned int link,
    nvmlPciInfo_t* pci) {
  NVML_CALL(nvmlDeviceGetNvLinkRemotePciInfo, device, link, pci);
}

Status NvmlApi::nvmlDeviceGetNvLinkCapability(
    nvmlDevice_t device,
    unsigned int link,
    nvmlNvLinkCapability_t capability,
    unsigned int* capResult) {
  NVML_CALL(nvmlDeviceGetNvLinkCapability, device, link, capability, capResult);
}

Status NvmlApi::nvmlDeviceGetCudaComputeCapability(
    nvmlDevice_t device,
    int* major,
    int* minor) {
  NVML_ENSURE_INIT();

  for (int d = 0; d < g_deviceCount; d++) {
    if (device == g_devices[d].handle) {
      *major = g_devices[d].computeCapabilityMajor;
      *minor = g_devices[d].computeCapabilityMinor;
      return Ok();
    }
  }
  return Err(
      ErrCode::InvalidArgument,
      "nvmlDeviceGetCudaComputeCapability: device not found");
}

Status NvmlApi::nvmlDeviceGetP2PStatus(
    nvmlDevice_t device1,
    nvmlDevice_t device2,
    nvmlGpuP2PCapsIndex_t p2pIndex,
    nvmlGpuP2PStatus_t* p2pStatus) {
  if (p2pIndex == NVML_P2P_CAPS_INDEX_READ ||
      p2pIndex == NVML_P2P_CAPS_INDEX_WRITE) {
    NVML_ENSURE_INIT();
    int a = -1, b = -1;
    for (int d = 0; d < g_deviceCount; d++) {
      if (device1 == g_devices[d].handle) {
        a = d;
      }
      if (device2 == g_devices[d].handle) {
        b = d;
      }
    }
    if (a == -1 || b == -1) {
      return Err(
          ErrCode::InvalidArgument,
          "nvmlDeviceGetP2PStatus: device1 or device2 not found");
    }
    if (p2pIndex == NVML_P2P_CAPS_INDEX_READ) {
      *p2pStatus = g_devicePairs[a][b].p2pStatusRead;
    } else {
      *p2pStatus = g_devicePairs[a][b].p2pStatusWrite;
    }
    return Ok();
  } else {
    NVML_CALL(nvmlDeviceGetP2PStatus, device1, device2, p2pIndex, p2pStatus);
  }
}

Status NvmlApi::nvmlDeviceGetFieldValues(
    nvmlDevice_t device,
    int valuesCount,
    nvmlFieldValue_t* values) {
  NVML_CALL(nvmlDeviceGetFieldValues, device, valuesCount, values);
}

// MNNVL support
Status NvmlApi::nvmlDeviceGetGpuFabricInfoV(
    nvmlDevice_t device,
    nvmlGpuFabricInfoV_t* info) {
  info->version = nvmlGpuFabricInfo_v2;
  NVML_CALL(nvmlDeviceGetGpuFabricInfoV, device, info);
}

Status NvmlApi::nvmlDeviceGetPlatformInfo(
    nvmlDevice_t device,
    nvmlPlatformInfo_t* platformInfo) {
  platformInfo->version = nvmlPlatformInfo_v2;
  NVML_CALL(nvmlDeviceGetPlatformInfo, device, platformInfo);
}

Status NvmlApi::nvmlSystemGetConfComputeStatus(NvmlCCStatus* status) {
  nvmlCCInfoInternal ccInfo;
  NVML_ENSURE_INIT();
  std::lock_guard<std::mutex> _lock(g_mutex);
  *status = {};
#if UNIFLOW_NVML_DIRECT
  // Direct link: both symbols are always available.
  constexpr bool haveSettings = true;
  constexpr bool haveState = true;
#else
  bool haveSettings = pfn_nvmlSystemGetConfComputeSettings != nullptr;
  bool haveState = pfn_nvmlSystemGetConfComputeState != nullptr;
#endif
  if (haveSettings) {
    ccInfo.settingV12040.version = nvmlSystemConfComputeSettings_v1;
    NVML_CHECK(nvmlSystemGetConfComputeSettings, &ccInfo.settingV12040);
    status->CCEnabled =
        ccInfo.settingV12040.ccFeature == NVML_CC_SYSTEM_FEATURE_ENABLED;
    status->multiGpuProtectedPCIE = ccInfo.settingV12040.multiGpuMode ==
        NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE;
#ifdef NVML_CC_SYSTEM_MULTIGPU_NVLE
    status->multiGpuNVLE =
        ccInfo.settingV12040.multiGpuMode == NVML_CC_SYSTEM_MULTIGPU_NVLE;
#endif
  } else if (haveState) {
    NVML_CHECK(nvmlSystemGetConfComputeState, &ccInfo.settingV12020);
    status->CCEnabled =
        ccInfo.settingV12020.ccFeature == NVML_CC_SYSTEM_FEATURE_ENABLED;
  }
  return Ok();
}

} // namespace uniflow
