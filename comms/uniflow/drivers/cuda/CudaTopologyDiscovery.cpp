// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaTopologyDiscovery.h"

#include "comms/uniflow/drivers/TopologyDiscovery.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ranges>
#include <tuple>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

#include "comms/uniflow/drivers/sysfs/SysfsApi.h"

namespace uniflow {

namespace {

// --- sysfs helpers ---

/// Normalize a CUDA PCI bus ID string to lowercase (e.g. "0000:07:00.0").
void normalizePciBusId(char* p, size_t len) {
  if (p == nullptr) {
    return;
  }
  for (size_t i = 0; i < len && p[i]; ++i) {
    p[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(p[i])));
  }
}

/// Extract the BDF (last path component) from a sysfs device path.
/// E.g. "/sys/devices/pci0000:00/0000:00:01.0/0000:07:00.0" → "0000:07:00.0"
std::string_view extractBdf(std::string_view sysfsPath) {
  auto pos = sysfsPath.rfind('/');
  if (pos == std::string::npos) {
    return sysfsPath;
  }
  return sysfsPath.substr(pos + 1);
}

/// Returns true if the path component looks like a PCI BDF
/// (contains at least two colons, e.g. "0000:07:00.0").
bool isPciBdf(std::string_view component) {
  int colonCount = 0;
  for (char c : component) {
    colonCount += (c == ':') ? 1 : 0;
    if (colonCount >= 2) {
      return true;
    }
  }
  return false;
}

/// Walk up one directory via SysfsApi and return the parent.
/// Returns empty string if the parent is not a PCI device (no BDF component).
std::string getPciParent(SysfsApi& sysfs, const std::string& sysfsPath) {
  std::string parentPath = sysfsPath + "/..";
  auto resolved = sysfs.resolvePath(parentPath);
  if (!resolved) {
    return {};
  }
  auto parent = std::move(resolved).value();
  auto bdf = extractBdf(parent);
  if (!isPciBdf(bdf)) {
    return {};
  }
  return parent;
}

/// Build the PCI bridge ancestor chain by walking up from a sysfs device path.
/// Returns a vector of resolved sysfs paths from the device up to the root
/// complex. Depth-limited to 32 to avoid runaway loops.
constexpr int kMaxAncestorChainDepth = 32;
std::vector<std::string> buildAncestorChain(
    SysfsApi& sysfs,
    const std::string& sysfsPath) {
  std::vector<std::string> chain;
  chain.push_back(sysfsPath);
  std::string current = sysfsPath;
  for (int depth = 0; depth < kMaxAncestorChainDepth; ++depth) {
    std::string parent = getPciParent(sysfs, current);
    if (parent.empty()) {
      break;
    }
    chain.push_back(parent);
    current = parent;
  }
  return chain;
}

/// Read the NUMA node ID from a PCI device's sysfs path.
int readNumaNode(SysfsApi& sysfs, std::string_view pciDevicePath) {
  std::string numaPath = std::string(pciDevicePath) + "/numa_node";
  std::string content = sysfs.readFile(numaPath);
  if (content.empty()) {
    return -1;
  }
  char* end = nullptr;
  long val = std::strtol(content.c_str(), &end, 10);
  if (end == content.c_str()) {
    return -1; // No digits parsed.
  }
  return static_cast<int>(val);
}

/// Read the backing netdev name for an IB device by listing
/// /sys/class/infiniband/<dev>/device/net/. Vendor-neutral (works for mlx5,
/// bnxt_re, etc.). Returns the first entry, or empty if none is found.
std::string readNetdevName(SysfsApi& sysfs, const std::string& ibdevPath) {
  auto entries = sysfs.listDir(ibdevPath + "/device/net");
  return entries.empty() ? std::string{} : entries.front();
}

/// Convert a sysfs max_link_speed string to Mbps per lane.
/// Handles both old ("16 GT/s") and new ("16.0 GT/s PCIe") kernel formats.
/// Encoding overhead is already baked into the returned value.
/// Returns 0 if the string is unrecognized; defaults to Gen3 (6000) like NCCL.
struct SpeedEntry {
  const char* prefix;
  uint32_t mbpsPerLane;
};
// clang-format off
constexpr SpeedEntry kSpeedTable[] = {
    {"2.5 GT/s",      1500},  // Gen1: 2.5 GT/s * 8/10 encoding = 2000 Mbps → effective ~1500
    {"5 GT/s",         3000}, // Gen2
    {"5.0 GT/s",       3000}, // Gen2 (new kernel format)
    {"8 GT/s",         6000}, // Gen3 (128b/130b)
    {"8.0 GT/s",       6000}, // Gen3
    {"16 GT/s",       12000}, // Gen4
    {"16.0 GT/s",     12000}, // Gen4
    {"32 GT/s",       24000}, // Gen5
    {"32.0 GT/s",     24000}, // Gen5
    {"64 GT/s",       48000}, // Gen6
    {"64.0 GT/s",     48000}, // Gen6
};
// clang-format on

uint32_t parseLinkSpeed(SysfsApi& sysfs, const std::string& filePath) {
  std::string content = sysfs.readFile(filePath);
  if (content.empty()) {
    return 0;
  }
  for (const auto& entry : kSpeedTable) {
    if (content.find(entry.prefix) == 0) {
      return entry.mbpsPerLane;
    }
  }
  return 6000; // Default to Gen3 like NCCL (src/graph/topo.cc:449).
}

/// Parse max_link_width from a sysfs file, e.g. "16" → 16.
uint16_t parseLinkWidth(SysfsApi& sysfs, const std::string& filePath) {
  std::string content = sysfs.readFile(filePath);
  if (content.empty()) {
    return 0;
  }
  char* end = nullptr;
  long val = std::strtol(content.c_str(), &end, 10);
  if (end == content.c_str() || val <= 0) {
    return 0;
  }
  return static_cast<uint16_t>(val);
}

/// Read PCIe link info for a device at the given sysfs path.
/// Takes the minimum of the device's own capability and the upstream
/// parent port's capability, matching NCCL's approach. A Gen5 GPU in
/// a Gen4 slot will report Gen4 speed.
PcieLinkInfo readPcieLinkInfo(
    SysfsApi& sysfs,
    const std::string& sysfsDevicePath) {
  std::string devSpeedPath = sysfsDevicePath + "/max_link_speed";
  std::string devWidthPath = sysfsDevicePath + "/max_link_width";
  std::string portSpeedPath = sysfsDevicePath + "/../max_link_speed";
  std::string portWidthPath = sysfsDevicePath + "/../max_link_width";

  uint32_t devSpeed = parseLinkSpeed(sysfs, devSpeedPath);
  uint32_t portSpeed = parseLinkSpeed(sysfs, portSpeedPath);
  uint16_t devWidth = parseLinkWidth(sysfs, devWidthPath);
  uint16_t portWidth = parseLinkWidth(sysfs, portWidthPath);

  // Take the minimum of device and parent port (bottleneck).
  // speed == 0 means "uninitialized" or "unknown".
  auto minNonZero = [](auto a, auto b) {
    if (a == 0) {
      return b;
    }
    if (b == 0) {
      return a;
    }
    return std::min(a, b);
  };
  uint32_t speed = minNonZero(devSpeed, portSpeed);
  uint16_t width = minNonZero(devWidth, portWidth);

  return PcieLinkInfo{
      .speedMbpsPerLane = speed,
      .width = width,
  };
}

/// Count NUMA nodes by scanning /sys/devices/system/node/.
constexpr std::string_view kNodePrefix = "node";
int discoverNumaNodeCount(SysfsApi& sysfs) {
  auto entries = sysfs.listDir("/sys/devices/system/node", kNodePrefix);
  int count = 0;
  for (const auto& name : entries) {
    if (name.size() > kNodePrefix.size() &&
        std::isdigit(static_cast<unsigned char>(name[kNodePrefix.size()]))) {
      ++count;
    }
  }
  return count;
}

// --- Discovery data ---

/// Temporary GPU data gathered during discovery, before graph construction.
struct GpuDiscovery {
  TopoNode::GpuData data; // Goes into the node
  std::string sysfsPath;
  std::vector<std::string> ancestorChain;
  PcieLinkInfo linkInfo;
  int nvSwitchLinkCount{0};
  uint32_t c2cBwMBps{0}; // Total C2C bandwidth (activeCount * perLinkBw)
};

/// IB width multipliers indexed by bit position in ibv_port_attr.active_width.
/// Bit 0 = 1x, bit 1 = 4x, bit 2 = 8x, bit 3 = 12x, bit 4 = 2x.
// clang-format off
constexpr int kIbvWidths[] = {1, 4, 8, 12, 2};
// clang-format on

/// IB per-lane speed (Mbps) indexed by bit position in active_speed.
/// Bit 0 = SDR, 1 = DDR, 2 = QDR, 3 = QDR, 4 = FDR, 5 = EDR,
/// 6 = HDR, 7 = NDR.
// clang-format off
constexpr int kIbvSpeeds[] = {
    2500,   // SDR
    5000,   // DDR
    10000,  // QDR
    10000,  // QDR
    14000,  // FDR
    25000,  // EDR
    50000,  // HDR
    100000, // NDR
};
// clang-format on

/// Return the index of the lowest set bit, or -1 if none.
int firstBitSet(uint32_t v) {
  for (int i = 0; v != 0; ++i, v >>= 1) {
    if (v & 1) {
      return i;
    }
  }
  return -1;
}

/// Compute RDMA port speed in Mbps from ibv_port_attr bitmask fields.
uint32_t ibPortSpeedMbps(uint32_t activeSpeed, uint32_t activeWidth) {
  int si = firstBitSet(activeSpeed);
  int wi = firstBitSet(activeWidth);
  if (si < 0 || wi < 0) {
    return 0;
  }
  int speed =
      (si < static_cast<int>(std::size(kIbvSpeeds))) ? kIbvSpeeds[si] : 0;
  int width =
      (wi < static_cast<int>(std::size(kIbvWidths))) ? kIbvWidths[wi] : 0;
  return static_cast<uint32_t>(speed * width);
}

/// Temporary NIC data gathered during discovery, before graph construction.
/// One entry per physical device, with all active ports collected.
struct NicDiscovery {
  std::string name; // IB device name (e.g. "mlx5_0")
  TopoNode::NicData data; // Goes into the node
  std::string sysfsPath;
  std::vector<std::string> ancestorChain;
  PcieLinkInfo linkInfo;
};

/// NVSwitch PCI device class (matches NCCL's kvDictPciClass in topo.cc).
constexpr std::string_view kNvSwitchPciClass = "0x068000";

/// Read the PCI device class from sysfs (e.g. "0x068000" for NVSwitch).
std::string readPciClass(SysfsApi& sysfs, const std::string& sysfsDevicePath) {
  return sysfs.readFile(sysfsDevicePath + "/class");
}

/// Check if a PCI device is an NVSwitch by reading its sysfs class.
bool isNvSwitchByClass(SysfsApi& sysfs, const std::string& sysfsDevicePath) {
  std::string pciClass = readPciClass(sysfs, sysfsDevicePath);
  return pciClass.find(kNvSwitchPciClass) == 0;
}

// --- Discovery functions ---

Status discoverC2C(
    NvmlApi& nvmlApi,
    std::vector<GpuDiscovery>& gpus,
    const std::vector<nvmlDevice_t>& nvmlHandles) {
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (nvmlHandles[i] == nullptr) {
      continue;
    }
    nvmlDevice_t nvmlDev = nvmlHandles[i];
    auto& gpu = gpus[i];
    if (gpu.data.sm < 90) {
      continue;
    }

    nvmlFieldValue_t countFv{};
    countFv.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
    auto st = nvmlApi.nvmlDeviceGetFieldValues(nvmlDev, 1, &countFv);
    if (!st || countFv.nvmlReturn != NVML_SUCCESS || countFv.value.uiVal == 0) {
      continue;
    }
    int totalC2CLinks = static_cast<int>(countFv.value.uiVal);

    for (int l = 0; l < totalC2CLinks; ++l) {
      nvmlFieldValue_t fvs[2]{};
      fvs[0].fieldId = NVML_FI_DEV_C2C_LINK_GET_STATUS;
      fvs[0].scopeId = l;
      fvs[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
      fvs[1].scopeId = l;
      st = nvmlApi.nvmlDeviceGetFieldValues(nvmlDev, 2, fvs);
      if (!st) {
        continue;
      }
      if (fvs[0].nvmlReturn == NVML_SUCCESS && fvs[0].value.uiVal == 1 &&
          fvs[1].nvmlReturn == NVML_SUCCESS) {
        gpu.c2cBwMBps += fvs[1].value.uiVal;
      }
    }
  }
  return Ok();
}

/// Discover GPUs via NVML/CUDA: PCI info, NVLink, C2C.
Result<std::vector<GpuDiscovery>>
discoverGpus(CudaApi& cudaApi, NvmlApi& nvmlApi, SysfsApi& sysfs) {
  // Use CUDA device count (respects CUDA_VISIBLE_DEVICES) instead of
  // NVML count (sees all physical GPUs). NVML ignores
  // CUDA_VISIBLE_DEVICES and always sees all GPUs, so iterating by
  // NVML count would call cudaSetDevice on indices beyond the visible
  // set, causing failures. We iterate by CUDA count and resolve each
  // CUDA device to its NVML handle via PCI bus ID matching.
  auto count = cudaApi.getDeviceCount();
  CHECK_RETURN(count);
  int gpuCount = count.value();

  constexpr int kPciBusIdLen = 64;
  char pciBusIdBuf[kPciBusIdLen];

  // Save current device if a CUDA context already exists.
  auto oldDev = cudaApi.getDevice();
  CHECK_RETURN(oldDev);
  CudaDeviceGuard guard(cudaApi, oldDev.value());

  std::vector<GpuDiscovery> gpus(gpuCount);
  // NVML handles per CUDA device, resolved via PCI bus ID.
  std::vector<nvmlDevice_t> nvmlHandles(gpuCount, nullptr);

  for (int i = 0; i < gpuCount; ++i) {
    auto& gpu = gpus[i];
    gpu.data.cudaDeviceId = i;

    // Initialize CUDA context on this device before querying PCI bus ID.
    CHECK_EXPR(cudaApi.setDevice(i));

    // Get PCI bus ID from CUDA.
    std::memset(pciBusIdBuf, 0, kPciBusIdLen);
    CHECK_EXPR(cudaApi.getDevicePCIBusId(pciBusIdBuf, kPciBusIdLen, i));
    normalizePciBusId(pciBusIdBuf, kPciBusIdLen);

    // Resolve NVML handle for this CUDA device via PCI bus ID.
    // This correctly maps CUDA device indices to NVML devices
    // regardless of CUDA_VISIBLE_DEVICES reordering.
    nvmlDevice_t nvmlDev = nullptr;
    auto nvmlSt = nvmlApi.nvmlDeviceGetHandleByPciBusId(pciBusIdBuf, &nvmlDev);
    if (nvmlSt) {
      nvmlHandles[i] = nvmlDev;
    }

    // Resolve the sysfs path.
    std::string sysfsLinkPath =
        std::string("/sys/bus/pci/devices/") + pciBusIdBuf;
    auto resolvedPath = sysfs.resolvePath(sysfsLinkPath);
    gpu.sysfsPath =
        resolvedPath ? std::move(resolvedPath).value() : sysfsLinkPath;

    gpu.data.bdf = std::string(extractBdf(gpu.sysfsPath));
    gpu.data.numaNode = readNumaNode(sysfs, gpu.sysfsPath);
    gpu.ancestorChain = buildAncestorChain(sysfs, gpu.sysfsPath);
    gpu.linkInfo = readPcieLinkInfo(sysfs, gpu.sysfsPath);

    // Get SM version via the resolved NVML handle.
    if (nvmlHandles[i] != nullptr) {
      int major = -1, minor = -1;
      auto ccSt = nvmlApi.nvmlDeviceGetCudaComputeCapability(
          nvmlHandles[i], &major, &minor);
      if (ccSt) {
        gpu.data.sm = major * 10 + minor;
      }
    }
  }

  // NVLink connectivity via NVML.
  // Max NVLink count per SM generation, ordered highest SM first.
  struct SmNvLinkEntry {
    int minSm;
    int maxLinks;
  };
  // clang-format off
  constexpr SmNvLinkEntry kMaxNvLinksTable[] = {
      {90, 18}, // Hopper+
      {80, 12}, // Ampere
      {70,  6}, // Volta
      {60,  4}, // Pascal
  };
  // clang-format on
  auto getMaxNvLinks = [&](int sm) -> int {
    for (const auto& [minSm, maxLinks] : kMaxNvLinksTable) {
      if (sm >= minSm) {
        return maxLinks;
      }
    }
    return 0;
  };

  for (int i = 0; i < gpuCount; ++i) {
    if (nvmlHandles[i] == nullptr) {
      continue;
    }
    nvmlDevice_t nvmlDev = nvmlHandles[i];
    auto& gpu = gpus[i];
    int sm = gpu.data.sm;
    int maxNvLinks = getMaxNvLinks(sm);

    for (int link = 0; link < maxNvLinks; ++link) {
      unsigned int canP2P = 0;
      auto st = nvmlApi.nvmlDeviceGetNvLinkCapability(
          nvmlDev, link, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P);
      if (!st || !canP2P) {
        continue;
      }

      nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;
#if CUDART_VERSION >= 11080
      if (sm >= 90) {
        nvmlFieldValue_t fv{};
        fv.fieldId = NVML_FI_DEV_NVLINK_GET_STATE;
        fv.scopeId = link;
        st = nvmlApi.nvmlDeviceGetFieldValues(nvmlDev, 1, &fv);
        if (st && fv.nvmlReturn == NVML_SUCCESS) {
          isActive = static_cast<nvmlEnableState_t>(fv.value.uiVal);
        }
      } else
#endif
      {
        nvmlApi.nvmlDeviceGetNvLinkState(nvmlDev, link, &isActive);
      }

      if (isActive != NVML_FEATURE_ENABLED) {
        continue;
      }

      nvmlPciInfo_t remotePci{};
      st = nvmlApi.nvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, link, &remotePci);
      if (!st) {
        continue;
      }

      normalizePciBusId(remotePci.busId, sizeof(remotePci.busId));
      std::string remoteBdf(remotePci.busId);

      // Classify the remote NVLink endpoint.
      // Sentinel BDF (all-f) means NVSwitch not visible (e.g. in a VM).
      // Otherwise resolve sysfs path and read PCI class to confirm.
      bool isNvSwitch = false;
      if (remoteBdf == "fffffff:ffff:ff") {
        isNvSwitch = true;
      } else {
        std::string remoteSysfs = "/sys/bus/pci/devices/" + remoteBdf;
        auto remoteResolved = sysfs.resolvePath(remoteSysfs);
        if (!remoteResolved) {
          // Not visible in sysfs (e.g. VM) — assume NVSwitch.
          isNvSwitch = true;
        } else {
          // Visible — read PCI class to check (bare-metal NVSwitch).
          isNvSwitch = isNvSwitchByClass(sysfs, remoteResolved.value());
        }
      }

      if (isNvSwitch) {
        gpu.nvSwitchLinkCount++;
      }
    }
  }

  // C2C link detection via NVML (using resolved handles).
  discoverC2C(nvmlApi, gpus, nvmlHandles);

  return gpus;
}

/// Check whether an ibdev sysfs path belongs to a virtual (software) RDMA
/// device such as RXE (Soft-RoCE).  These live under
/// /sys/devices/virtual/ and have no PCI backing.
bool isVirtualRdmaDevice(SysfsApi& sysfs, const std::string& ibdevPath) {
  auto resolved = sysfs.resolvePath(ibdevPath);
  if (!resolved) {
    return false;
  }
  return resolved.value().find("/devices/virtual/") != std::string::npos;
}

constexpr size_t kDefaultRxeSpeed = 10000; // 10 Gbps

/// Discover NICs via IbvApi: enumerate IB devices and active ports.
/// Returns empty vector if ibverbs is not available.
Result<std::vector<NicDiscovery>> discoverNics(
    IbvApi& ibvApi,
    SysfsApi& sysfs) {
  std::vector<NicDiscovery> nics;
  CHECK_EXPR(ibvApi.init());

  int numDevices = 0;
  auto devListResult = ibvApi.getDeviceList(&numDevices);
  CHECK_RETURN(devListResult);

  auto devListDeleter = [&ibvApi](ibv_device** ptr) {
    if (ptr) {
      ibvApi.freeDeviceList(ptr);
    }
  };
  std::unique_ptr<ibv_device*[], decltype(devListDeleter)> devListGuard(
      devListResult.value(), devListDeleter);
  ibv_device** devList = devListGuard.get();
  if (devList == nullptr) {
    return nics;
  }

  for (int i = 0; i < numDevices; ++i) {
    ibv_device* dev = devList[i];
    if (dev == nullptr) {
      continue;
    }

    auto name = ibvApi.getDeviceName(dev);
    if (!name) {
      continue;
    }
    std::string devName = name.value();

    // Resolve PCI device path from ibdev sysfs path.
    std::string ibdevPath = dev->ibdev_path;

    bool isVirtual = isVirtualRdmaDevice(sysfs, ibdevPath);
    std::string pciDevicePath;

    if (!isVirtual) {
      auto resolved = sysfs.resolvePath(ibdevPath + "/device");
      if (!resolved) {
        continue;
      }
      pciDevicePath = std::move(resolved).value();
    }

    // Open device and query ports.
    auto ctxResult = ibvApi.openDevice(dev);
    if (!ctxResult) {
      continue;
    }
    ibv_context* ctx = ctxResult.value();

    // Collect active ports and RDMA port speed for this device.
    int activePort = -1;
    uint32_t portSpeedMbps = 0;
    ibv_device_attr devAttr{};
    if (ibvApi.queryDevice(ctx, &devAttr)) {
      for (uint8_t port = 1; port <= devAttr.phys_port_cnt; ++port) {
        ibv_port_attr portAttr{};
        if (ibvApi.queryPort(ctx, port, &portAttr) &&
            portAttr.state == IBV_PORT_ACTIVE) {
          auto speed =
              ibPortSpeedMbps(portAttr.active_speed, portAttr.active_width);
          if (activePort == -1 || speed > portSpeedMbps) {
            activePort = port;
            portSpeedMbps =
                ibPortSpeedMbps(portAttr.active_speed, portAttr.active_width);
          }
        }
      }
    }

    if (activePort != -1) {
      if (isVirtual) {
        if (portSpeedMbps == 0) {
          portSpeedMbps = kDefaultRxeSpeed;
        }
        // Virtual devices have no PCI topology. Use NUMA 0, empty ancestor
        // chain, and a default port speed if the ibverbs query returned 0.
        nics.push_back({
            .name = std::move(devName),
            .data =
                TopoNode::NicData{
                    .bdf = "virtual",
                    .numaNode = 0,
                    .port = activePort,
                    .portSpeedMbps = portSpeedMbps,
                    .netdevName = readNetdevName(sysfs, ibdevPath),
                },
            .sysfsPath = ibdevPath,
            .ancestorChain = {},
            .linkInfo = {},
        });
      } else {
        nics.push_back({
            .name = std::move(devName),
            .data =
                TopoNode::NicData{
                    .bdf = std::string(extractBdf(pciDevicePath)),
                    .numaNode = readNumaNode(sysfs, pciDevicePath),
                    .port = activePort,
                    .portSpeedMbps = portSpeedMbps,
                    .netdevName = readNetdevName(sysfs, ibdevPath),
                },
            .sysfsPath = pciDevicePath,
            .ancestorChain = buildAncestorChain(sysfs, pciDevicePath),
            .linkInfo = readPcieLinkInfo(sysfs, pciDevicePath),
        });
      }
    }

    ibvApi.closeDevice(ctx);
  }
  return nics;
}

/// Unified PCI device info for PCIe edge construction.
struct PciDevice {
  int nodeId;
  int numaNode;
  std::vector<std::string> ancestorChain;
  uint32_t portSpeedMBps{0}; // RDMA port speed cap (MB/s), 0 = no cap (GPUs)
};

/// Find the nearest common ancestor between two PCI ancestor chains.
/// Returns {indexInA, indexInB} or {-1, -1} if none found.
Result<std::pair<int, int>> findCommonAncestor(
    const std::vector<std::string>& chainA,
    const std::vector<std::string>& chainB) {
  for (size_t a = 0; a < chainA.size(); ++a) {
    for (size_t b = 0; b < chainB.size(); ++b) {
      if (chainA[a] == chainB[b]) {
        return std::make_pair(a, b);
      }
    }
  }
  return ErrCode::NotConnected;
}

/// Compute the minimum non-zero PCIe bandwidth (MB/s) over a range of
/// sysfs paths. Returns UINT32_MAX if no positive bandwidth is found.
template <typename Range>
uint32_t pcieBottleneckMBps(SysfsApi& sysfs, Range&& paths) {
  auto bws = std::forward<Range>(paths) |
      std::views::transform([&sysfs](const auto& p) {
               return readPcieLinkInfo(sysfs, p).bandwidthMBps();
             }) |
      std::views::filter([](uint32_t bw) { return bw > 0; });
  auto it = std::ranges::min_element(bws);
  return (it == bws.end()) ? UINT32_MAX : *it;
}

/// Compute bottleneck bandwidth (MB/s) along a PCI ancestor chain.
uint32_t computeChainBottleneckMBps(
    SysfsApi& sysfs,
    const std::vector<std::string>& chain) {
  uint32_t minBw = pcieBottleneckMBps(sysfs, chain);
  return (minBw == UINT32_MAX) ? 0 : minBw;
}

/// Compute bottleneck bandwidth (MB/s) between two devices at their
/// common ancestor in the PCI hierarchy.
uint32_t computePairBottleneckMBps(
    SysfsApi& sysfs,
    const std::vector<std::string>& chainA,
    const std::vector<std::string>& chainB,
    int commonIdxA,
    int commonIdxB) {
  uint32_t bwA =
      pcieBottleneckMBps(sysfs, chainA | std::views::take(commonIdxA + 1));
  uint32_t bwB =
      pcieBottleneckMBps(sysfs, chainB | std::views::take(commonIdxB + 1));
  uint32_t minBw = std::min(bwA, bwB);
  return (minBw == UINT32_MAX) ? 0 : minBw;
}

/// Per-link NVLink effective bandwidth in MB/s based on GPU SM version.
/// Values reference NCCL's effective bandwidth (nccl topo.h).
/// Ordered highest SM first. First match wins.
struct SmBwEntry {
  int minSm;
  uint32_t bwMBps;
};
// clang-format off
constexpr SmBwEntry kNvLinkBwTable[] = {
    {100, 40100}, // Blackwell+: NVLink 5.0
    { 90, 20600}, // Hopper: NVLink 4.0
    { 87, 20000}, // Ampere SM87+: NVLink 3.0
    { 86, 12000}, // Ampere SM86: NVLink 3.0 (different BW)
    { 80, 20000}, // Ampere: NVLink 3.0
    { 70, 20000}, // Volta: NVLink 2.0
    { 60, 18000}, // Pascal: NVLink 1.0
};
// clang-format on

uint32_t nvlinkPerLinkBwMBps(int sm) {
  for (const auto& [minSm, bw] : kNvLinkBwTable) {
    if (sm >= minSm) {
      return bw;
    }
  }
  return 0;
}

/// Inter-CPU bandwidth constants in GB/s, matching NCCL (topo.h).
// clang-format off
[[maybe_unused]] constexpr uint32_t kBwP9MBps      = 32000;  // IBM POWER9
[[maybe_unused]] constexpr uint32_t kBwArmMBps     =  6000;  // ARM
constexpr uint32_t kBwAmdMBps     = 16000;  // AMD (all models)
constexpr uint32_t kBwBdwQpiMBps  =  6000;  // Intel Broadwell and older
constexpr uint32_t kBwSklQpiMBps  = 10000;  // Intel Skylake
constexpr uint32_t kBwSrpQpiMBps  = 22000;  // Intel Sapphire Rapids
constexpr uint32_t kBwErpQpiMBps  = 40000;  // Intel Emerald Rapids
constexpr uint32_t kBwDefaultMBps = 12000;  // Fallback
// clang-format on

/// Detect inter-CPU bandwidth based on CPU architecture, vendor, and model.
/// Uses CPUID on x86_64 (same approach as NCCL xml.cc / topo.cc).
uint32_t detectInterCpuBwMBps() {
#if defined(__aarch64__)
  return kBwArmMBps;
#elif defined(__powerpc64__)
  return kBwP9MBps;
#elif defined(__x86_64__)
  // CPUID leaf 0: vendor string.
  uint32_t eax, ebx, ecx, edx;
  __cpuid(0, eax, ebx, ecx, edx);

  // Vendor string is EBX + EDX + ECX (12 bytes).
  char vendor[13];
  std::memcpy(vendor, &ebx, 4);
  std::memcpy(vendor + 4, &edx, 4);
  std::memcpy(vendor + 8, &ecx, 4);
  vendor[12] = '\0';

  if (std::strcmp(vendor, "AuthenticAMD") == 0) {
    return kBwAmdMBps;
  }

  if (std::strcmp(vendor, "GenuineIntel") == 0) {
    // CPUID leaf 1: family/model.
    uint32_t eax1;
    __cpuid(1, eax1, ebx, ecx, edx);
    int familyId = ((eax1 >> 8) & 0xF) + ((eax1 >> 20) & 0xFF);
    int modelId = ((eax1 >> 4) & 0xF) + (((eax1 >> 16) & 0xF) << 4);

    // Model IDs are only valid for family 6 processors.
    if (familyId == 6) {
      // Ordered highest model first. First match wins.
      struct ModelBwEntry {
        int minModel;
        uint32_t bw;
      };
      // clang-format off
      constexpr ModelBwEntry kIntelTable[] = {
          {0xCF, kBwErpQpiMBps}, // Emerald Rapids
          {0x8F, kBwSrpQpiMBps}, // Sapphire Rapids
          {0x55, kBwSklQpiMBps}, // Skylake
      };
      // clang-format on
      for (const auto& [minModel, bw] : kIntelTable) {
        if (modelId >= minModel) {
          return bw;
        }
      }
    }
    return kBwBdwQpiMBps; // Broadwell and older
  }

  return kBwDefaultMBps;
#else
  return kBwDefaultMBps;
#endif
}

// --- Graph construction ---

/// All data probed from hardware before the graph is built.
struct DiscoveryData {
  std::vector<GpuDiscovery> gpus;
  std::vector<NicDiscovery> nics;
  int numaCount{0};
};

Status discoverHardware(
    CudaApi& cudaApi,
    NvmlApi& nvmlApi,
    IbvApi& ibvApi,
    SysfsApi& sysfs,
    DiscoveryData& data) {
  auto gpuResult = discoverGpus(cudaApi, nvmlApi, sysfs);
  if (gpuResult) {
    data.gpus = std::move(gpuResult).value();
  }

  auto nicResult = discoverNics(ibvApi, sysfs);
  if (nicResult) {
    data.nics = std::move(nicResult).value();
  }

  data.numaCount = discoverNumaNodeCount(sysfs);
  if (data.numaCount == 0) {
    data.numaCount = 1;
  }
  return Ok();
}

void buildNodes(Topology& topo, const DiscoveryData& data) {
  int gpuCount = data.gpus.size();
  int nicCount = data.nics.size();
  int numaCount = data.numaCount;

  for (int i = 0; i < gpuCount; ++i) {
    int nodeId = topo.addNode({
        .type = NodeType::GPU,
        .name = "cuda:" + std::to_string(data.gpus[i].data.cudaDeviceId),
        .data = data.gpus[i].data,
    });
    topo.registerGpuNode(i, nodeId);
  }

  for (int i = 0; i < numaCount; ++i) {
    int nodeId = topo.addNode({
        .type = NodeType::CPU,
        .name = "cpu:" + std::to_string(i),
        .data = TopoNode::CpuData{.numaId = i},
    });
    topo.registerCpuNode(i, nodeId);
  }

  for (int i = 0; i < nicCount; ++i) {
    int nodeId = topo.addNode({
        .type = NodeType::NIC,
        .name = data.nics[i].name,
        .data = data.nics[i].data,
    });
    topo.registerNicNode(i, nodeId);
  }
}

void buildP2pMatrix(Topology& topo, CudaApi& cudaApi) {
  int gpuCount = static_cast<int>(topo.gpuCount());
  std::vector<std::vector<bool>> matrix(
      gpuCount, std::vector<bool>(gpuCount, false));
  for (int i = 0; i < gpuCount; ++i) {
    matrix[i][i] = true;
    for (int j = 0; j < gpuCount; ++j) {
      if (i == j) {
        continue;
      }
      auto result = cudaApi.deviceCanAccessPeer(i, j);
      if (result.hasValue()) {
        matrix[i][j] = result.value();
      }
    }
  }
  topo.setP2pMatrix(std::move(matrix));
}

void buildEdges(Topology& topo, SysfsApi& sysfs, const DiscoveryData& data) {
  int gpuCount = data.gpus.size();
  int nicCount = data.nics.size();
  int numaCount = data.numaCount;

  // Collect all PCI devices for PCIe edge construction.
  std::vector<PciDevice> pciDevices;
  pciDevices.reserve(gpuCount + nicCount);
  for (int i = 0; i < gpuCount; ++i) {
    pciDevices.push_back({
        .nodeId = topo.getGpuNode(i).id,
        .numaNode = data.gpus[i].data.numaNode,
        .ancestorChain = data.gpus[i].ancestorChain,
    });
  }
  for (int i = 0; i < nicCount; ++i) {
    // Convert RDMA port speed from Mbps to MB/s for bandwidth comparison.
    uint32_t portMBps = data.nics[i].data.portSpeedMbps / 8;
    pciDevices.push_back({
        .nodeId = topo.getNicNode(i).id,
        .numaNode = data.nics[i].data.numaNode,
        .ancestorChain = data.nics[i].ancestorChain,
        .portSpeedMBps = portMBps,
    });
  }

  // Connect each PCI device to its CPU node.
  // For NICs, cap bandwidth with the RDMA port speed (like NCCL's LINK_NET).
  // For virtual devices (empty ancestor chain), use port speed directly.
  for (const auto& dev : pciDevices) {
    int numa =
        (dev.numaNode >= 0 && dev.numaNode < numaCount) ? dev.numaNode : 0;
    uint32_t bw;
    if (dev.ancestorChain.empty()) {
      // Virtual device — no PCIe topology; use port speed as bandwidth.
      bw = dev.portSpeedMBps;
    } else {
      bw = computeChainBottleneckMBps(sysfs, dev.ancestorChain);
      if (dev.portSpeedMBps > 0) {
        bw = std::min(bw, dev.portSpeedMBps);
      }
    }
    topo.addLink(dev.nodeId, topo.getCpuNode(numa).id, PathType::PHB, bw);
  }

  // C2C edges (GPU → CPU for Grace Hopper systems).
  for (int i = 0; i < gpuCount; ++i) {
    if (data.gpus[i].c2cBwMBps == 0) {
      continue;
    }
    int numa = data.gpus[i].data.numaNode;
    if (numa < 0 || numa >= numaCount) {
      continue;
    }
    topo.addLink(
        topo.getGpuNode(i).id,
        topo.getCpuNode(numa).id,
        PathType::C2C,
        data.gpus[i].c2cBwMBps);
  }

  // Direct PCIe edges between devices sharing a PCIe switch.
  for (size_t i = 0; i < pciDevices.size(); ++i) {
    for (size_t j = i + 1; j < pciDevices.size(); ++j) {
      auto ancestor = findCommonAncestor(
          pciDevices[i].ancestorChain, pciDevices[j].ancestorChain);
      if (!ancestor) {
        continue;
      }
      auto [idxA, idxB] = ancestor.value();
      // PIX = at most one PCIe switch between each device and the common
      // ancestor. One switch = device → downstream port → upstream port,
      // so idxA/idxB <= 2 means at most one switch per side.
      // PXB = either side traverses more than one switch (idx > 2), or
      // the common ancestor is the root port of exactly one chain.
      bool isRootA =
          idxA == static_cast<int>(pciDevices[i].ancestorChain.size()) - 1;
      bool isRootB =
          idxB == static_cast<int>(pciDevices[j].ancestorChain.size()) - 1;
      bool multiSwitch = idxA > 2 || idxB > 2 || (isRootA != isRootB);
      PathType contrib = multiSwitch ? PathType::PXB : PathType::PIX;
      uint32_t bw = computePairBottleneckMBps(
          sysfs,
          pciDevices[i].ancestorChain,
          pciDevices[j].ancestorChain,
          idxA,
          idxB);
      topo.addLink(pciDevices[i].nodeId, pciDevices[j].nodeId, contrib, bw);
    }
  }

  // NVLink edges (direct GPU → GPU for NVSwitch-connected pairs).
  for (int i = 0; i < gpuCount; ++i) {
    if (data.gpus[i].nvSwitchLinkCount <= 0) {
      continue;
    }
    uint32_t bwI = data.gpus[i].nvSwitchLinkCount *
        nvlinkPerLinkBwMBps(data.gpus[i].data.sm);
    for (int j = i + 1; j < gpuCount; ++j) {
      if (data.gpus[j].nvSwitchLinkCount <= 0 || !topo.canGpuAccess(i, j)) {
        continue;
      }
      uint32_t bwJ = data.gpus[j].nvSwitchLinkCount *
          nvlinkPerLinkBwMBps(data.gpus[j].data.sm);
      topo.addLink(
          topo.getGpuNode(i).id,
          topo.getGpuNode(j).id,
          PathType::NVL,
          std::min(bwI, bwJ));
    }
  }

  // Inter-NUMA edges (CPU → CPU).
  uint32_t interCpuBw = detectInterCpuBwMBps();
  for (int i = 0; i < numaCount; ++i) {
    for (int j = i + 1; j < numaCount; ++j) {
      topo.addLink(
          topo.getCpuNode(i).id,
          topo.getCpuNode(j).id,
          PathType::SYS,
          interCpuBw);
    }
  }
}

} // namespace

// CUDA implementation of the cross-platform discovery backend. Selected at
// build time via Buck `select()` in `comms/uniflow/drivers:topology-discovery`.
// Mirrors the per-platform pattern used by DeviceAdapter (see
// `CudaDeviceAdapter.cpp`).
CudaTopologyDiscovery::CudaTopologyDiscovery(
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<NvmlApi> nvmlApi,
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<SysfsApi> sysfsApi)
    : cudaApi_(cudaApi ? std::move(cudaApi) : std::make_shared<CudaApi>()),
      nvmlApi_(nvmlApi ? std::move(nvmlApi) : createNvmlApi()),
      ibvApi_(ibvApi ? std::move(ibvApi) : std::make_shared<IbvApi>()),
      sysfsApi_(sysfsApi ? std::move(sysfsApi) : std::make_shared<SysfsApi>()) {
}

Status CudaTopologyDiscovery::discover(Topology& topology) {
  topology.clear();

  DiscoveryData data;
  CHECK_EXPR(
      discoverHardware(*cudaApi_, *nvmlApi_, *ibvApi_, *sysfsApi_, data));
  buildNodes(topology, data);
  buildP2pMatrix(topology, *cudaApi_);
  buildEdges(topology, *sysfsApi_, data);
  topology.recomputePaths();
  topology.setStatus(Ok());
  return Ok();
}

std::unique_ptr<TopologyDiscoveryBackend> createDefaultDiscoveryBackend() {
  return std::make_unique<CudaTopologyDiscovery>();
}

} // namespace uniflow
