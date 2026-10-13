// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaTopologyDiscovery.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <memory>
#include <utility>

#if !defined(UNIFLOW_CMAKE_BUILD) || defined(UNIFLOW_ENABLE_RDMA_TRANSPORT)
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#endif
#include "comms/uniflow/drivers/cuda/CudaDeviceAdapter.h"
#include "comms/uniflow/drivers/topology/NicTopologyDiscovery.h"
#include "comms/uniflow/registry/PlatformRegistry.h"

namespace uniflow {

namespace {

using topology::buildAncestorChain;
using topology::extractBdf;
using topology::NicDiscovery;
using topology::PciDevice;
using topology::readNumaNode;
using topology::readPcieLinkInfo;

/// Normalize a CUDA PCI bus ID string to lowercase (e.g. "0000:07:00.0").
void normalizePciBusId(char* p, size_t len) {
  if (p == nullptr) {
    return;
  }
  for (size_t i = 0; i < len && p[i]; ++i) {
    p[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(p[i])));
  }
}

/// Temporary GPU data gathered during discovery, before graph construction.
struct GpuDiscovery {
  TopoNode::GpuData data; // Goes into the node
  std::string sysfsPath;
  std::vector<std::string> ancestorChain;
  PcieLinkInfo linkInfo;
  int nvSwitchLinkCount{0};
  uint32_t c2cBwMBps{0}; // Total C2C bandwidth (activeCount * perLinkBw)
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
  if (gpuCount == 0) {
    return std::vector<GpuDiscovery>{};
  }

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

/// All data probed from hardware before the graph is built.
struct DiscoveryData {
  std::vector<GpuDiscovery> gpus;
  std::vector<NicDiscovery> nics;
  int numaCount{0};
};

Status discoverHardware(
    CudaApi& cudaApi,
    NvmlApi& nvmlApi,
    [[maybe_unused]] IbvApi* ibvApi,
    SysfsApi& sysfs,
    DiscoveryData& data) {
  auto gpuResult = discoverGpus(cudaApi, nvmlApi, sysfs);
  if (gpuResult) {
    data.gpus = std::move(gpuResult).value();
  }

  auto nicData = topology::discoverNicHardware(ibvApi, sysfs);
  data.nics = std::move(nicData.nics);
  data.numaCount = nicData.numaCount;
  return Ok();
}

void buildNodes(Topology& topo, const DiscoveryData& data) {
  int gpuCount = data.gpus.size();
  for (int i = 0; i < gpuCount; ++i) {
    int nodeId = topo.addNode({
        .type = NodeType::GPU,
        .name = "cuda:" + std::to_string(data.gpus[i].data.cudaDeviceId),
        .data = data.gpus[i].data,
    });
    topo.registerGpuNode(i, nodeId);
  }
  topology::addNicNodes(topo, data.nics, data.numaCount);
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
  topology::appendNicPciDevices(pciDevices, topo, data.nics);
  topology::addPciCpuEdges(topo, sysfs, pciDevices, numaCount);

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

  topology::addPciPeerEdges(topo, sysfs, pciDevices);

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

  topology::addInterCpuEdges(topo, numaCount);
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
#if !defined(UNIFLOW_CMAKE_BUILD) || defined(UNIFLOW_ENABLE_RDMA_TRANSPORT)
      ibvApi_(ibvApi ? std::move(ibvApi) : std::make_shared<IbvApi>()),
#else
      ibvApi_(std::move(ibvApi)),
#endif
      sysfsApi_(sysfsApi ? std::move(sysfsApi) : std::make_shared<SysfsApi>()) {
}

Status CudaTopologyDiscovery::discover(Topology& topology) {
  topology.clear();

  DiscoveryData data;
  CHECK_EXPR(
      discoverHardware(*cudaApi_, *nvmlApi_, ibvApi_.get(), *sysfsApi_, data));
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

#ifdef __HIP_PLATFORM_AMD__
constexpr const char* kBuiltinGpuPlatform = "amd";
#else
constexpr const char* kBuiltinGpuPlatform = "nvidia";
#endif

UNIFLOW_REGISTER_PLATFORM(
    builtinCudaPlatform,
    PlatformRegistration{
        .name = kBuiltinGpuPlatform,
        .createDeviceAdapter = []() -> Result<std::shared_ptr<DeviceAdapter>> {
          return std::make_shared<CudaDeviceAdapter>(
              std::make_shared<CudaApi>(), std::make_shared<CudaDriverApi>());
        },
        .createDiscoveryBackend =
            []() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
          return std::make_unique<CudaTopologyDiscovery>();
        },
    });

void ensureDefaultPlatformRegistration() {
  (void)builtinCudaPlatform;
}

} // namespace uniflow
