// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/rdma/NicDiscovery.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <dirent.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace comms::pipes {

namespace {

// List all IB device names from sysfs
std::vector<std::string> listIbDevices() {
  std::vector<std::string> devices;
  DIR* dir = opendir("/sys/class/infiniband/");
  if (dir == nullptr) {
    return devices;
  }
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    if (name != "." && name != "..") {
      devices.push_back(name);
    }
  }
  closedir(dir);
  return devices;
}

// Check if a port is active via sysfs
// The state file contains e.g. "4: ACTIVE", where 4 == IBV_PORT_ACTIVE
bool isPortActive(const std::string& devName, int port) {
  std::string path = "/sys/class/infiniband/" + devName + "/ports/" +
      std::to_string(port) + "/state";
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }
  int state = 0;
  file >> state;
  return state == 4; // 4 == ACTIVE
}

// Get port rate in Gb/s from sysfs
// The rate file contains e.g. "400 Gb/sec (4X NDR)"
int getPortRateGbps(const std::string& devName, int port) {
  std::string path = "/sys/class/infiniband/" + devName + "/ports/" +
      std::to_string(port) + "/rate";
  std::ifstream file(path);
  if (!file.is_open()) {
    return 0;
  }
  int rateGbps = 0;
  file >> rateGbps;
  if (file.fail()) {
    return 0;
  }
  return rateGbps;
}

// Get the parent PCI bridge of a device by traversing sysfs
// NOTE: pciAddr must already be normalized (lowercase)
std::string getPciParent(const std::string& pciAddr) {
  std::string path = "/sys/bus/pci/devices/" + pciAddr + "/..";
  char resolved[PATH_MAX];
  if (realpath(path.c_str(), resolved) == nullptr) {
    return "";
  }
  // Extract last component (parent PCI address)
  std::string fullPath(resolved);
  auto pos = fullPath.rfind('/');
  if (pos != std::string::npos) {
    std::string parent = fullPath.substr(pos + 1);
    // Check if it looks like a PCI device address (e.g., "0000:1b:00.0")
    // Reject PCI domain roots like "pci0000:00" which also contain ':'
    if (parent.find(':') != std::string::npos &&
        std::isxdigit(static_cast<unsigned char>(parent[0]))) {
      return parent;
    }
  }
  return "";
}

// Build ancestor chain for a PCIe address (already normalized)
// Depth-limited to avoid infinite loops from sysfs symlink cycles
constexpr int kMaxPcieDepth = 32;

std::vector<std::string> buildAncestorChain(const std::string& normalizedPcie) {
  std::vector<std::string> chain;
  std::string current = normalizedPcie;
  while (!current.empty() && chain.size() < kMaxPcieDepth) {
    chain.push_back(current);
    current = getPciParent(current);
  }
  return chain;
}

// Build ancestor chain from a raw sysfs path (e.g.,
// "/sys/devices/pci0009:00/0009:00:00.0/0009:01:00.0/0009:03:00.0")
// Returns PCI bus IDs in leaf-first order: ["0009:03:00.0", "0009:01:00.0",
// "0009:00:00.0"]
// Used for Data Direct NICs whose leaf bus IDs may not exist under
// /sys/bus/pci/devices/, making the normal getPciParent() sysfs walk fail.
std::vector<std::string> buildAncestorChainFromSysfsPath(
    const std::string& sysfsPath) {
  std::vector<std::string> chain;
  // Split on '/' and keep only components that look like PCI addresses
  // PCI addresses start with a hex digit (e.g., "0009:03:00.0")
  // Skip domain roots like "pci0000:00"
  size_t pos = 0;
  while (pos < sysfsPath.size()) {
    auto next = sysfsPath.find('/', pos);
    std::string component;
    if (next == std::string::npos) {
      component = sysfsPath.substr(pos);
      pos = sysfsPath.size();
    } else {
      component = sysfsPath.substr(pos, next - pos);
      pos = next + 1;
    }
    if (!component.empty() && component.find(':') != std::string::npos &&
        std::isxdigit(static_cast<unsigned char>(component[0]))) {
      chain.push_back(component);
    }
  }
  // Reverse to get leaf-first order
  std::reverse(chain.begin(), chain.end());
  return chain;
}

} // namespace

// Free functions

int getCurrentNumaNode() {
  unsigned cpu = 0;
  unsigned node = 0;
  if (syscall(SYS_getcpu, &cpu, &node, nullptr) == 0) {
    return static_cast<int>(node);
  }
  return -1;
}

// =============================================================================
// NicDiscovery (base class)
// =============================================================================

std::string NicDiscovery::normalizePcieAddress(const std::string& pciBusId) {
  std::string result = pciBusId;
  for (char& c : result) {
    c = std::tolower(static_cast<unsigned char>(c));
  }
  return result;
}

int NicDiscovery::getNumaNodeForPcie(const std::string& pciBusId) {
  std::string normalized = normalizePcieAddress(pciBusId);
  std::string numaPath = "/sys/bus/pci/devices/" + normalized + "/numa_node";
  std::ifstream numaFile(numaPath);
  if (!numaFile.is_open()) {
    return -1;
  }
  int numaNode = -1;
  numaFile >> numaNode;
  return numaNode;
}

int NicDiscovery::getNumaNodeForIbDev(const char* devName) {
  std::string numaPath =
      std::string("/sys/class/infiniband/") + devName + "/device/numa_node";
  std::ifstream numaFile(numaPath);
  if (!numaFile.is_open()) {
    return -1;
  }
  int numaNode = -1;
  numaFile >> numaNode;
  return numaNode;
}

std::string NicDiscovery::getPcieForIbDev(const char* devName) {
  std::string devPath =
      std::string("/sys/class/infiniband/") + devName + "/device";
  char linkBuf[PATH_MAX];
  ssize_t len = readlink(devPath.c_str(), linkBuf, sizeof(linkBuf) - 1);
  if (len <= 0) {
    return "";
  }
  linkBuf[len] = '\0';
  // linkBuf is like "../../../0000:18:00.0", extract the last component
  std::string path(linkBuf);
  auto pos = path.rfind('/');
  if (pos != std::string::npos) {
    return path.substr(pos + 1);
  }
  return path;
}

NicDiscovery::NicDiscovery(const std::string& ibHcaEnv)
    : ibHcaParser_(ibHcaEnv) {
  if (!ibHcaParser_.empty()) {
    spdlog::info(
        "NicDiscovery: IB HCA filter with {} entries",
        ibHcaParser_.entries().size());
  }
}

void NicDiscovery::discover() {
  auto devices = listIbDevices();
  if (devices.empty()) {
    throw std::runtime_error("No IB devices found");
  }

  candidates_.clear();

  for (const auto& devName : devices) {
    // Skip NICs that don't pass the HCA filter
    if (!ibHcaParser_.matches(devName)) {
      spdlog::info(
          "NicDiscovery: skipping NIC {} due to IB HCA filter", devName);
      continue;
    }

    // Check port 1 is active via sysfs
    if (!isPortActive(devName, 1)) {
      continue;
    }

    // Get bandwidth from sysfs rate file
    int bandwidth = getPortRateGbps(devName, 1);

    std::string nicPcie = getPcieForIbDev(devName.c_str());
    int nicNuma = getNumaNodeForIbDev(devName.c_str());
    auto [pathType, nhops] = computePathType(nicPcie, nicNuma);

    spdlog::info(
        "NicDiscovery: NIC {} PCIe={} NUMA={} path={} nhops={} bandwidth={} Gb/s",
        devName,
        nicPcie,
        nicNuma,
        pathTypeToString(pathType),
        nhops,
        bandwidth);

    candidates_.push_back(
        NicCandidate{devName, nicPcie, pathType, bandwidth, nicNuma, nhops});
  }

  if (candidates_.empty()) {
    std::string errMsg = "No suitable IB device found with active port";
    if (!ibHcaParser_.empty()) {
      errMsg +=
          " (IB HCA filter excluded all devices; check ibHca config value)";
    }
    throw std::runtime_error(errMsg);
  }

  sortCandidates();

  // Log sorted candidates for debugging
  spdlog::info("NicDiscovery: NIC candidates after sorting:");
  for (size_t i = 0; i < candidates_.size(); i++) {
    spdlog::info(
        "  [{}] {} path={} bandwidth={} Gb/s nhops={}",
        i,
        candidates_[i].name,
        pathTypeToString(candidates_[i].pathType),
        candidates_[i].bandwidthGbps,
        candidates_[i].nhops);
  }
}

void NicDiscovery::sortCandidates() {
  std::stable_sort(
      candidates_.begin(),
      candidates_.end(),
      [](const NicCandidate& a, const NicCandidate& b) {
        if (a.isDataDirect != b.isDataDirect) {
          return a.isDataDirect > b.isDataDirect;
        }
        if (a.pathType != b.pathType) {
          return static_cast<int>(a.pathType) < static_cast<int>(b.pathType);
        }
        return a.bandwidthGbps > b.bandwidthGbps;
      });
}

void NicDiscovery::logBestCandidate() {
  if (!candidates_.empty()) {
    const NicCandidate& best = candidates_[0];
    spdlog::info(
        "NicDiscovery: best candidate NIC {} for {} (path={}, bandwidth={} Gb/s, dd={}) (numa={}, nhops={})",
        best.name,
        anchorDescription(),
        pathTypeToString(best.pathType),
        best.bandwidthGbps,
        best.isDataDirect,
        best.numaNode,
        best.nhops);
  }
}

// =============================================================================
// GpuNicDiscovery
// =============================================================================

std::string GpuNicDiscovery::getCudaPciBusId(int cudaDevice) {
  char busId[32];
  cudaError_t err = cudaDeviceGetPCIBusId(busId, sizeof(busId), cudaDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get CUDA device PCIe bus ID: " +
        std::string(cudaGetErrorString(err)));
  }
  return std::string(busId);
}

GpuNicDiscovery::GpuNicDiscovery(
    int cudaDevice,
    const std::string& ibHcaEnv,
    DataDirectMode ddMode)
    : NicDiscovery(ibHcaEnv), cudaDevice_(cudaDevice), dataDirectMode_(ddMode) {
  initGpuTopology();
  discover();
  augmentWithDataDirect();
  logBestCandidate();
}

void GpuNicDiscovery::initGpuTopology() {
  anchorPciBusId_ = getCudaPciBusId(cudaDevice_);
  std::string normalized = normalizePcieAddress(anchorPciBusId_);

  // Build ancestor chain for topology comparison (O(1) lookups later)
  anchorAncestorChain_ = buildAncestorChain(normalized);
  anchorAncestors_ = std::unordered_set<std::string>(
      anchorAncestorChain_.begin(), anchorAncestorChain_.end());

  // Get NUMA node using pre-normalized address
  std::string numaPath = "/sys/bus/pci/devices/" + normalized + "/numa_node";
  std::ifstream numaFile(numaPath);
  if (numaFile.is_open()) {
    numaFile >> anchorNumaNode_;
  }

  spdlog::info(
      "NicDiscovery: GPU {} PCIe {} NUMA {}",
      cudaDevice_,
      anchorPciBusId_,
      anchorNumaNode_);
}

std::pair<PathType, int> GpuNicDiscovery::computePathType(
    const std::string& nicPcie,
    int nicNuma) const {
  std::string nicNormalized = normalizePcieAddress(nicPcie);
  std::vector<std::string> nicChain = buildAncestorChain(nicNormalized);
  return computePathType(nicChain, nicNuma);
}

std::pair<PathType, int> GpuNicDiscovery::computePathType(
    const std::vector<std::string>& nicAncestorChain,
    int nicNuma) const {
  // If different NUMA nodes, it's PATH_SYS
  if (anchorNumaNode_ >= 0 && nicNuma >= 0 && anchorNumaNode_ != nicNuma) {
    return {PathType::SYS, -1};
  }

  // Find common ancestor between GPU chain and NIC chain
  int nicHops = 0;
  for (const auto& ancestor : nicAncestorChain) {
    if (anchorAncestors_.count(ancestor)) {
      // Count hops from GPU to this ancestor
      int gpuHops = 0;
      for (const auto& g : anchorAncestorChain_) {
        if (g == ancestor) {
          break;
        }
        gpuHops++;
      }

      int totalHops = gpuHops + nicHops;

      if (totalHops <= 2) {
        return {PathType::PIX, totalHops};
      }
      if (totalHops <= 4) {
        return {PathType::PXB, totalHops};
      }
      return {PathType::PHB, totalHops};
    }
    nicHops++;
  }

  // No common ancestor found in PCI tree
  if (anchorNumaNode_ >= 0 && anchorNumaNode_ == nicNuma) {
    int nhops = static_cast<int>(
                    anchorAncestorChain_.size() + nicAncestorChain.size()) +
        2;
    return {PathType::NODE, nhops};
  }
  return {PathType::SYS, -1};
}

// Static methods for Data Direct detection

bool GpuNicDiscovery::isMlx5Supported(ibv_device* device) {
  return lazy_mlx5dv_is_supported(device) != 0;
}

bool GpuNicDiscovery::isDmaBufCapable(ibv_context* ctx) {
  struct ibv_pd* pd = nullptr;
  doca_error_t err = doca_verbs_wrapper_ibv_alloc_pd(ctx, &pd);
  if (err != DOCA_SUCCESS || !pd) {
    return false;
  }

  // Probe DMA-BUF support with a dummy call (fd=-1)
  // If not supported, errno will be EOPNOTSUPP or EPROTONOSUPPORT
  // If supported but invalid args, errno will be EBADF (which means supported)
  (void)lazy_ibv_reg_dmabuf_mr(
      pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  bool notSupported = (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  doca_verbs_wrapper_ibv_dealloc_pd(pd);
  return !notSupported;
}

bool GpuNicDiscovery::getDataDirectSysfsPath(
    ibv_context* ctx,
    std::string& path) {
#if !TORCHCOMMS_HAVE_IBVERBS || !TORCHCOMMS_HAVE_MLX5DV
  (void)ctx;
  (void)path;
  return false;
#else
  char buf[PATH_MAX];
  // Prepend "/sys" prefix
  constexpr const char* kSysPrefix = "/sys";
  int prefixLen = strlen(kSysPrefix);
  memcpy(buf, kSysPrefix, prefixLen);

  int rc = lazy_mlx5dv_get_data_direct_sysfs_path(
      ctx, buf + prefixLen, sizeof(buf) - prefixLen);
  if (rc != 0) {
    return false;
  }
  path = std::string(buf);
  return true;
#endif
}

void GpuNicDiscovery::augmentWithDataDirect() {
  if (dataDirectMode_ == DataDirectMode::Disabled) {
    return;
  }

  int numDevices = 0;
  struct ibv_device** deviceList = nullptr;
  doca_error_t docaRet =
      doca_verbs_wrapper_ibv_get_device_list(&numDevices, &deviceList);
  if (docaRet != DOCA_SUCCESS || !deviceList || numDevices == 0) {
    spdlog::warn("NicDiscovery: ibv_get_device_list() failed for DD probing");
    return;
  }

  // Build map of ibv_device* by name for quick lookup
  std::unordered_map<std::string, ibv_device*> devMap;
  for (int i = 0; i < numDevices; i++) {
    const char* devName = nullptr;
    doca_verbs_wrapper_ibv_get_device_name(deviceList[i], &devName);
    if (devName) {
      devMap[devName] = deviceList[i];
    }
  }

  std::vector<NicCandidate> ddCandidates;
  std::unordered_set<std::string> ddCapableNames;

  for (const auto& candidate : candidates_) {
    auto it = devMap.find(candidate.name);
    if (it == devMap.end()) {
      continue;
    }

    ibv_device* dev = it->second;
    if (!isMlx5Supported(dev)) {
      continue;
    }

    ibv_context* ctx = nullptr;
    docaRet = doca_verbs_wrapper_ibv_open_device(dev, &ctx);
    if (docaRet != DOCA_SUCCESS || !ctx) {
      continue;
    }

    bool ddCapable = false;
    if (isDmaBufCapable(ctx)) {
      std::string ddSysfsPath;
      if (getDataDirectSysfsPath(ctx, ddSysfsPath)) {
        ddCapable = true;
        ddCapableNames.insert(candidate.name);

        // Build ancestor chain from the DD sysfs path for topology computation
        auto ddAncestorChain = buildAncestorChainFromSysfsPath(ddSysfsPath);
        auto [pathType, nhops] =
            computePathType(ddAncestorChain, candidate.numaNode);

        NicCandidate ddCandidate;
        ddCandidate.name = candidate.name;
        ddCandidate.pcie = ddSysfsPath;
        ddCandidate.pathType = pathType;
        ddCandidate.bandwidthGbps = candidate.bandwidthGbps;
        ddCandidate.numaNode = candidate.numaNode;
        ddCandidate.nhops = nhops;
        ddCandidate.isDataDirect = true;
        ddCandidate.forceFlush = true;
        ddCandidates.push_back(std::move(ddCandidate));

        spdlog::info(
            "NicDiscovery: DD NIC {} sysfs={} path={} nhops={}",
            candidate.name,
            ddSysfsPath,
            pathTypeToString(pathType),
            nhops);
      }
    }

    doca_verbs_wrapper_ibv_close_device(ctx);

    if (!ddCapable) {
      spdlog::debug(
          "NicDiscovery: NIC {} does not support Data Direct", candidate.name);
    }
  }

  // Apply mode policy
  if (dataDirectMode_ == DataDirectMode::Only) {
    // Remove regular candidates that have DD variants
    candidates_.erase(
        std::remove_if(
            candidates_.begin(),
            candidates_.end(),
            [&ddCapableNames](const NicCandidate& c) {
              return ddCapableNames.count(c.name) > 0;
            }),
        candidates_.end());
  }

  // Append DD candidates
  for (auto& dd : ddCandidates) {
    candidates_.push_back(std::move(dd));
  }

  doca_verbs_wrapper_ibv_free_device_list(deviceList);

  sortCandidates();

  // Re-log sorted candidates
  spdlog::info("NicDiscovery: NIC candidates after Data Direct augmentation:");
  for (size_t i = 0; i < candidates_.size(); i++) {
    spdlog::info(
        "  [{}] {} path={} bandwidth={} Gb/s nhops={} dd={}",
        i,
        candidates_[i].name,
        pathTypeToString(candidates_[i].pathType),
        candidates_[i].bandwidthGbps,
        candidates_[i].nhops,
        candidates_[i].isDataDirect);
  }
}

std::string GpuNicDiscovery::anchorDescription() const {
  return "GPU " + anchorPciBusId_;
}

// =============================================================================
// CpuNicDiscovery
// =============================================================================

CpuNicDiscovery::CpuNicDiscovery(int numaNode, const std::string& ibHcaEnv)
    : NicDiscovery(ibHcaEnv) {
  std::string numaPath =
      "/sys/devices/system/node/node" + std::to_string(numaNode);
  if (access(numaPath.c_str(), F_OK) != 0) {
    throw std::invalid_argument(
        "Invalid NUMA node " + std::to_string(numaNode) + ": " + numaPath +
        " does not exist");
  }
  anchorNumaNode_ = numaNode;
  spdlog::info(
      "NicDiscovery: CPU-anchored discovery, NUMA node {}", anchorNumaNode_);
  discover();
  logBestCandidate();
}

std::pair<PathType, int> CpuNicDiscovery::computePathType(
    const std::string& /* nicPcie */,
    int nicNuma) const {
  if (anchorNumaNode_ >= 0 && nicNuma >= 0 && anchorNumaNode_ == nicNuma) {
    return {PathType::NODE, -1};
  }
  return {PathType::SYS, -1};
}

std::string CpuNicDiscovery::anchorDescription() const {
  return "CPU NUMA " + std::to_string(anchorNumaNode_);
}

} // namespace comms::pipes
