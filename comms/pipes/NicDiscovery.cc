// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/NicDiscovery.h"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <unistd.h>

#include <dirent.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <fstream>
#include <stdexcept>

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

} // namespace

// Static methods

std::string NicDiscovery::normalizePcieAddress(const std::string& pciBusId) {
  std::string result = pciBusId;
  for (char& c : result) {
    c = std::tolower(static_cast<unsigned char>(c));
  }
  return result;
}

std::string NicDiscovery::getCudaPciBusId(int cudaDevice) {
  char busId[32];
  cudaError_t err = cudaDeviceGetPCIBusId(busId, sizeof(busId), cudaDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get CUDA device PCIe bus ID: " +
        std::string(cudaGetErrorString(err)));
  }
  return std::string(busId);
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

std::pair<PathType, int> NicDiscovery::computePathType(
    const std::string& nicPcie,
    int nicNuma) const {
  // If different NUMA nodes, it's PATH_SYS
  if (gpuNumaNode_ >= 0 && nicNuma >= 0 && gpuNumaNode_ != nicNuma) {
    return {PathType::SYS, -1};
  }

  // Normalize NIC address and build its chain
  std::string nicNormalized = normalizePcieAddress(nicPcie);
  std::vector<std::string> nicChain = buildAncestorChain(nicNormalized);

  // Find common ancestor
  int nicHops = 0;
  for (const auto& ancestor : nicChain) {
    if (gpuAncestors_.count(ancestor)) {
      // Found common ancestor
      // Count hops from GPU to this ancestor
      int gpuHops = 0;
      for (const auto& g : gpuAncestorChain_) {
        if (g == ancestor) {
          break;
        }
        gpuHops++;
      }

      int totalHops = gpuHops + nicHops;

      // Heuristic based on PCI topology depth:
      // - 2 hops (GPU->switch->NIC) = PIX (same switch)
      // - 3-4 hops = PXB (multiple switches, same NUMA)
      // - More = PHB (through host bridge)
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
  if (gpuNumaNode_ >= 0 && gpuNumaNode_ == nicNuma) {
    // Same NUMA node but different PCI domains
    int nhops =
        static_cast<int>(gpuAncestorChain_.size() + nicChain.size()) + 2;
    return {PathType::NODE, nhops};
  }
  return {PathType::SYS, -1};
}

void NicDiscovery::initGpuTopology() {
  // Skip if already initialized
  if (!gpuPciBusId_.empty()) {
    return;
  }

  // Get GPU PCIe bus ID
  gpuPciBusId_ = getCudaPciBusId(cudaDevice_);
  gpuPcieNormalized_ = normalizePcieAddress(gpuPciBusId_);

  // Build GPU ancestor chain for topology comparison (O(1) lookups later)
  gpuAncestorChain_ = buildAncestorChain(gpuPcieNormalized_);
  gpuAncestors_ = std::unordered_set<std::string>(
      gpuAncestorChain_.begin(), gpuAncestorChain_.end());

  // Get GPU NUMA node using pre-normalized address
  std::string numaPath =
      "/sys/bus/pci/devices/" + gpuPcieNormalized_ + "/numa_node";
  std::ifstream numaFile(numaPath);
  if (numaFile.is_open()) {
    numaFile >> gpuNumaNode_;
  }

  LOG(INFO) << "NicDiscovery: GPU " << cudaDevice_ << " PCIe " << gpuPciBusId_
            << " NUMA " << gpuNumaNode_;
}

NicDiscovery::NicDiscovery(int cudaDevice, const std::string& ibHcaEnv)
    : cudaDevice_(cudaDevice), ibHcaParser_(ibHcaEnv) {
  if (!ibHcaParser_.empty()) {
    LOG(INFO) << "NicDiscovery: IB HCA filter with "
              << ibHcaParser_.entries().size() << " entries";
  }
  discover();
}

void NicDiscovery::discover() {
  // Initialize GPU topology for auto-discovery
  initGpuTopology();

  auto devices = listIbDevices();
  if (devices.empty()) {
    throw std::runtime_error("No IB devices found");
  }

  candidates_.clear();

  for (const auto& devName : devices) {
    // Skip NICs that don't pass the HCA filter
    if (!ibHcaParser_.matches(devName)) {
      LOG(INFO) << "NicDiscovery: skipping NIC " << devName
                << " due to IB HCA filter";
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

    LOG(INFO) << "NicDiscovery: NIC " << devName << " PCIe=" << nicPcie
              << " NUMA=" << nicNuma << " path=" << pathTypeToString(pathType)
              << " nhops=" << nhops << " bandwidth=" << bandwidth << " Gb/s";

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

  // Sort by (pathType ASC, bandwidth DESC) - stable sort preserves enum order
  std::stable_sort(
      candidates_.begin(),
      candidates_.end(),
      [](const NicCandidate& a, const NicCandidate& b) {
        if (a.pathType != b.pathType) {
          return static_cast<int>(a.pathType) < static_cast<int>(b.pathType);
        }
        return a.bandwidthGbps > b.bandwidthGbps;
      });

  // Log sorted candidates for debugging
  LOG(INFO) << "NicDiscovery: NIC candidates after sorting:";
  for (size_t i = 0; i < candidates_.size(); i++) {
    LOG(INFO) << "  [" << i << "] " << candidates_[i].name
              << " path=" << pathTypeToString(candidates_[i].pathType)
              << " bandwidth=" << candidates_[i].bandwidthGbps << " Gb/s"
              << " nhops=" << candidates_[i].nhops;
  }

  const NicCandidate& best = candidates_[0];
  LOG(INFO) << "NicDiscovery: best candidate NIC " << best.name << " for GPU "
            << gpuPciBusId_ << " (path=" << pathTypeToString(best.pathType)
            << ", bandwidth=" << best.bandwidthGbps << " Gb/s)"
            << " (numa=" << best.numaNode << ", nhops=" << best.nhops << ")";
}

} // namespace comms::pipes
