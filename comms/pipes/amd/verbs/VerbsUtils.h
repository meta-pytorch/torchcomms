// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <fmt/format.h>
#include <folly/logging/xlog.h>

#include <infiniband/verbs.h>

#include "nic/NicSelector.h" // @manual

namespace pipes_gda::tests {

// =============================================================================
// Constants
// =============================================================================

constexpr uint16_t kDefaultQueueSize = 2048;
constexpr uint8_t kDefaultHopLimit = 255;
constexpr uint8_t kDefaultPortNum = 1;

// GID index: use 3 for standard mlx5, 1 for FE NIC
#if defined(USE_FE_NIC)
constexpr int kDefaultGidIndex = 1;
#else
constexpr int kDefaultGidIndex = 3;
#endif

// =============================================================================
// IB Device Utilities
// =============================================================================

/**
 * Read a sysfs file and return its content as a string.
 */
inline std::string readSysfs(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) {
    return "";
  }
  char buf[256] = {};
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  // Trim trailing newline
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) {
    buf[--n] = '\0';
  }
  return std::string(buf);
}

/**
 * Get the NUMA node for a PCIe device.
 * @param pciAddr PCIe address in format "0000:1B:00.0" or "1B:00.0"
 * @return NUMA node ID, or -1 if not found
 */
inline int getNumaNode(const std::string& pciAddr) {
  // Normalize address to include domain if missing
  std::string addr = pciAddr;
  if (addr.length() < 12) { // "0000:XX:XX.X" is 12 chars
    addr = "0000:" + addr;
  }
  // Convert to lowercase for consistency
  for (char& c : addr) {
    c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
  }

  std::string path = "/sys/bus/pci/devices/" + addr + "/numa_node";
  std::string content = readSysfs(path);
  if (content.empty()) {
    return -1;
  }
  return std::stoi(content);
}

/**
 * Get the PCIe address for an InfiniBand device.
 * @param ibDevName IB device name (e.g., "mlx5_0")
 * @return PCIe address or empty string if not found
 */
inline std::string getIbDevicePciAddr(const std::string& ibDevName) {
  std::string symlinkPath = "/sys/class/infiniband/" + ibDevName + "/device";
  char resolvedPath[PATH_MAX] = {};
  if (realpath(symlinkPath.c_str(), resolvedPath) == nullptr) {
    return "";
  }
  // resolvedPath is like "/sys/devices/pci0000:00/.../0000:1b:00.0"
  // Extract the last component which is the PCIe address
  std::string path(resolvedPath);
  size_t lastSlash = path.rfind('/');
  if (lastSlash == std::string::npos) {
    return "";
  }
  return path.substr(lastSlash + 1);
}

/**
 * Parse PCIe address and extract bus number.
 * @param pciAddr PCIe address in format "0000:1B:00.0" or "1B:00.0"
 * @return Bus number (0-255), or -1 on parse error
 */
inline int parsePciBus(const std::string& pciAddr) {
  // Format: [domain:]bus:device.function
  // Examples: "0000:1B:00.0" or "1B:00.0"
  size_t colonPos = pciAddr.find(':');
  if (colonPos == std::string::npos) {
    return -1;
  }

  std::string busStr;
  size_t secondColon = pciAddr.find(':', colonPos + 1);
  if (secondColon != std::string::npos) {
    // Has domain: "0000:1B:00.0"
    busStr = pciAddr.substr(colonPos + 1, secondColon - colonPos - 1);
  } else {
    // No domain: "1B:00.0"
    busStr = pciAddr.substr(0, colonPos);
  }

  try {
    return std::stoi(busStr, nullptr, 16);
  } catch (...) {
    return -1;
  }
}

/**
 * Find the closest NIC to a GPU based on PCIe topology.
 *
 * Selection criteria (in order of priority):
 * 1. Same NUMA node as the GPU
 * 2. Closest PCIe bus number (proxy for physical proximity)
 *
 * @param gpuPciAddr GPU's PCIe address (e.g., "0000:1B:00.0")
 * @return Best NIC device name, or empty string if none found
 */
inline std::string findClosestNic(const std::string& gpuPciAddr) {
  int numDevs = 0;
  struct ibv_device** devList = ibv_get_device_list(&numDevs);
  if (!devList || numDevs == 0) {
    return "";
  }

  const char* vendorPrefix = pipes_gda::ActiveNicBackend::vendorPrefix();
  int gpuNuma = getNumaNode(gpuPciAddr);
  int gpuBus = parsePciBus(gpuPciAddr);

  std::string bestNic;
  int bestScore = INT_MIN; // Higher is better

  for (int i = 0; i < numDevs; i++) {
    const char* devName = ibv_get_device_name(devList[i]);
    if (strncmp(devName, vendorPrefix, strlen(vendorPrefix)) != 0) {
      continue;
    }

    std::string nicPciAddr = getIbDevicePciAddr(devName);
    if (nicPciAddr.empty()) {
      continue;
    }

    int nicNuma = getNumaNode(nicPciAddr);
    int nicBus = parsePciBus(nicPciAddr);

    // Score: NUMA match is worth 1000 points, bus proximity adds 0-255 points
    int score = 0;
    if (gpuNuma >= 0 && nicNuma >= 0 && gpuNuma == nicNuma) {
      score += 1000; // NUMA match is most important
    }
    if (gpuBus >= 0 && nicBus >= 0) {
      // Closer bus numbers suggest closer physical proximity
      // Max bus difference is 255, so invert to make closer = higher score
      score += 255 - std::abs(gpuBus - nicBus);
    }

    XLOGF(
        DBG,
        "NIC {} (pci={}, numa={}, bus={}) score={} for GPU (pci={}, numa={}, bus={})",
        devName,
        nicPciAddr,
        nicNuma,
        nicBus,
        score,
        gpuPciAddr,
        gpuNuma,
        gpuBus);

    if (score > bestScore) {
      bestScore = score;
      bestNic = devName;
    }
  }

  ibv_free_device_list(devList);

  if (!bestNic.empty()) {
    XLOGF(
        INFO,
        "Selected NIC {} for GPU {} (score={})",
        bestNic,
        gpuPciAddr,
        bestScore);
  }

  return bestNic;
}

/**
 * Open an IB device by name.
 * @param name Device name (e.g., "mlx5_0")
 * @return IB context or nullptr on failure
 */
inline struct ibv_context* openIbDevice(const std::string& name) {
  int numDevs = 0;
  struct ibv_device** devList = ibv_get_device_list(&numDevs);
  if (!devList || numDevs == 0) {
    return nullptr;
  }

  struct ibv_context* ctx = nullptr;
  for (int i = 0; i < numDevs; i++) {
    if (name == ibv_get_device_name(devList[i])) {
      ctx = ibv_open_device(devList[i]);
      break;
    }
  }
  ibv_free_device_list(devList);
  return ctx;
}

/**
 * Find the first device matching the selected NIC vendor.
 * @return Device name or empty string if not found
 */
inline std::string findFirstNicDevice() {
  int numDevs = 0;
  struct ibv_device** devList = ibv_get_device_list(&numDevs);
  if (!devList || numDevs == 0) {
    return "";
  }

  const char* vendorPrefix = pipes_gda::ActiveNicBackend::vendorPrefix();
  std::string name;
  for (int i = 0; i < numDevs; i++) {
    const char* devName = ibv_get_device_name(devList[i]);
    if (strncmp(devName, vendorPrefix, strlen(vendorPrefix)) == 0) {
      name = devName;
      break;
    }
  }
  ibv_free_device_list(devList);
  return name;
}

// Backward-compatible alias for mlx5 builds
inline std::string findFirstMlx5Device() {
  return findFirstNicDevice();
}

// Swap bytes for GPU mkey (required by PIPES_GDA_VERBS_MKEY_SWAPPED).
// Delegates to the active NIC backend.
inline uint32_t swapMkey(uint32_t key) {
  return pipes_gda::ActiveNicBackend::swapMkey(key);
}

} // namespace pipes_gda::tests
