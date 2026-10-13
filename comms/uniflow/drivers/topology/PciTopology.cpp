// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/topology/PciTopology.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ranges>
#include <utility>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

#include "comms/uniflow/drivers/sysfs/SysfsApi.h"

namespace uniflow::topology {

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
namespace {

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
} // namespace

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

/// Convert a sysfs max_link_speed string to Mbps per lane.
/// Handles both old ("16 GT/s") and new ("16.0 GT/s PCIe") kernel formats.
/// Encoding overhead is already baked into the returned value.
/// Returns 0 if the string is unrecognized; defaults to Gen3 (6000) like NCCL.
namespace {

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

} // namespace

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
namespace {

constexpr std::string_view kNodePrefix = "node";
} // namespace

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

/// Find the nearest common ancestor between two PCI ancestor chains.
/// Returns {indexInA, indexInB} or {-1, -1} if none found.
namespace {

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
  // NOLINTNEXTLINE(facebook-security-vulnerable-memcpy)
  std::memcpy(vendor + 4, &edx, 4);
  // NOLINTNEXTLINE(facebook-security-vulnerable-memcpy)
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

} // namespace

void addPciCpuEdges(
    Topology& topo,
    SysfsApi& sysfs,
    std::span<const PciDevice> devices,
    int numaCount) {
  // Connect each PCI device to its CPU node.
  // For NICs, cap bandwidth with the RDMA port speed (like NCCL's LINK_NET).
  // For virtual devices (empty ancestor chain), use port speed directly.
  for (const auto& dev : devices) {
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
}

void addPciPeerEdges(
    Topology& topo,
    SysfsApi& sysfs,
    std::span<const PciDevice> devices) {
  // Direct PCIe edges between devices sharing a PCIe switch.
  for (size_t i = 0; i < devices.size(); ++i) {
    for (size_t j = i + 1; j < devices.size(); ++j) {
      auto ancestor = findCommonAncestor(
          devices[i].ancestorChain, devices[j].ancestorChain);
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
          idxA == static_cast<int>(devices[i].ancestorChain.size()) - 1;
      bool isRootB =
          idxB == static_cast<int>(devices[j].ancestorChain.size()) - 1;
      bool multiSwitch = idxA > 2 || idxB > 2 || (isRootA != isRootB);
      PathType contrib = multiSwitch ? PathType::PXB : PathType::PIX;
      uint32_t bw = computePairBottleneckMBps(
          sysfs,
          devices[i].ancestorChain,
          devices[j].ancestorChain,
          idxA,
          idxB);
      topo.addLink(devices[i].nodeId, devices[j].nodeId, contrib, bw);
    }
  }
}

void addInterCpuEdges(Topology& topo, int numaCount) {
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

} // namespace uniflow::topology
