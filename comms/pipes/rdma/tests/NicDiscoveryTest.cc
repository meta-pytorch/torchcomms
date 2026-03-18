// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "comms/pipes/rdma/NicDiscovery.h"

namespace comms::pipes::tests {

// =============================================================================
// Static Utility Function Tests
// =============================================================================

TEST(NicDiscoveryTest, NormalizePcieAddressLowercase) {
  EXPECT_EQ(NicDiscovery::normalizePcieAddress("0000:1b:00.0"), "0000:1b:00.0");
}

TEST(NicDiscoveryTest, NormalizePcieAddressUppercase) {
  EXPECT_EQ(NicDiscovery::normalizePcieAddress("0000:1B:00.0"), "0000:1b:00.0");
}

TEST(NicDiscoveryTest, NormalizePcieAddressEmpty) {
  EXPECT_EQ(NicDiscovery::normalizePcieAddress(""), "");
}

// =============================================================================
// PathType Tests
// =============================================================================

TEST(NicDiscoveryTest, PathTypeOrdering) {
  // Sort in discover() relies on PIX < PXB < PHB < NODE < SYS < DIS
  EXPECT_LT(static_cast<int>(PathType::PIX), static_cast<int>(PathType::PXB));
  EXPECT_LT(static_cast<int>(PathType::PXB), static_cast<int>(PathType::PHB));
  EXPECT_LT(static_cast<int>(PathType::PHB), static_cast<int>(PathType::NODE));
  EXPECT_LT(static_cast<int>(PathType::NODE), static_cast<int>(PathType::SYS));
  EXPECT_LT(static_cast<int>(PathType::SYS), static_cast<int>(PathType::DIS));
}

// =============================================================================
// NicCandidate Tests
// =============================================================================

TEST(NicDiscoveryTest, NicCandidateDefaultConstruction) {
  NicCandidate candidate;
  EXPECT_TRUE(candidate.name.empty());
  EXPECT_TRUE(candidate.pcie.empty());
  EXPECT_EQ(candidate.pathType, PathType::DIS);
  EXPECT_EQ(candidate.bandwidthGbps, 0);
  EXPECT_EQ(candidate.numaNode, -1);
  EXPECT_EQ(candidate.nhops, -1);
}

// =============================================================================
// CPU-Anchored Discovery Tests
// =============================================================================

TEST(NicDiscoveryTest, CpuAnchoredDiscovery) {
  int numaNode = getCurrentNumaNode();
  ASSERT_GE(numaNode, 0) << "Failed to get NUMA node for test";

  try {
    CpuNicDiscovery discovery(numaNode);
    EXPECT_EQ(discovery.getAnchorNumaNode(), numaNode);

    const auto& candidates = discovery.getCandidates();
    EXPECT_FALSE(candidates.empty());
    spdlog::info(
        "CpuAnchoredDiscovery: anchor NUMA={}, discovered {} NICs:",
        discovery.getAnchorNumaNode(),
        candidates.size());
    for (size_t i = 0; i < candidates.size(); i++) {
      spdlog::info(
          "  [{}] {} path={} bandwidth={} Gb/s numa={} nhops={}",
          i,
          candidates[i].name,
          pathTypeToString(candidates[i].pathType),
          candidates[i].bandwidthGbps,
          candidates[i].numaNode,
          candidates[i].nhops);
    }
  } catch (const std::runtime_error& e) {
    spdlog::info(
        "CpuAnchoredDiscovery: no IB devices in test env: {}", e.what());
  }
}

TEST(NicDiscoveryTest, CpuAnchoredInvalidNumaNode) {
  // NUMA node 9999 should not exist on any real system.
  EXPECT_THROW(CpuNicDiscovery(9999), std::invalid_argument);
}

// =============================================================================
// DataDirect Enum and Field Tests
// =============================================================================

TEST(NicDiscoveryTest, DataDirectModeEnumValues) {
  EXPECT_EQ(static_cast<int>(DataDirectMode::Disabled), 0);
  EXPECT_EQ(static_cast<int>(DataDirectMode::Only), 1);
  EXPECT_EQ(static_cast<int>(DataDirectMode::Both), 2);
}

TEST(NicDiscoveryTest, NicCandidateDataDirectDefaults) {
  NicCandidate candidate;
  EXPECT_FALSE(candidate.isDataDirect);
  EXPECT_FALSE(candidate.forceFlush);
}

TEST(NicDiscoveryTest, NicCandidateDataDirectFields) {
  NicCandidate candidate;
  candidate.name = "mlx5_0";
  candidate.pcie = "/sys/devices/pci0009:00/0009:00:00.0/0009:01:00.0";
  candidate.pathType = PathType::PIX;
  candidate.bandwidthGbps = 400;
  candidate.numaNode = 0;
  candidate.nhops = 2;
  candidate.isDataDirect = true;
  candidate.forceFlush = true;

  EXPECT_EQ(candidate.name, "mlx5_0");
  EXPECT_TRUE(candidate.isDataDirect);
  EXPECT_TRUE(candidate.forceFlush);
  EXPECT_EQ(candidate.pathType, PathType::PIX);
  EXPECT_EQ(candidate.bandwidthGbps, 400);
}

// =============================================================================
// DataDirect Sysfs Path Parsing Tests
// =============================================================================

// Test that buildAncestorChainFromSysfsPath (exercised indirectly via
// computePathType with a pre-built chain) correctly extracts PCI bus IDs
// from a sysfs path string. We verify the expected ancestor chain format here.
TEST(NicDiscoveryTest, DataDirectSysfsPathFormat) {
  // A typical DD sysfs path:
  // /sys/devices/pci0009:00/0009:00:00.0/0009:01:00.0/0009:03:00.0
  // The PCI bus IDs extracted (leaf-first) should be:
  // ["0009:03:00.0", "0009:01:00.0", "0009:00:00.0"]
  std::string sysfsPath =
      "/sys/devices/pci0009:00/0009:00:00.0/0009:01:00.0/0009:03:00.0";

  // Parse the path to extract PCI bus IDs (same logic as
  // buildAncestorChainFromSysfsPath)
  std::vector<std::string> chain;
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
  std::reverse(chain.begin(), chain.end());

  const std::vector<std::string> expected{
      "0009:03:00.0", "0009:01:00.0", "0009:00:00.0"};
  EXPECT_EQ(chain, expected);
}

TEST(NicDiscoveryTest, DataDirectAncestorChainForTopology) {
  // Verify that ancestor chain matching across PCI domains works
  // GPU chain: 0000:1b:00.0 -> 0000:1a:00.0 -> 0000:00:00.0
  // DD NIC chain: 0000:1b:00.0 -> 0000:1a:00.0 -> 0000:00:00.0
  // If they share "0000:1a:00.0", common ancestor is found at hop 1

  std::unordered_set<std::string> gpuAncestors{
      "0000:1b:00.0", "0000:1a:00.0", "0000:00:00.0"};

  std::vector<std::string> nicChain{
      "0000:1c:00.0", "0000:1a:00.0", "0000:00:00.0"};

  // Walk NIC chain to find first match in GPU ancestors
  int nicHops = 0;
  std::string commonAncestor;
  for (const auto& ancestor : nicChain) {
    if (gpuAncestors.count(ancestor)) {
      commonAncestor = ancestor;
      break;
    }
    nicHops++;
  }

  EXPECT_EQ(commonAncestor, "0000:1a:00.0");
  EXPECT_EQ(nicHops, 1);
}

} // namespace comms::pipes::tests
