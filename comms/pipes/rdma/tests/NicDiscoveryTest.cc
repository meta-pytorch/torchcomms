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

} // namespace comms::pipes::tests
