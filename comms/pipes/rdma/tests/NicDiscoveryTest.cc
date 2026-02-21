// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

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

} // namespace comms::pipes::tests
