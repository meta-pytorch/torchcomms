// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/backends/ib/VcLayout.h"
#include "comms/ctran/utils/Exception.h"

using ctran::ib::VcLayout;
using Devs = std::vector<std::vector<int>>;

TEST(VcLayoutTest, LegacyMode_1VcPer2Nics) {
  const VcLayout layout(/*numNics=*/2, /*maxVcsPerPeer=*/1);
  const Devs expected{{0, 1}};
  EXPECT_EQ(layout.maxVcsPerPeer, 1);
  EXPECT_EQ(layout.maxVcsPerNic, 1);
  EXPECT_EQ(layout.vcToActiveDevices, expected);
}

TEST(VcLayoutTest, LegacyMode_1VcPer1Nic) {
  const VcLayout layout(/*numNics=*/1, /*maxVcsPerPeer=*/1);
  const Devs expected{{0}};
  EXPECT_EQ(layout.maxVcsPerPeer, 1);
  EXPECT_EQ(layout.maxVcsPerNic, 1);
  EXPECT_EQ(layout.vcToActiveDevices, expected);
}

TEST(VcLayoutTest, MultiVcMode_4VcsPer2Nics) {
  const VcLayout layout(/*numNics=*/2, /*maxVcsPerPeer=*/4);
  const Devs expected{{0}, {0}, {1}, {1}};
  EXPECT_EQ(layout.maxVcsPerPeer, 4);
  EXPECT_EQ(layout.maxVcsPerNic, 2);
  EXPECT_EQ(layout.vcToActiveDevices, expected);
}

TEST(VcLayoutTest, MultiVcMode_2VcsPer2Nics) {
  // Boundary: maxVcsPerPeer == numNics; pinned and striped both reduce to
  // one VC per NIC, maxVcsPerNic=1.
  const VcLayout layout(/*numNics=*/2, /*maxVcsPerPeer=*/2);
  const Devs expected{{0}, {1}};
  EXPECT_EQ(layout.maxVcsPerPeer, 2);
  EXPECT_EQ(layout.maxVcsPerNic, 1);
  EXPECT_EQ(layout.vcToActiveDevices, expected);
}

TEST(VcLayoutTest, FutureMode_2VcsPer4Nics) {
  const VcLayout layout(/*numNics=*/4, /*maxVcsPerPeer=*/2);
  const Devs expected{{0, 1}, {2, 3}};
  EXPECT_EQ(layout.maxVcsPerPeer, 2);
  EXPECT_EQ(layout.maxVcsPerNic, 1);
  EXPECT_EQ(layout.vcToActiveDevices, expected);
}

TEST(VcLayoutTest, FutureMode_1VcPer4Nics) {
  const VcLayout layout(/*numNics=*/4, /*maxVcsPerPeer=*/1);
  const Devs expected{{0, 1, 2, 3}};
  EXPECT_EQ(layout.maxVcsPerPeer, 1);
  EXPECT_EQ(layout.maxVcsPerNic, 1);
  EXPECT_EQ(layout.vcToActiveDevices, expected);
}

TEST(VcLayoutTest, InvalidConfig_NotDivisible_4VcsPer3Nics) {
  // 4 % 3 != 0, and 4 < 3 doesn't hold either.
  EXPECT_THROW(
      VcLayout(/*numNics=*/3, /*maxVcsPerPeer=*/4), ctran::utils::Exception);
}

TEST(VcLayoutTest, InvalidConfig_NotDivisible_3VcsPer4Nics) {
  // 4 % 3 != 0 (striped requires numNics % maxVcsPerPeer == 0).
  EXPECT_THROW(
      VcLayout(/*numNics=*/4, /*maxVcsPerPeer=*/3), ctran::utils::Exception);
}

TEST(VcLayoutTest, InvalidConfig_ZeroNics) {
  EXPECT_THROW(
      VcLayout(/*numNics=*/0, /*maxVcsPerPeer=*/1), ctran::utils::Exception);
}

TEST(VcLayoutTest, InvalidConfig_ZeroVcs) {
  EXPECT_THROW(
      VcLayout(/*numNics=*/2, /*maxVcsPerPeer=*/0), ctran::utils::Exception);
}

TEST(VcLayoutTest, DescribeContainsExpectedFields) {
  const VcLayout layout(/*numNics=*/2, /*maxVcsPerPeer=*/4);
  const auto desc = layout.describe();
  EXPECT_THAT(desc, ::testing::HasSubstr("maxVcsPerPeer=4"));
  EXPECT_THAT(desc, ::testing::HasSubstr("maxVcsPerNic=2"));
  EXPECT_THAT(desc, ::testing::HasSubstr("VC[0]"));
  EXPECT_THAT(desc, ::testing::HasSubstr("VC[1]"));
  EXPECT_THAT(desc, ::testing::HasSubstr("VC[2]"));
  EXPECT_THAT(desc, ::testing::HasSubstr("VC[3]"));
}
