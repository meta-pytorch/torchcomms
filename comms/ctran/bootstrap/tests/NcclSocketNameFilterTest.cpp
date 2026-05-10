// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/bootstrap/NcclSocketNameFilter.h"

using namespace ctran::bootstrap;

// --- parseIfNameSpec tests ---

TEST(IfNameFilter, ParseEmpty) {
  auto filter = parseIfNameSpec("");
  EXPECT_TRUE(filter.entries.empty());
  EXPECT_FALSE(filter.negate);
  EXPECT_FALSE(filter.exactMatch);
}

TEST(IfNameFilter, ParseSinglePrefix) {
  auto filter = parseIfNameSpec("eth");
  ASSERT_EQ(filter.entries.size(), 1);
  EXPECT_EQ(filter.entries[0], "eth");
  EXPECT_FALSE(filter.negate);
  EXPECT_FALSE(filter.exactMatch);
}

TEST(IfNameFilter, ParseCommaSeparated) {
  auto filter = parseIfNameSpec("beth0,beth1,beth2");
  ASSERT_EQ(filter.entries.size(), 3);
  EXPECT_EQ(filter.entries[0], "beth0");
  EXPECT_EQ(filter.entries[1], "beth1");
  EXPECT_EQ(filter.entries[2], "beth2");
  EXPECT_FALSE(filter.negate);
  EXPECT_FALSE(filter.exactMatch);
}

TEST(IfNameFilter, ParseNegate) {
  auto filter = parseIfNameSpec("^docker,lo");
  ASSERT_EQ(filter.entries.size(), 2);
  EXPECT_EQ(filter.entries[0], "docker");
  EXPECT_EQ(filter.entries[1], "lo");
  EXPECT_TRUE(filter.negate);
  EXPECT_FALSE(filter.exactMatch);
}

TEST(IfNameFilter, ParseExactMatch) {
  auto filter = parseIfNameSpec("=eth0,eth1");
  ASSERT_EQ(filter.entries.size(), 2);
  EXPECT_EQ(filter.entries[0], "eth0");
  EXPECT_EQ(filter.entries[1], "eth1");
  EXPECT_FALSE(filter.negate);
  EXPECT_TRUE(filter.exactMatch);
}

TEST(IfNameFilter, ParseNegateExact) {
  auto filter = parseIfNameSpec("^=docker0");
  ASSERT_EQ(filter.entries.size(), 1);
  EXPECT_EQ(filter.entries[0], "docker0");
  EXPECT_TRUE(filter.negate);
  EXPECT_TRUE(filter.exactMatch);
}

// --- matchesIfName tests ---

TEST(IfNameFilter, MatchEmptyFilter) {
  auto filter = parseIfNameSpec("");
  EXPECT_TRUE(matchesIfName(filter, "eth0"));
  EXPECT_TRUE(matchesIfName(filter, "anything"));
}

TEST(IfNameFilter, MatchPrefixSingle) {
  auto filter = parseIfNameSpec("eth");
  EXPECT_TRUE(matchesIfName(filter, "eth0"));
  EXPECT_TRUE(matchesIfName(filter, "eth1"));
  EXPECT_TRUE(matchesIfName(filter, "ethernet"));
  EXPECT_FALSE(matchesIfName(filter, "lo"));
  EXPECT_FALSE(matchesIfName(filter, "beth0"));
}

TEST(IfNameFilter, MatchPrefixCommaSeparated) {
  // This is the exact bug scenario:
  // NCCL_SOCKET_IFNAME="beth0,beth1,beth2,beth3,beth4,beth5,beth6,beth7"
  auto filter =
      parseIfNameSpec("beth0,beth1,beth2,beth3,beth4,beth5,beth6,beth7");
  EXPECT_TRUE(matchesIfName(filter, "beth0"));
  EXPECT_TRUE(matchesIfName(filter, "beth1"));
  EXPECT_TRUE(matchesIfName(filter, "beth7"));
  EXPECT_FALSE(matchesIfName(filter, "beth8"));
  EXPECT_FALSE(matchesIfName(filter, "eth0"));
  EXPECT_FALSE(matchesIfName(filter, "lo"));
}

TEST(IfNameFilter, MatchExact) {
  auto filter = parseIfNameSpec("=eth0");
  EXPECT_TRUE(matchesIfName(filter, "eth0"));
  EXPECT_FALSE(matchesIfName(filter, "eth0:1"));
  EXPECT_FALSE(matchesIfName(filter, "eth00"));
  EXPECT_FALSE(matchesIfName(filter, "eth1"));
}

TEST(IfNameFilter, MatchExactCommaSeparated) {
  auto filter = parseIfNameSpec("=eth0,eth1");
  EXPECT_TRUE(matchesIfName(filter, "eth0"));
  EXPECT_TRUE(matchesIfName(filter, "eth1"));
  EXPECT_FALSE(matchesIfName(filter, "eth2"));
  EXPECT_FALSE(matchesIfName(filter, "eth0:1"));
}

TEST(IfNameFilter, MatchNegate) {
  auto filter = parseIfNameSpec("^docker,lo");
  EXPECT_FALSE(matchesIfName(filter, "docker0"));
  EXPECT_FALSE(matchesIfName(filter, "docker1"));
  EXPECT_FALSE(matchesIfName(filter, "lo"));
  EXPECT_TRUE(matchesIfName(filter, "eth0"));
  EXPECT_TRUE(matchesIfName(filter, "beth0"));
}

TEST(IfNameFilter, MatchNegateExact) {
  auto filter = parseIfNameSpec("^=docker0");
  EXPECT_FALSE(matchesIfName(filter, "docker0"));
  EXPECT_TRUE(matchesIfName(filter, "docker1"));
  EXPECT_TRUE(matchesIfName(filter, "eth0"));
}

// Bare "^" with no interface names should be a no-op (matches everything)
TEST(IfNameFilter, BarePrefixOnlyNotEffective) {
  auto filter = parseIfNameSpec("^");
  EXPECT_TRUE(filter.entries.empty());
  EXPECT_TRUE(filter.negate);
  EXPECT_TRUE(matchesIfName(filter, "eth0"));
  EXPECT_TRUE(matchesIfName(filter, "lo"));
  EXPECT_TRUE(matchesIfName(filter, "anything"));
}

// Edge case: prefix that is also a full name
TEST(IfNameFilter, MatchPrefixFullName) {
  auto filter = parseIfNameSpec("eth0");
  EXPECT_TRUE(matchesIfName(filter, "eth0"));
  // "eth0" as prefix also matches "eth0:1" or "eth0_alias"
  EXPECT_TRUE(matchesIfName(filter, "eth0:1"));
  EXPECT_TRUE(matchesIfName(filter, "eth0_alias"));
  EXPECT_FALSE(matchesIfName(filter, "eth1"));
}
