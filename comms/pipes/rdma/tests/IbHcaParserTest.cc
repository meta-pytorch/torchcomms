// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/rdma/IbHcaParser.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace comms::pipes {
namespace {

// =============================================================================
// Construction tests
// =============================================================================

TEST(IbHcaParserTest, DefaultConstruction) {
  IbHcaParser parser;
  EXPECT_TRUE(parser.empty());
  EXPECT_EQ(parser.match_mode(), HcaMatchMode::PREFIX_INCLUDE);
  EXPECT_TRUE(parser.entries().empty());
}

TEST(IbHcaParserTest, EmptyString) {
  IbHcaParser parser("");
  EXPECT_TRUE(parser.empty());
  EXPECT_TRUE(parser.entries().empty());
}

TEST(IbHcaParserTest, SingleDevice) {
  IbHcaParser parser("mlx5_0");
  EXPECT_FALSE(parser.empty());
  EXPECT_EQ(parser.entries().size(), 1);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
  EXPECT_EQ(parser.entries()[0].port, -1);
}

TEST(IbHcaParserTest, MultipleDevices) {
  IbHcaParser parser("mlx5_0,mlx5_1,mlx5_2");
  EXPECT_EQ(parser.entries().size(), 3);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
  EXPECT_EQ(parser.entries()[1].name, "mlx5_1");
  EXPECT_EQ(parser.entries()[2].name, "mlx5_2");
}

TEST(IbHcaParserTest, DeviceWithPort) {
  IbHcaParser parser("mlx5_0:1");
  EXPECT_EQ(parser.entries().size(), 1);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
  EXPECT_EQ(parser.entries()[0].port, 1);
}

TEST(IbHcaParserTest, MultipleDevicesWithPorts) {
  IbHcaParser parser("mlx5_0:1,mlx5_1:2");
  EXPECT_EQ(parser.entries().size(), 2);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
  EXPECT_EQ(parser.entries()[0].port, 1);
  EXPECT_EQ(parser.entries()[1].name, "mlx5_1");
  EXPECT_EQ(parser.entries()[1].port, 2);
}

TEST(IbHcaParserTest, MixedPortAndNoPort) {
  IbHcaParser parser("mlx5_0:1,mlx5_1");
  EXPECT_EQ(parser.entries().size(), 2);
  EXPECT_EQ(parser.entries()[0].port, 1);
  EXPECT_EQ(parser.entries()[1].port, -1);
}

// =============================================================================
// Prefix parsing tests
// =============================================================================

TEST(IbHcaParserTest, PrefixIncludeDefault) {
  IbHcaParser parser("mlx5_0");
  EXPECT_EQ(parser.match_mode(), HcaMatchMode::PREFIX_INCLUDE);
}

TEST(IbHcaParserTest, ExactIncludePrefix) {
  IbHcaParser parser("=mlx5_0");
  EXPECT_EQ(parser.match_mode(), HcaMatchMode::EXACT_INCLUDE);
  EXPECT_EQ(parser.entries().size(), 1);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
}

TEST(IbHcaParserTest, PrefixExcludePrefix) {
  IbHcaParser parser("^mlx5_0");
  EXPECT_EQ(parser.match_mode(), HcaMatchMode::PREFIX_EXCLUDE);
  EXPECT_EQ(parser.entries().size(), 1);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
}

TEST(IbHcaParserTest, ExactExcludePrefix) {
  IbHcaParser parser("^=mlx5_0");
  EXPECT_EQ(parser.match_mode(), HcaMatchMode::EXACT_EXCLUDE);
  EXPECT_EQ(parser.entries().size(), 1);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
}

// =============================================================================
// PREFIX_INCLUDE matching tests
// =============================================================================

TEST(IbHcaParserTest, PrefixIncludeMatches) {
  IbHcaParser parser("mlx5");
  EXPECT_TRUE(parser.matches("mlx5_0"));
  EXPECT_TRUE(parser.matches("mlx5_10"));
  EXPECT_TRUE(parser.matches("mlx5"));
  EXPECT_FALSE(parser.matches("mlx4_0"));
  EXPECT_FALSE(parser.matches("ib0"));
}

TEST(IbHcaParserTest, PrefixIncludeMultipleEntries) {
  IbHcaParser parser("mlx5_0,mlx5_1");
  EXPECT_TRUE(parser.matches("mlx5_0"));
  EXPECT_TRUE(parser.matches("mlx5_1"));
  EXPECT_TRUE(parser.matches("mlx5_10")); // prefix match on mlx5_1
  EXPECT_FALSE(parser.matches("mlx5_2"));
}

TEST(IbHcaParserTest, PrefixIncludeWithPort) {
  IbHcaParser parser("mlx5_0:1");
  EXPECT_TRUE(parser.matches("mlx5_0", 1));
  EXPECT_FALSE(parser.matches("mlx5_0", 2)); // wrong port
  EXPECT_TRUE(parser.matches("mlx5_0")); // no port query, matches
  EXPECT_TRUE(parser.matches("mlx5_0", -1)); // default port, matches
}

// =============================================================================
// EXACT_INCLUDE matching tests
// =============================================================================

TEST(IbHcaParserTest, ExactIncludeMatches) {
  IbHcaParser parser("=mlx5_0");
  EXPECT_TRUE(parser.matches("mlx5_0"));
  EXPECT_FALSE(parser.matches("mlx5_00")); // not exact
  EXPECT_FALSE(parser.matches("mlx5_0x")); // not exact
  EXPECT_FALSE(parser.matches("mlx5")); // not exact
}

TEST(IbHcaParserTest, ExactIncludeMultiple) {
  IbHcaParser parser("=mlx5_0,mlx5_1");
  EXPECT_TRUE(parser.matches("mlx5_0"));
  EXPECT_TRUE(parser.matches("mlx5_1"));
  EXPECT_FALSE(parser.matches("mlx5_10")); // exact, no prefix
  EXPECT_FALSE(parser.matches("mlx5_2"));
}

// =============================================================================
// PREFIX_EXCLUDE matching tests
// =============================================================================

TEST(IbHcaParserTest, PrefixExcludeMatches) {
  IbHcaParser parser("^mlx5_1");
  EXPECT_TRUE(parser.matches("mlx5_0")); // not excluded
  EXPECT_FALSE(parser.matches("mlx5_1")); // excluded
  EXPECT_FALSE(parser.matches("mlx5_10")); // excluded (prefix match)
  EXPECT_TRUE(parser.matches("mlx5_2")); // not excluded
}

TEST(IbHcaParserTest, PrefixExcludeMultiple) {
  IbHcaParser parser("^mlx5_0,mlx5_1");
  EXPECT_FALSE(parser.matches("mlx5_0")); // excluded
  EXPECT_FALSE(parser.matches("mlx5_1")); // excluded
  EXPECT_FALSE(parser.matches("mlx5_10")); // excluded (prefix on mlx5_1)
  EXPECT_TRUE(parser.matches("mlx5_2")); // not excluded
  EXPECT_TRUE(parser.matches("ib0")); // not excluded
}

// =============================================================================
// EXACT_EXCLUDE matching tests
// =============================================================================

TEST(IbHcaParserTest, ExactExcludeMatches) {
  IbHcaParser parser("^=mlx5_1");
  EXPECT_TRUE(parser.matches("mlx5_0")); // not excluded
  EXPECT_FALSE(parser.matches("mlx5_1")); // exactly excluded
  EXPECT_TRUE(parser.matches("mlx5_10")); // not exactly mlx5_1
  EXPECT_TRUE(parser.matches("mlx5_2")); // not excluded
}

TEST(IbHcaParserTest, ExactExcludeMultiple) {
  IbHcaParser parser("^=mlx5_0,mlx5_1");
  EXPECT_FALSE(parser.matches("mlx5_0")); // excluded
  EXPECT_FALSE(parser.matches("mlx5_1")); // excluded
  EXPECT_TRUE(parser.matches("mlx5_10")); // not exact match
  EXPECT_TRUE(parser.matches("mlx5_2")); // not excluded
}

// =============================================================================
// Edge cases
// =============================================================================

TEST(IbHcaParserTest, EmptyFilterMatchesAll) {
  IbHcaParser parser;
  EXPECT_TRUE(parser.matches("anything"));
  EXPECT_TRUE(parser.matches("mlx5_0"));
  EXPECT_TRUE(parser.matches("mlx5_0", 1));
  EXPECT_TRUE(parser.matches(""));
}

TEST(IbHcaParserTest, WhitespaceHandling) {
  IbHcaParser parser("mlx5_0 , mlx5_1 , mlx5_2");
  EXPECT_EQ(parser.entries().size(), 3);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
  EXPECT_EQ(parser.entries()[1].name, "mlx5_1");
  EXPECT_EQ(parser.entries()[2].name, "mlx5_2");
}

TEST(IbHcaParserTest, PortZero) {
  IbHcaParser parser("mlx5_0:0");
  EXPECT_EQ(parser.entries()[0].port, 0);
  EXPECT_TRUE(parser.matches("mlx5_0", 0));
  EXPECT_FALSE(parser.matches("mlx5_0", 1));
}

TEST(IbHcaParserTest, MaxDeviceLimitExceeded) {
  // Build a string with kMaxHcaDevices + 1 entries
  std::string hcaStr;
  for (int i = 0; i <= IbHcaParser::kMaxHcaDevices; i++) {
    if (i > 0) {
      hcaStr += ",";
    }
    hcaStr += "mlx5_" + std::to_string(i);
  }
  EXPECT_THROW(IbHcaParser{hcaStr}, std::runtime_error);
}

TEST(IbHcaParserTest, MaxDeviceLimitExact) {
  // Build a string with exactly kMaxHcaDevices entries (should succeed)
  std::string hcaStr;
  for (int i = 0; i < IbHcaParser::kMaxHcaDevices; i++) {
    if (i > 0) {
      hcaStr += ",";
    }
    hcaStr += "mlx5_" + std::to_string(i);
  }
  EXPECT_NO_THROW(IbHcaParser{hcaStr});
}

// =============================================================================
// Accessor tests
// =============================================================================

TEST(IbHcaParserTest, MatchModeAccessor) {
  EXPECT_EQ(IbHcaParser().match_mode(), HcaMatchMode::PREFIX_INCLUDE);
  EXPECT_EQ(IbHcaParser("mlx5").match_mode(), HcaMatchMode::PREFIX_INCLUDE);
  EXPECT_EQ(IbHcaParser("=mlx5").match_mode(), HcaMatchMode::EXACT_INCLUDE);
  EXPECT_EQ(IbHcaParser("^mlx5").match_mode(), HcaMatchMode::PREFIX_EXCLUDE);
  EXPECT_EQ(IbHcaParser("^=mlx5").match_mode(), HcaMatchMode::EXACT_EXCLUDE);
}

TEST(IbHcaParserTest, EntriesAccessor) {
  IbHcaParser parser("mlx5_0:1,mlx5_1");
  const auto& entries = parser.entries();
  EXPECT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].name, "mlx5_0");
  EXPECT_EQ(entries[0].port, 1);
  EXPECT_EQ(entries[1].name, "mlx5_1");
  EXPECT_EQ(entries[1].port, -1);
}

TEST(IbHcaParserTest, EmptyAccessor) {
  EXPECT_TRUE(IbHcaParser().empty());
  EXPECT_TRUE(IbHcaParser("").empty());
  EXPECT_FALSE(IbHcaParser("mlx5_0").empty());
}

// =============================================================================
// Malformed input tests
// =============================================================================

TEST(IbHcaParserTest, MalformedPortNonNumeric) {
  // "mlx5_0:abc" — std::stoi throws std::invalid_argument
  EXPECT_THROW(IbHcaParser{"mlx5_0:abc"}, std::invalid_argument);
}

TEST(IbHcaParserTest, MalformedPortEmpty) {
  // "mlx5_0:" — empty string after colon, std::stoi throws
  EXPECT_THROW(IbHcaParser{"mlx5_0:"}, std::invalid_argument);
}

TEST(IbHcaParserTest, MalformedPortOverflow) {
  // Port number too large for int
  EXPECT_THROW(IbHcaParser{"mlx5_0:99999999999999999999"}, std::out_of_range);
}

TEST(IbHcaParserTest, TrailingComma) {
  // Trailing comma produces an empty token which is skipped
  IbHcaParser parser("mlx5_0,");
  EXPECT_EQ(parser.entries().size(), 1);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
}

TEST(IbHcaParserTest, LeadingComma) {
  // Leading comma produces an empty token which is skipped
  IbHcaParser parser(",mlx5_0");
  EXPECT_EQ(parser.entries().size(), 1);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
}

TEST(IbHcaParserTest, ConsecutiveCommas) {
  // Multiple commas produce empty tokens which are skipped
  IbHcaParser parser("mlx5_0,,mlx5_1");
  EXPECT_EQ(parser.entries().size(), 2);
  EXPECT_EQ(parser.entries()[0].name, "mlx5_0");
  EXPECT_EQ(parser.entries()[1].name, "mlx5_1");
}

TEST(IbHcaParserTest, OnlyCommas) {
  // All tokens are empty, so no entries are added
  IbHcaParser parser(",,,");
  EXPECT_TRUE(parser.empty());
}

TEST(IbHcaParserTest, PrefixOnly) {
  // Just "^" with no device names — no entries, matches everything
  IbHcaParser parser("^");
  EXPECT_TRUE(parser.empty());
  EXPECT_EQ(parser.match_mode(), HcaMatchMode::PREFIX_EXCLUDE);
  EXPECT_TRUE(parser.matches("mlx5_0"));
}

TEST(IbHcaParserTest, ExactPrefixOnly) {
  // Just "^=" with no device names
  IbHcaParser parser("^=");
  EXPECT_TRUE(parser.empty());
  EXPECT_EQ(parser.match_mode(), HcaMatchMode::EXACT_EXCLUDE);
  EXPECT_TRUE(parser.matches("mlx5_0"));
}

} // namespace
} // namespace comms::pipes
