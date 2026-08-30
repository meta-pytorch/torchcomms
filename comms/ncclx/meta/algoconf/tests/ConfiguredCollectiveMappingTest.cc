// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "meta/algoconf/ConfiguredCollective.h"
#include "meta/algoconf/InfoExt.h"

using ::ncclx::algoconf::adjustChunkSizeForExt;
using ::ncclx::algoconf::algoProtoToString;
using ::ncclx::algoconf::maybeInfoExtOverride;
using ::ncclx::algoconf::ncclInfoExt;

#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
using ::ncclx::algoconf::makeCollConfigFromExt;
using ::ncclx::algoconf::makeCollConfigFromHints;
#endif

// ---------------------------------------------------------------------------
// Version-agnostic tests
// ---------------------------------------------------------------------------

TEST(ConfiguredCollectiveTest, AlgoProtoToString) {
  EXPECT_EQ(
      algoProtoToString(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE), "RING_SIMPLE");
  EXPECT_EQ(algoProtoToString(NCCL_ALGO_RING, NCCL_PROTO_LL), "RING_LL");
  EXPECT_EQ(algoProtoToString(NCCL_ALGO_RING, NCCL_PROTO_LL128), "RING_LL128");
  EXPECT_EQ(
      algoProtoToString(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE), "TREE_SIMPLE");
  EXPECT_EQ(algoProtoToString(NCCL_ALGO_TREE, NCCL_PROTO_LL), "TREE_LL");
  EXPECT_EQ(algoProtoToString(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE), "PAT_SIMPLE");
  EXPECT_EQ(
      algoProtoToString(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE),
      "COLLNET_DIRECT_SIMPLE");
  EXPECT_EQ(
      algoProtoToString(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE), "NVLS_SIMPLE");
  // Unknown pair -> empty
  EXPECT_EQ(algoProtoToString(999, 999), "");
  // PAT only supports SIMPLE in 2.31 registry; LL should be empty
  EXPECT_EQ(algoProtoToString(NCCL_ALGO_PAT, NCCL_PROTO_LL), "");
}

TEST(ConfiguredCollectiveTest, AdjustChunkSizeForExt) {
  // No ext -> unchanged
  EXPECT_EQ(adjustChunkSizeForExt(std::nullopt, 1024), 1024);
  // Ext without quant seed -> unchanged
  ncclInfoExt ext1(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4, 8);
  EXPECT_EQ(adjustChunkSizeForExt(ext1, 1024), 1024);
  // Ext with quant seed -> doubled
  uint64_t seed = 42;
  ncclInfoExt ext2(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, 4, 8, std::nullopt, &seed);
  EXPECT_EQ(adjustChunkSizeForExt(ext2, 1024), 2048);
  EXPECT_EQ(adjustChunkSizeForExt(ext2, 0), 0);
}

TEST(ConfiguredCollectiveTest, MaybeInfoExtOverrideReturnsNulloptByDefault) {
  // Facade returns nullopt when no generic override is configured.
  // This keeps the 2.29/2.30 path as-is.
  auto ext = maybeInfoExtOverride(nullptr, nullptr);
  EXPECT_FALSE(ext.has_value());
}

// ---------------------------------------------------------------------------
// 2.31-only tests (guarded so they still compile on 2.30)
// ---------------------------------------------------------------------------

#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)

TEST(ConfiguredCollectiveTest, MakeCollConfigFromExtMapsFields) {
  ncclInfoExt ext(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4, 8);
  auto cfg = makeCollConfigFromExt(ext);
  ASSERT_NE(cfg.algSelection, nullptr);
  EXPECT_STREQ(cfg.algSelection, "RING_SIMPLE");
  EXPECT_EQ(cfg.maxCTAs, 4);
  // nWarps is dropped, so no field to check; just ensure cfg is valid
  EXPECT_EQ(cfg.size, sizeof(ncclCollConfig_t));
  EXPECT_EQ(cfg.magic, (unsigned int)NCCL_API_MAGIC);
  // cleanup strdup
  free((void*)cfg.algSelection);
}

TEST(ConfiguredCollectiveTest, MakeCollConfigFromExtPat) {
  ncclInfoExt ext(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, 2, 16);
  auto cfg = makeCollConfigFromExt(ext);
  ASSERT_NE(cfg.algSelection, nullptr);
  EXPECT_STREQ(cfg.algSelection, "PAT_SIMPLE");
  EXPECT_EQ(cfg.maxCTAs, 2);
  free((void*)cfg.algSelection);
}

TEST(ConfiguredCollectiveTest, MakeCollConfigFromExtClearsSelectionForOpDev) {
  // A custom device reduction op has no collConfig representation, so no
  // field of the returned config may carry an override — not the mappable
  // algorithm/protocol pair, and not nMaxChannels either.
  ncclDevRedOpFull opDev{};
  ncclInfoExt ext(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4, 8, opDev);
  auto cfg = makeCollConfigFromExt(ext);
  EXPECT_EQ(cfg.algSelection, (const char*)NCCL_CONFIG_UNDEF_PTR);
  EXPECT_EQ(cfg.maxCTAs, NCCL_CONFIG_UNDEF_INT);
}

TEST(ConfiguredCollectiveTest, MakeCollConfigFromExtEmpty) {
  auto cfg = makeCollConfigFromExt(std::nullopt);
  EXPECT_EQ(cfg.algSelection, (const char*)NCCL_CONFIG_UNDEF_PTR);
  EXPECT_EQ(cfg.maxCTAs, NCCL_CONFIG_UNDEF_INT);
}

TEST(ConfiguredCollectiveTest, MakeCollConfigFromHints) {
  std::unordered_map<std::string, std::string> hints = {
      {"algSelection", "TREE_LL"},
      {"maxCTAs", "8"},
      {"minCTAs", "2"},
      {"nvlsCTAs", "4"},
      {"cgaClusterSize", "2"},
      {"CTAPolicy", "1"},
      {"userProfilerTag", "12345"},
  };
  auto cfg = makeCollConfigFromHints(hints);
  ASSERT_NE(cfg.algSelection, nullptr);
  EXPECT_STREQ(cfg.algSelection, "TREE_LL");
  EXPECT_EQ(cfg.maxCTAs, 8);
  EXPECT_EQ(cfg.minCTAs, 2);
  EXPECT_EQ(cfg.nvlsCTAs, 4);
  EXPECT_EQ(cfg.cgaClusterSize, 2);
  EXPECT_EQ(cfg.CTAPolicy, 1);
  EXPECT_EQ(cfg.userProfilerTag, 12345u);
  free((void*)cfg.algSelection);
}

TEST(ConfiguredCollectiveTest, MakeCollConfigFromHintsLegacyAlgoProtocol) {
  // Compat path: hints carry integer algo/protocol
  std::unordered_map<std::string, std::string> hints = {
      {"algo", std::to_string(NCCL_ALGO_RING)},
      {"protocol", std::to_string(NCCL_PROTO_SIMPLE)},
  };
  auto cfg = makeCollConfigFromHints(hints);
  ASSERT_NE(cfg.algSelection, nullptr);
  EXPECT_STREQ(cfg.algSelection, "RING_SIMPLE");
  free((void*)cfg.algSelection);
}

TEST(ConfiguredCollectiveTest, MakeCollConfigFromHintsIgnoresUnknown) {
  std::unordered_map<std::string, std::string> hints = {
      {"unknownKey", "someValue"},
      {"maxCTAs", "4"},
  };
  auto cfg = makeCollConfigFromHints(hints);
  EXPECT_EQ(cfg.maxCTAs, 4);
  // unknown key should not crash; algSelection stays UNDEF
  EXPECT_EQ(cfg.algSelection, (const char*)NCCL_CONFIG_UNDEF_PTR);
}

TEST(ConfiguredCollectiveTest, MakeCollConfigDropsWarps) {
  std::unordered_map<std::string, std::string> hints = {
      {"algSelection", "RING_SIMPLE"},
      {"nWarps", "16"},
      {"warps", "8"},
  };
  auto cfg = makeCollConfigFromHints(hints);
  ASSERT_NE(cfg.algSelection, nullptr);
  EXPECT_STREQ(cfg.algSelection, "RING_SIMPLE");
  // warps hint is dropped, not mapped to any collConfig field
  free((void*)cfg.algSelection);
}

#endif // NCCL_VERSION_CODE >= 2.31
