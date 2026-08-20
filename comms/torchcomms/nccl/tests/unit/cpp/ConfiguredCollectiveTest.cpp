// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/torchcomms/nccl/ConfiguredCollective.hpp"

namespace torch::comms {
namespace {

namespace d = configured_collective_detail;

TEST(ConfiguredCollectiveTest, MapsKnownAlgoProtoPairs) {
  EXPECT_EQ(algoProtoToString(d::kAlgoRing, d::kProtoSimple), "RING_SIMPLE");
  EXPECT_EQ(algoProtoToString(d::kAlgoRing, d::kProtoLL), "RING_LL");
  EXPECT_EQ(algoProtoToString(d::kAlgoRing, d::kProtoLL128), "RING_LL128");
  EXPECT_EQ(algoProtoToString(d::kAlgoTree, d::kProtoSimple), "TREE_SIMPLE");
  EXPECT_EQ(algoProtoToString(d::kAlgoTree, d::kProtoLL), "TREE_LL");
  EXPECT_EQ(algoProtoToString(d::kAlgoTree, d::kProtoLL128), "TREE_LL128");
  EXPECT_EQ(
      algoProtoToString(d::kAlgoCollnetDirect, d::kProtoSimple),
      "COLLNET_DIRECT_SIMPLE");
  EXPECT_EQ(
      algoProtoToString(d::kAlgoCollnetChain, d::kProtoSimple),
      "COLLNET_CHAIN_SIMPLE");
  EXPECT_EQ(algoProtoToString(d::kAlgoNvls, d::kProtoSimple), "NVLS_SIMPLE");
  EXPECT_EQ(
      algoProtoToString(d::kAlgoNvlsTree, d::kProtoSimple), "NVLSTREE_SIMPLE");
  EXPECT_EQ(algoProtoToString(d::kAlgoPat, d::kProtoSimple), "PAT_SIMPLE");
}

TEST(ConfiguredCollectiveTest, UnregisteredPairsMapToEmpty) {
  // PAT and the CollNet/NVLS families only have SIMPLE kernels registered.
  EXPECT_EQ(algoProtoToString(d::kAlgoPat, d::kProtoLL), "");
  EXPECT_EQ(algoProtoToString(d::kAlgoNvls, d::kProtoLL128), "");
  EXPECT_EQ(algoProtoToString(999, 999), "");
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)

TEST(ConfiguredCollectiveTest, HintsPopulateCollConfigFields) {
  const std::unordered_map<std::string, std::string> hints{
      {"algSelection", "PAT_SIMPLE"},
      {"maxCTAs", "8"},
      {"min_ctas", "2"},
      {"userProfilerTag", "42"},
  };

  ncclCollConfig_t cfg = makeCollConfigFromHints(hints);
  EXPECT_STREQ(cfg.algSelection, "PAT_SIMPLE");
  EXPECT_EQ(cfg.maxCTAs, 8);
  EXPECT_EQ(cfg.minCTAs, 2);
  EXPECT_EQ(cfg.userProfilerTag, 42U);
  freeCollConfig(cfg);
}

TEST(ConfiguredCollectiveTest, LegacyAlgoProtocolPairSynthesizesSelection) {
  const std::unordered_map<std::string, std::string> hints{
      {"algo", std::to_string(d::kAlgoRing)},
      {"protocol", std::to_string(d::kProtoLL)},
  };

  ncclCollConfig_t cfg = makeCollConfigFromHints(hints);
  EXPECT_STREQ(cfg.algSelection, "RING_LL");
  freeCollConfig(cfg);
}

TEST(ConfiguredCollectiveTest, MalformedHintsAreIgnored) {
  const std::unordered_map<std::string, std::string> hints{
      {"maxCTAs", "not-a-number"},
      {"userProfilerTag", "also-not-a-number"},
  };

  ncclCollConfig_t cfg = makeCollConfigFromHints(hints);
  EXPECT_EQ(cfg.maxCTAs, NCCL_CONFIG_UNDEF_INT);
  freeCollConfig(cfg);
}

TEST(ConfiguredCollectiveTest, FreeCollConfigIsIdempotent) {
  ncclCollConfig_t cfg =
      makeCollConfigFromHints({{"algSelection", "TREE_LL128"}});
  freeCollConfig(cfg);
  freeCollConfig(cfg);
  EXPECT_EQ(cfg.algSelection, (const char*)NCCL_CONFIG_UNDEF_PTR);
}

#endif // NCCL_VERSION_CODE >= 2.31

} // namespace
} // namespace torch::comms
