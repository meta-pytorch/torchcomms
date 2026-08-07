// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "comms/utils/cvars/nccl_cvars.h"
#include "nccl.h" // @manual

#include "meta/NcclxConfig.h" // @manual

// ----- ncclxParseCommConfig tests -----

TEST(ConfigHintsUT, NoHintsCreatesDefaults) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  // hints is (void*)NCCL_CONFIG_UNDEF_PTR by default
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  // ncclx::Config should be created with defaults
  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_EQ(ncclxCfg->commDesc, "undefined");
  EXPECT_TRUE(ncclxCfg->splitGroupRanks.empty());
  EXPECT_EQ(ncclxCfg->useCtran, false);
  EXPECT_EQ(ncclxCfg->usePatAvg, false);
  EXPECT_EQ(ncclxCfg->noLocal, false);
  EXPECT_EQ(ncclxCfg->sendrecvAlgo, NCCL_SENDRECV_ALGO::orig);
  EXPECT_EQ(ncclxCfg->allgatherAlgo, NCCL_ALLGATHER_ALGO::orig);
  EXPECT_EQ(ncclxCfg->allreduceAlgo, NCCL_ALLREDUCE_ALGO::orig);
  EXPECT_EQ(ncclxCfg->alltoallAlgo, NCCL_ALLTOALL_ALGO::orig);
  EXPECT_EQ(ncclxCfg->alltoallvAlgo, NCCL_ALLTOALLV_ALGO::orig);
  EXPECT_EQ(ncclxCfg->rmaAlgo, NCCL_RMA_ALGO::ctran);

  // Upstream NCCL fields should be untouched
  EXPECT_EQ(config.blocking, NCCL_CONFIG_UNDEF_INT);
  EXPECT_EQ(config.cgaClusterSize, NCCL_CONFIG_UNDEF_INT);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, HintsCreateNcclxConfig) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("commDesc", "test_desc");
  hints.set("fastInitMode", "1");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);

  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, commDesc), "test_desc");
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, fastInitMode));

  // Upstream NCCL fields should be untouched
  EXPECT_EQ(config.blocking, NCCL_CONFIG_UNDEF_INT);

  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, PrefixedKeysMatchBareKeys) {
  // Set hints using "ncclx::" prefix — should produce the same config
  // as bare keys (tested in HintsCreateNcclxConfig above).
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ncclx::commDesc", "test_desc");
  hints.set("ncclx::fastInitMode", "1");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);

  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, commDesc), "test_desc");
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, fastInitMode));

  // Also verify get() with prefixed key returns the same value
  std::string val;
  EXPECT_EQ(hints.get("ncclx::commDesc", val), ncclSuccess);
  EXPECT_EQ(val, "test_desc");
  // And get() with bare key still works
  EXPECT_EQ(hints.get("commDesc", val), ncclSuccess);
  EXPECT_EQ(val, "test_desc");

  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, OldFormatFlatFields) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  // Set fields via old format (directly on ncclConfig_t)
  config.commDesc = "old_desc";
  config.fastInitMode = 2;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);

  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, commDesc), "old_desc");
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, fastInitMode));

  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, DoubleParseReturnsError) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("commDesc", "first_call");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, commDesc), "first_call");

  // Second call must fail — ncclxParseCommConfig must be called exactly once
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument);

  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

// ----- splitGroupRanks tests -----

TEST(ConfigHintsUT, SplitGroupRanksSetViaHints) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("splitGroupRanks", "0,1,2,3");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  const std::vector<int> expected = {0, 1, 2, 3};
  EXPECT_EQ(ncclxCfg->splitGroupRanks, expected);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, SplitGroupRanksSingleRank) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("splitGroupRanks", "7");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  const std::vector<int> expected = {7};
  EXPECT_EQ(ncclxCfg->splitGroupRanks, expected);

  delete ncclxCfg;
}

// ----- ncclBuffSize tests -----

TEST(ConfigHintsUT, NcclBuffSizeSetViaHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ncclBuffSize", "8388608");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->ncclBuffSize.has_value());
  EXPECT_EQ(ncclxCfg->ncclBuffSize.value(), 8388608);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, NcclBuffSizeDefaultUnset) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ncclBuffSize.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, NcclBuffSizeRejectsNegative) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ncclBuffSize", "-1");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ncclBuffSize.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, NcclBuffSizeRejectsZero) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ncclBuffSize", "0");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ncclBuffSize.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, NcclBuffSizeRejectsInvalidString) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ncclBuffSize", "notanumber");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ncclBuffSize.has_value());

  delete ncclxCfg;
}

// ----- ibSplitDataOnQps tests -----

TEST(ConfigHintsUT, IbSplitDataOnQpsSetViaHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ibSplitDataOnQps", "1");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->ibSplitDataOnQps.has_value());
  EXPECT_EQ(ncclxCfg->ibSplitDataOnQps.value(), 1);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, IbSplitDataOnQpsAcceptsZero) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ibSplitDataOnQps", "0");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->ibSplitDataOnQps.has_value());
  EXPECT_EQ(ncclxCfg->ibSplitDataOnQps.value(), 0);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, IbSplitDataOnQpsRejectsInvalid) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ibSplitDataOnQps", "2");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ibSplitDataOnQps.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, IbSplitDataOnQpsDefaultUnset) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ibSplitDataOnQps.has_value());

  delete ncclxCfg;
}

// ----- ibQpsPerConnection tests -----

TEST(ConfigHintsUT, IbQpsPerConnectionSetViaHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ibQpsPerConnection", "4");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->ibQpsPerConnection.has_value());
  EXPECT_EQ(ncclxCfg->ibQpsPerConnection.value(), 4);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, IbQpsPerConnectionRejectsZero) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ibQpsPerConnection", "0");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ibQpsPerConnection.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, IbQpsPerConnectionRejectsNegative) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ibQpsPerConnection", "-1");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ibQpsPerConnection.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, IbQpsPerConnectionDefaultUnset) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ibQpsPerConnection.has_value());

  delete ncclxCfg;
}

// ----- deviceIbLazyConnect tests -----

TEST(ConfigHintsUT, LazyPeerInit_DefaultIsTrue) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, deviceIbLazyConnect));
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, LazyPeerInit_HintOverrides) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("deviceIbLazyConnect", "true");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, deviceIbLazyConnect));
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, UseCtranHintOverride) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("useCtran", "1");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, useCtran));
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, UsePatAvgHintOverride) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("usePatAvg", "true");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, usePatAvg));
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, NoLocalHintOverride) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("noLocal", "1");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, noLocal));
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, AllgatherAlgoHintOverride) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("allgatherAlgo", "ctring");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_EQ(
      NCCLX_CONFIG_FIELD(config, allgatherAlgo), NCCL_ALLGATHER_ALGO::ctring);
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, SendrecvAlgoHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("sendrecvAlgo", "ctran");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_EQ(
      NCCLX_CONFIG_FIELD(config, sendrecvAlgo), NCCL_SENDRECV_ALGO::ctran);
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, AllreduceAlgoHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("allreduceAlgo", "ctdirect");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_EQ(
      NCCLX_CONFIG_FIELD(config, allreduceAlgo), NCCL_ALLREDUCE_ALGO::ctdirect);
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, AlltoallvAlgoHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("alltoallvAlgo", "ctran");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_EQ(
      NCCLX_CONFIG_FIELD(config, alltoallvAlgo), NCCL_ALLTOALLV_ALGO::ctran);
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, AlltoallAlgoHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("alltoallAlgo", "ctran");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_EQ(
      NCCLX_CONFIG_FIELD(config, alltoallAlgo), NCCL_ALLTOALL_ALGO::ctran);
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, RmaAlgoHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("rmaAlgo", "orig");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, rmaAlgo), NCCL_RMA_ALGO::orig);
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, InvalidAlgoHintFallsBackToDefault) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("sendrecvAlgo", "invalid_algo");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, sendrecvAlgo), NCCL_SENDRECV_ALGO::orig);
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

// ----- Per-communicator Prims transport overrides -----
//
// These three hints override MCCL_CHANNEL_BUFFER_SIZE,
// MCCL_CHANNEL_PIPELINE_DEPTH and NCCL_CTRAN_USE_PIPES for a single
// communicator. `primsChannelBufferSize` is per-channel, per-direction --
// the same unit as the CVAR it overrides.

TEST(ConfigHintsUT, PrimsChannelBufferSizeSetViaHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("primsChannelBufferSize", "4194304");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->primsChannelBufferSize.has_value());
  EXPECT_EQ(ncclxCfg->primsChannelBufferSize.value(), 4194304U);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, PrimsChannelBufferSizeDefaultUnset) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->primsChannelBufferSize.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, PrimsChannelBufferSizeRejectsZero) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("primsChannelBufferSize", "0");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->primsChannelBufferSize.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, PrimsChannelBufferSizeRejectsInvalidString) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("primsChannelBufferSize", "four-megabytes");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->primsChannelBufferSize.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, PrimsChannelPipelineDepthSetViaHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("primsChannelPipelineDepth", "4");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->primsChannelPipelineDepth.has_value());
  EXPECT_EQ(ncclxCfg->primsChannelPipelineDepth.value(), 4);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, PrimsChannelPipelineDepthDefaultUnset) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->primsChannelPipelineDepth.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, PrimsChannelPipelineDepthRejectsZeroAndNegative) {
  for (const char* bad : {"0", "-4"}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("primsChannelPipelineDepth", bad);
    config.hints = &hints;

    EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

    auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
    EXPECT_FALSE(ncclxCfg->primsChannelPipelineDepth.has_value()) << bad;

    delete ncclxCfg;
  }
}

// enablePrims is tri-state: absent leaves the optional unset so
// ctranPrimsEnabled() falls back to NCCL_CTRAN_USE_PIPES; 0 is an explicit
// disable and must NOT be confused with absent.
TEST(ConfigHintsUT, EnablePrimsAcceptsBooleanSpellings) {
  const std::vector<std::pair<const char*, int64_t>> cases = {
      {"1", 1},
      {"0", 0},
      {"true", 1},
      {"false", 0},
      {"yes", 1},
      {"no", 0},
      {"t", 1},
      {"f", 0},
  };
  for (const auto& [text, expected] : cases) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("enablePrims", text);
    config.hints = &hints;

    EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

    auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
    EXPECT_TRUE(ncclxCfg->enablePrims.has_value()) << text;
    if (ncclxCfg->enablePrims.has_value()) {
      EXPECT_EQ(ncclxCfg->enablePrims.value(), expected) << text;
    }

    delete ncclxCfg;
  }
}

TEST(ConfigHintsUT, EnablePrimsDefaultUnset) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->enablePrims.has_value());

  delete ncclxCfg;
}

// The torchcomms bootstrap only forwards "ncclx::"-prefixed hints; Hints::set
// strips the prefix. A mismatch between the prefixed and bare spelling would
// silently drop the hint, so pin both to the same result.
TEST(ConfigHintsUT, PrimsPrefixedKeysMatchBareKeys) {
  ncclConfig_t prefixed = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints prefixedHints;
  prefixedHints.set("ncclx::enablePrims", "1");
  prefixedHints.set("ncclx::primsChannelBufferSize", "4194304");
  prefixedHints.set("ncclx::primsChannelPipelineDepth", "4");
  prefixedHints.set("ncclx::primsMaxChannels", "8");
  prefixedHints.set("ncclx::primsMaxBlocks", "12");
  prefixed.hints = &prefixedHints;
  EXPECT_EQ(ncclxParseCommConfig(&prefixed), ncclSuccess);

  ncclConfig_t bare = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints bareHints;
  bareHints.set("enablePrims", "1");
  bareHints.set("primsChannelBufferSize", "4194304");
  bareHints.set("primsChannelPipelineDepth", "4");
  bareHints.set("primsMaxChannels", "8");
  bareHints.set("primsMaxBlocks", "12");
  bare.hints = &bareHints;
  EXPECT_EQ(ncclxParseCommConfig(&bare), ncclSuccess);

  auto* p = static_cast<ncclx::Config*>(prefixed.ncclxConfig);
  auto* b = static_cast<ncclx::Config*>(bare.ncclxConfig);
  EXPECT_EQ(p->enablePrims, b->enablePrims);
  EXPECT_EQ(p->primsChannelBufferSize, b->primsChannelBufferSize);
  EXPECT_EQ(p->primsChannelPipelineDepth, b->primsChannelPipelineDepth);
  EXPECT_EQ(p->primsMaxChannels, b->primsMaxChannels);
  EXPECT_EQ(p->primsMaxBlocks, b->primsMaxBlocks);
  ASSERT_TRUE(p->primsChannelBufferSize.has_value());
  EXPECT_EQ(p->primsChannelBufferSize.value(), 4194304U);

  delete p;
  delete b;
}

// An unparseable enablePrims must leave the optional UNSET so every rank falls
// back to the same CVAR. Resolving a typo to an explicit disable on only the
// ranks carrying it is what produces a mismatched-transport comm-init hang.
TEST(ConfigHintsUT, EnablePrimsInvalidLeavesUnset) {
  for (const char* bad : {"maybe", "2x", ""}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("enablePrims", bad);
    config.hints = &hints;

    EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

    auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
    EXPECT_FALSE(ncclxCfg->enablePrims.has_value()) << bad;

    delete ncclxCfg;
  }
}

// std::stoull wraps a leading '-', so "-1" would otherwise parse as 2^64-1 and
// pass the positivity check.
TEST(ConfigHintsUT, PrimsChannelBufferSizeRejectsNegative) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("primsChannelBufferSize", "-1");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->primsChannelBufferSize.has_value());

  delete ncclxCfg;
}

// primsMaxChannels / primsMaxBlocks override MCCL_MAX_NCHANNELS /
// MCCL_MAX_NBLOCKS for one communicator. Both the transport (comm init) and
// the collective launch geometry resolve through the same helpers.
TEST(ConfigHintsUT, PrimsMaxChannelsAndBlocksSetViaHint) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("primsMaxChannels", "8");
  hints.set("primsMaxBlocks", "12");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->primsMaxChannels.has_value());
  EXPECT_EQ(ncclxCfg->primsMaxChannels.value(), 8);
  ASSERT_TRUE(ncclxCfg->primsMaxBlocks.has_value());
  EXPECT_EQ(ncclxCfg->primsMaxBlocks.value(), 12);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, PrimsMaxChannelsAndBlocksDefaultUnset) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->primsMaxChannels.has_value());
  EXPECT_FALSE(ncclxCfg->primsMaxBlocks.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, PrimsMaxChannelsAndBlocksRejectBadValues) {
  for (const char* key : {"primsMaxChannels", "primsMaxBlocks"}) {
    for (const char* bad : {"0", "-1", "lots"}) {
      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      ncclx::Hints hints;
      hints.set(key, bad);
      config.hints = &hints;

      EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

      auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
      EXPECT_FALSE(ncclxCfg->primsMaxChannels.has_value()) << key << "=" << bad;
      EXPECT_FALSE(ncclxCfg->primsMaxBlocks.has_value()) << key << "=" << bad;

      delete ncclxCfg;
    }
  }
}
