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
  EXPECT_EQ(ncclxCfg->alltoallvAlgo, NCCL_ALLTOALLV_ALGO::orig);

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

// ----- ibLazyConnect tests -----

TEST(ConfigHintsUT, LazyPeerInit_DefaultIsFalse) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_FALSE(NCCLX_CONFIG_FIELD(config, ibLazyConnect));
  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, LazyPeerInit_HintOverrides) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ibLazyConnect", "true");
  config.hints = &hints;
  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);
  ASSERT_NE(config.ncclxConfig, nullptr);
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, ibLazyConnect));
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
