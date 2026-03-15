// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <string>
#include <vector>

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
  EXPECT_EQ(ncclxCfg->ncclAllGatherAlgo, "undefined");
  EXPECT_FALSE(ncclxCfg->lazyConnect);

  // Upstream NCCL fields should be untouched
  EXPECT_EQ(config.blocking, NCCL_CONFIG_UNDEF_INT);
  EXPECT_EQ(config.cgaClusterSize, NCCL_CONFIG_UNDEF_INT);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, HintsCreateNcclxConfig) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("commDesc", "test_desc");
  hints.set("lazyConnect", "1");
  hints.set("lazySetupChannels", "0");
  hints.set("fastInitMode", "1");
  hints.set("ncclAllGatherAlgo", "custom_algo");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);

  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, commDesc), "test_desc");
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, lazyConnect));
  EXPECT_FALSE(NCCLX_CONFIG_FIELD(config, lazySetupChannels));
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, fastInitMode));
  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, ncclAllGatherAlgo), "custom_algo");

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
  hints.set("ncclx::lazyConnect", "1");
  hints.set("ncclx::lazySetupChannels", "0");
  hints.set("ncclx::fastInitMode", "1");
  hints.set("ncclx::ncclAllGatherAlgo", "custom_algo");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);

  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, commDesc), "test_desc");
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, lazyConnect));
  EXPECT_FALSE(NCCLX_CONFIG_FIELD(config, lazySetupChannels));
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, fastInitMode));
  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, ncclAllGatherAlgo), "custom_algo");

  // Also verify get() with prefixed key returns the same value
  std::string val;
  EXPECT_EQ(hints.get("ncclx::commDesc", val), ncclSuccess);
  EXPECT_EQ(val, "test_desc");
  // And get() with bare key still works
  EXPECT_EQ(hints.get("commDesc", val), ncclSuccess);
  EXPECT_EQ(val, "test_desc");

  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, BoolHintFormats) {
  // Test various truthy values
  for (const char* trueVal :
       {"1", "yes", "YES", "Yes", "true", "TRUE", "True", "y", "Y", "t", "T"}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("lazyConnect", trueVal);
    config.hints = &hints;
    EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess) << trueVal;
    EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, lazyConnect)) << trueVal;
    delete static_cast<ncclx::Config*>(config.ncclxConfig);
  }
  // Test various falsy values
  for (const char* falseVal :
       {"0", "no", "NO", "No", "false", "FALSE", "False", "n", "N", "f", "F"}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("lazyConnect", falseVal);
    config.hints = &hints;
    EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess) << falseVal;
    EXPECT_FALSE(NCCLX_CONFIG_FIELD(config, lazyConnect)) << falseVal;
    delete static_cast<ncclx::Config*>(config.ncclxConfig);
  }
}

TEST(ConfigHintsUT, OldFormatFlatFields) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  // Set fields via old format (directly on ncclConfig_t)
  config.commDesc = "old_desc";
  config.lazyConnect = 1;
  config.fastInitMode = 2;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  ASSERT_NE(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
  ASSERT_NE(config.ncclxConfig, nullptr);

  EXPECT_EQ(NCCLX_CONFIG_FIELD(config, commDesc), "old_desc");
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, lazyConnect));
  EXPECT_TRUE(NCCLX_CONFIG_FIELD(config, fastInitMode));

  delete static_cast<ncclx::Config*>(config.ncclxConfig);
}

TEST(ConfigHintsUT, ConflictReturnsError) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  // Set lazyConnect in old format
  config.lazyConnect = 1;
  // Also set it in hints (new format)
  ncclx::Hints hints;
  hints.set("lazyConnect", "0");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument);

  // ncclxConfig should NOT have been created
  EXPECT_EQ(config.ncclxConfig, (void*)NCCL_CONFIG_UNDEF_PTR);
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
