// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <string>
#include <utility>
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

// ----- ctranIb* per-comm CTran IB override tests -----

TEST(ConfigHintsUT, CtranIbHintsSetAllFields) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ctranIbNumQps", "32");
  hints.set("ctranIbQpScalingTh", "1048576");
  hints.set("ctranIbQpMsgs", "256");
  hints.set("ctranIbVcMode", "dqplb");
  hints.set("ctranIbMaxNumCqe", "4096");
  hints.set("ctranIbMaxNumNic", "1");
  hints.set("ctranIbEnableLocalFlush", "1");
  hints.set("ctranIbTrafficClass", "200");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_EQ(ncclxCfg->ctranIbNumQps, 32);
  EXPECT_EQ(ncclxCfg->ctranIbQpScalingTh, 1048576u);
  EXPECT_EQ(ncclxCfg->ctranIbQpMsgs, 256);
  EXPECT_EQ(ncclxCfg->ctranIbVcMode, NCCL_CTRAN_IB_VC_MODE::dqplb);
  EXPECT_EQ(ncclxCfg->ctranIbMaxNumCqe, 4096);
  EXPECT_EQ(ncclxCfg->ctranIbMaxNumNic, 1);
  EXPECT_EQ(ncclxCfg->ctranIbEnableLocalFlush, true);
  EXPECT_EQ(ncclxCfg->ctranIbTrafficClass, 200);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, CtranIbHintsDefaultAllUnset) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_FALSE(ncclxCfg->ctranIbNumQps.has_value());
  EXPECT_FALSE(ncclxCfg->ctranIbQpScalingTh.has_value());
  EXPECT_FALSE(ncclxCfg->ctranIbQpMsgs.has_value());
  EXPECT_FALSE(ncclxCfg->ctranIbVcMode.has_value());
  EXPECT_FALSE(ncclxCfg->ctranIbMaxNumCqe.has_value());
  EXPECT_FALSE(ncclxCfg->ctranIbMaxNumNic.has_value());
  EXPECT_FALSE(ncclxCfg->ctranIbEnableLocalFlush.has_value());
  EXPECT_FALSE(ncclxCfg->ctranIbTrafficClass.has_value());

  delete ncclxCfg;
}

TEST(ConfigHintsUT, CtranIbVcModeAcceptsSpray) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ctranIbVcMode", "spray");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  EXPECT_EQ(ncclxCfg->ctranIbVcMode, NCCL_CTRAN_IB_VC_MODE::spray);

  delete ncclxCfg;
}

// An unrecognized mode fails parsing rather than silently selecting one.
TEST(ConfigHintsUT, CtranIbVcModeRejectsUnknownMode) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ctranIbVcMode", "roundrobin");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument);
}

// A bad value fails parsing rather than reverting to the cvar: a silently
// dropped override produces a run that looks tuned but is not.
TEST(ConfigHintsUT, CtranIbNumQpsRejectsNonPositive) {
  for (const char* bad : {"0", "-4", "notanumber"}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("ctranIbNumQps", bad);
    config.hints = &hints;

    EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument)
        << "value: " << bad;
  }
}

TEST(ConfigHintsUT, CtranIbQpMsgsRejectsNonPositive) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ctranIbQpMsgs", "0");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument);
}

// Zero is meaningful: it divides an operation evenly over the QPs.
TEST(ConfigHintsUT, CtranIbQpScalingThAcceptsZero) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ctranIbQpScalingTh", "0");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->ctranIbQpScalingTh.has_value());
  EXPECT_EQ(*ncclxCfg->ctranIbQpScalingTh, 0u);

  delete ncclxCfg;
}

TEST(ConfigHintsUT, CtranIbQpScalingThRejectsNegative) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ctranIbQpScalingTh", "-1");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument);
}

// resolveFirstIbvDevice hard-errors outside [1, DEVICES_PER_RANK]. Rejecting
// here reports the bad value by name instead of deeper in the transport.
TEST(ConfigHintsUT, CtranIbMaxNumNicRejectsOutOfRange) {
  for (const char* bad : {"0", "-1", "9999", "notanumber"}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("ctranIbMaxNumNic", bad);
    config.hints = &hints;

    EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument)
        << "value: " << bad;
  }
}

// An out-of-range traffic class would hard-error in resolveTrafficClass.
TEST(ConfigHintsUT, CtranIbTrafficClassRejectsOutOfRange) {
  for (const char* bad : {"-1", "256", "1024"}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("ctranIbTrafficClass", bad);
    config.hints = &hints;

    EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument)
        << "value: " << bad;
  }
}

TEST(ConfigHintsUT, CtranIbTrafficClassAcceptsRangeBounds) {
  for (const auto& [str, expected] :
       std::vector<std::pair<const char*, int64_t>>{{"0", 0}, {"255", 255}}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("ctranIbTrafficClass", str);
    config.hints = &hints;

    EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

    auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
    ASSERT_TRUE(ncclxCfg->ctranIbTrafficClass.has_value()) << "value: " << str;
    EXPECT_EQ(*ncclxCfg->ctranIbTrafficClass, expected);

    delete ncclxCfg;
  }
}

// Unset must stay unset so CtranIb keeps its arch-derived default.
TEST(ConfigHintsUT, CtranIbEnableLocalFlushAcceptsFalse) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ctranIbEnableLocalFlush", "0");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->ctranIbEnableLocalFlush.has_value());
  EXPECT_FALSE(*ncclxCfg->ctranIbEnableLocalFlush);

  delete ncclxCfg;
}

// A negative CQE cap is meaningful: it selects the device-reported maximum.
TEST(ConfigHintsUT, CtranIbMaxNumCqeAcceptsNonPositive) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("ctranIbMaxNumCqe", "-1");
  config.hints = &hints;

  EXPECT_EQ(ncclxParseCommConfig(&config), ncclSuccess);

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);
  ASSERT_TRUE(ncclxCfg->ctranIbMaxNumCqe.has_value());
  EXPECT_EQ(*ncclxCfg->ctranIbMaxNumCqe, -1);

  delete ncclxCfg;
}

// A present-but-unparseable bool fails config parsing. Falling back to unset
// would resolve to the arch default, which for local flush can be the opposite
// of what the caller asked for. Explicit true/false stay valid.
TEST(ConfigHintsUT, CtranIbEnableLocalFlushRejectsMalformed) {
  for (const char* bad : {"flase", "tru", "maybe", "2x", ""}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("ctranIbEnableLocalFlush", bad);
    config.hints = &hints;

    EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument)
        << "value: " << bad;
  }
}

// stoll stops at the first non-numeric character, so trailing text must be
// rejected rather than silently truncating the value.
TEST(ConfigHintsUT, CtranIbIntHintsRejectTrailingGarbage) {
  for (const char* bad : {"1e6", "8!", "4 2", "0x10"}) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("ctranIbQpScalingTh", bad);
    config.hints = &hints;

    EXPECT_EQ(ncclxParseCommConfig(&config), ncclInvalidArgument)
        << "value: " << bad;
  }
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
