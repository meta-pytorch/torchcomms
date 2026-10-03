// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <climits>
#include <cstdlib>

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comm.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranIbConfig.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "nccl.h"

// End-to-end coverage for the per-comm trafficClass hint: creates a real
// communicator via ncclCommInitRankConfig with config.trafficClass set,
// then verifies the value propagates into both ncclComm->config and the
// CtranComm mirror populated by MetaFactory.

class trafficClassTest : public NcclxBaseTestFixture {
 public:
  trafficClassTest() = default;

 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp();
  }

  void TearDown() override {
    NcclxBaseTestFixture::TearDown();
  }
};

// When the hint is set on ncclConfig_t.trafficClass at comm init, both the
// NCCL comm and the CtranIbConfig assembled by MetaFactory carry the value.
TEST_F(trafficClassTest, hintPropagatesToCommAndCtran) {
  EnvRAII ctranEnv(NCCL_CTRAN_ENABLE, true);
  constexpr int kHintValue = 192;

  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  inputConfig.trafficClass = kHintValue;

  ncclx::test::NcclCommRAII comm(
      globalRank,
      numRanks,
      localRank,
      bootstrap_.get(),
      /*isMock=*/false,
      &inputConfig);
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));

  EXPECT_EQ(comm->config.trafficClass, kHintValue);
  ASSERT_NE(nullptr, comm->ctranComm_);
  EXPECT_EQ(
      comm->ctranComm_->config_.ibConfig.trafficClass, int64_t{kHintValue});
}

// When the hint is not set, ncclConfig_t.trafficClass stays at the upstream
// NCCL_CONFIG_UNDEF_INT sentinel (INT_MIN) and the CTran-side optional stays
// unset, so CtranIb resolves from the env settings.
TEST_F(trafficClassTest, hintUnsetLeavesUndefSentinel) {
  EnvRAII ctranEnv(NCCL_CTRAN_ENABLE, true);
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));

  EXPECT_EQ(comm->config.trafficClass, NCCL_CONFIG_UNDEF_INT);
  ASSERT_NE(nullptr, comm->ctranComm_);
  EXPECT_FALSE(comm->ctranComm_->config_.ibConfig.trafficClass.has_value());
}

// An out-of-range ncclConfig_t.trafficClass degrades to the env settings
// rather than aborting the job, and says so. This check lives at the config
// boundary now; below it an out-of-range optional is a caller bug and
// hard-errors.
TEST_F(trafficClassTest, outOfRangeHintLeavesCtranOptionalUnset) {
  EnvRAII ctranEnv(NCCL_CTRAN_ENABLE, true);

  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  inputConfig.trafficClass = 1024; // > 255

  ncclx::test::NcclCommRAII comm(
      globalRank,
      numRanks,
      localRank,
      bootstrap_.get(),
      /*isMock=*/false,
      &inputConfig);
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_);

  EXPECT_FALSE(comm->ctranComm_->config_.ibConfig.trafficClass.has_value());
  // net_ib still sees the value it was given; only the CTran side drops it.
  EXPECT_EQ(comm->config.trafficClass, 1024);
}

// The CTran-specific hint is more specific than the general
// NcclConfig.traffic_class, which also feeds the plain net_ib transport.
TEST_F(trafficClassTest, ctranIbHintBeatsGeneralTrafficClass) {
  EnvRAII ctranEnv(NCCL_CTRAN_ENABLE, true);

  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  inputConfig.trafficClass = 96;
  ncclx::Hints hints({{"ctranIbTrafficClass", "200"}});
  inputConfig.hints = &hints;

  ncclx::test::NcclCommRAII comm(
      globalRank,
      numRanks,
      localRank,
      bootstrap_.get(),
      /*isMock=*/false,
      &inputConfig);
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_);

  // net_ib still sees the general value.
  EXPECT_EQ(comm->config.trafficClass, 96);
  EXPECT_EQ(comm->ctranComm_->config_.ibConfig.trafficClass, int64_t{200});
}

// Every ctranIb* hint must reach the CtranIbConfig that MetaFactory assembles
// onto the CtranComm, which is what CtranMapper hands to the IB backend.
TEST_F(trafficClassTest, ctranIbHintsPropagateToCtranIbConfig) {
  EnvRAII ctranEnv(NCCL_CTRAN_ENABLE, true);

  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({
      {"ctranIbNumQps", "16"},
      {"ctranIbQpScalingTh", "262144"},
      {"ctranIbQpMsgs", "128"},
      {"ctranIbVcMode", "dqplb"},
      {"ctranIbMaxNumCqe", "4096"},
      {"ctranIbMaxNumNic", "1"},
      {"ctranIbEnableLocalFlush", "1"},
      {"ctranIbTrafficClass", "200"},
  });
  inputConfig.hints = &hints;

  ncclx::test::NcclCommRAII comm(
      globalRank,
      numRanks,
      localRank,
      bootstrap_.get(),
      /*isMock=*/false,
      &inputConfig);
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_);

  const CtranIbConfig expected{
      .numQps = 16,
      .qpScalingTh = 262144,
      .vcMode = NCCL_CTRAN_IB_VC_MODE::dqplb,
      .qpMsgs = 128,
      .enableLocalFlush = true,
      .maxNumCqe = 4096,
      .maxNumNic = 1,
      .trafficClass = 200,
  };
  EXPECT_EQ(comm->ctranComm_->config_.ibConfig, expected);
}

// With no ctranIb* hint, every field stays unset so the IB backend resolves
// from cvars/topology exactly as it does today.
TEST_F(trafficClassTest, ctranIbConfigUnsetByDefault) {
  EnvRAII ctranEnv(NCCL_CTRAN_ENABLE, true);
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_);

  EXPECT_EQ(comm->ctranComm_->config_.ibConfig, CtranIbConfig{});
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
