// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <climits>
#include <cstdlib>

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comm.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/TestUtils.h"
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
// NCCL comm and its CtranComm mirror should carry the value.
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
  EXPECT_EQ(comm->ctranComm_->config_.trafficClass, kHintValue);
}

// When the hint is not set, ncclConfig_t.trafficClass stays at the upstream
// NCCL_CONFIG_UNDEF_INT sentinel (INT_MIN), and CtranComm mirrors that.
TEST_F(trafficClassTest, hintUnsetLeavesUndefSentinel) {
  EnvRAII ctranEnv(NCCL_CTRAN_ENABLE, true);
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));

  EXPECT_EQ(comm->config.trafficClass, NCCL_CONFIG_UNDEF_INT);
  ASSERT_NE(nullptr, comm->ctranComm_);
  EXPECT_EQ(comm->ctranComm_->config_.trafficClass, NCCL_CONFIG_UNDEF_INT);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
