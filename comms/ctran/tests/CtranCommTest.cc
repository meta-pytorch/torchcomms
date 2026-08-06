// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/CtranPipes.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::testing {

TEST(CtranCommTest, AbortUnavailable) {
  EXPECT_THROW(CtranComm comm(/*abort=*/nullptr), ctran::utils::Exception);
}

TEST(CtranCommTest, AbortAvailableAndEnabled) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  CtranComm comm(abort);
  ASSERT_NE(comm.getAbort(), nullptr);

  EXPECT_TRUE(comm.abortEnabled());

  comm.setAbort();

  EXPECT_TRUE(comm.testAbort());
}

TEST(CtranCommTest, AbortAvailableAndEnabledDoubleAbort) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  CtranComm comm(abort);
  ASSERT_NE(comm.getAbort(), nullptr);

  EXPECT_TRUE(comm.abortEnabled());

  comm.setAbort();
  comm.setAbort();

  EXPECT_TRUE(comm.testAbort());
}

TEST(CtranCommTest, AbortAvailableAndDisabled) {
  auto abort = ::comms::fault_tolerance::createAbort(/*enabled=*/false);
  CtranComm comm(abort);
  ASSERT_NE(comm.getAbort(), nullptr);

  EXPECT_FALSE(comm.abortEnabled());

  comm.setAbort();

  // disabled abort should not be set
  EXPECT_FALSE(comm.testAbort());
}

TEST(CtranCommTest, ctranCommConfigTest) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  ctranConfig config = {
      .backends = {CommBackend::IB, CommBackend::NVL, CommBackend::SOCKET},
      .pipesConfig = {.lazyChannels = true},
  };

  CtranComm comm(abort, config);
  EXPECT_EQ(comm.config_.backends.size(), 3);
  EXPECT_TRUE(comm.config_.pipesConfig.lazyChannels);

  /// Explicitly create comm with false abort as first argument is unomittable
  CtranComm comm2(comms::fault_tolerance::createAbort(false));
  EXPECT_EQ(comm2.config_.backends.size(), 0);
  EXPECT_FALSE(comm2.config_.pipesConfig.lazyChannels);
}

TEST(CtranCommTest, PrimsPolicyIsPerCommunicator) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  const bool savedUsePipes = NCCL_CTRAN_USE_PIPES;
  NCCL_CTRAN_USE_PIPES = false;
  auto restoreUsePipes = folly::makeGuard(
      [savedUsePipes] { NCCL_CTRAN_USE_PIPES = savedUsePipes; });

  CtranComm mcclComm(abort, ctranConfig{.pipesConfig = {.enablePrims = 1}});
  CtranComm ncclxComm(abort);

  EXPECT_TRUE(ctranPrimsEnabled(&mcclComm));
  EXPECT_FALSE(ctranPrimsEnabled(&ncclxComm));
}

TEST(CtranCommTest, ExplicitPrimsDisableDoesNotAffectLegacyPolicy) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  const bool savedUsePipes = NCCL_CTRAN_USE_PIPES;
  NCCL_CTRAN_USE_PIPES = true;
  auto restoreUsePipes = folly::makeGuard(
      [savedUsePipes] { NCCL_CTRAN_USE_PIPES = savedUsePipes; });

  CtranComm mcclComm(abort, ctranConfig{.pipesConfig = {.enablePrims = 0}});
  CtranComm ncclxComm(abort);

  EXPECT_FALSE(ctranPrimsEnabled(&mcclComm));
  EXPECT_TRUE(ctranPrimsEnabled(&ncclxComm));
}

} // namespace ctran::testing
