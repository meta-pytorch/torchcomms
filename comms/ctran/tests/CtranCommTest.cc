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
      .backends = {CommBackend::IB, CommBackend::NVL, CommBackend::SOCKET}};

  CtranComm comm(abort, config);
  EXPECT_EQ(comm.config_.backends.size(), 3);

  /// Explicitly create comm with false abort as first argument is unomittable
  CtranComm comm2(comms::fault_tolerance::createAbort(false));
  EXPECT_EQ(comm2.config_.backends.size(), 0);
}

TEST(CtranCommTest, PrimsPolicyIsPerCommunicator) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  const bool savedUsePipes = NCCL_CTRAN_USE_PIPES;
  NCCL_CTRAN_USE_PIPES = false;
  auto restoreUsePipes = folly::makeGuard(
      [savedUsePipes] { NCCL_CTRAN_USE_PIPES = savedUsePipes; });

  CtranComm mcclComm(abort, ctranConfig{.primsConfig = {.enablePrims = 1}});
  CtranComm ncclxComm(abort);

  EXPECT_TRUE(ctranPrimsEnabled(&mcclComm));
  EXPECT_FALSE(ctranPrimsEnabled(&ncclxComm));
}

// The transport binds its staging geometry at comm init while the collective
// resolves its launch geometry per call. Both go through these helpers so the
// two cannot disagree -- pin that hint-first / CVAR-second contract here.
TEST(CtranCommTest, PrimsGeometryResolvesHintBeforeCvar) {
  const int64_t savedChannels = MCCL_MAX_NCHANNELS;
  const int64_t savedBlocks = MCCL_MAX_NBLOCKS;
  auto restore = folly::makeGuard([savedChannels, savedBlocks] {
    MCCL_MAX_NCHANNELS = savedChannels;
    MCCL_MAX_NBLOCKS = savedBlocks;
  });
  MCCL_MAX_NCHANNELS = 32;
  MCCL_MAX_NBLOCKS = 16;

  // Unset (-1) falls back to the CVAR.
  const ctranPrimsConfig unset{};
  EXPECT_EQ(ctranPrimsResolvedMaxChannels(unset), 32);
  EXPECT_EQ(ctranPrimsResolvedMaxBlocks(unset), 16);

  // Set overrides the CVAR, independently per knob.
  ctranPrimsConfig channelsOnly{};
  channelsOnly.maxChannels = 8;
  EXPECT_EQ(ctranPrimsResolvedMaxChannels(channelsOnly), 8);
  EXPECT_EQ(ctranPrimsResolvedMaxBlocks(channelsOnly), 16);

  ctranPrimsConfig blocksOnly{};
  blocksOnly.maxBlocks = 12;
  EXPECT_EQ(ctranPrimsResolvedMaxChannels(blocksOnly), 32);
  EXPECT_EQ(ctranPrimsResolvedMaxBlocks(blocksOnly), 12);

  ctranPrimsConfig both{};
  both.maxChannels = 8;
  both.maxBlocks = 12;
  EXPECT_EQ(ctranPrimsResolvedMaxChannels(both), 8);
  EXPECT_EQ(ctranPrimsResolvedMaxBlocks(both), 12);
}

TEST(CtranCommTest, ExplicitPrimsDisableDoesNotAffectLegacyPolicy) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  const bool savedUsePipes = NCCL_CTRAN_USE_PIPES;
  NCCL_CTRAN_USE_PIPES = true;
  auto restoreUsePipes = folly::makeGuard(
      [savedUsePipes] { NCCL_CTRAN_USE_PIPES = savedUsePipes; });

  CtranComm mcclComm(abort, ctranConfig{.primsConfig = {.enablePrims = 0}});
  CtranComm ncclxComm(abort);

  EXPECT_FALSE(ctranPrimsEnabled(&mcclComm));
  EXPECT_TRUE(ctranPrimsEnabled(&ncclxComm));
}

} // namespace ctran::testing
