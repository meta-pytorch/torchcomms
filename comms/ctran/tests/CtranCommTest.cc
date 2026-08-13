// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/CtranPipes.h"
#if defined(ENABLE_PRIMS)
#include "comms/prims/transport/nvl/MultimemNvlTransportConfig.h"
#endif
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

#if defined(ENABLE_PRIMS)
TEST(CtranBuildMultimemConfigTest, UsesDedicatedDepthAndBufferSize) {
  const auto savedMultimemDepth = MCCL_NVL_MULTIMEM_PIPELINE_DEPTH;
  const auto savedP2pDepth = NCCL_CTRAN_P2P_NVL_COPY_PIPELINE_DEPTH;
  auto restoreCvars = folly::makeGuard([&] {
    MCCL_NVL_MULTIMEM_PIPELINE_DEPTH = savedMultimemDepth;
    NCCL_CTRAN_P2P_NVL_COPY_PIPELINE_DEPTH = savedP2pDepth;
  });
  MCCL_NVL_MULTIMEM_PIPELINE_DEPTH = 3;
  NCCL_CTRAN_P2P_NVL_COPY_PIPELINE_DEPTH = 7;

  comms::prims::MultimemNvlTransportConfig config{};
  EXPECT_EQ(
      ctranBuildMultimemNvlTransportConfig(
          ctranPrimsConfig{.maxChannels = 4, .maxBlocks = 2},
          /*bufferSize=*/4096,
          /*nLocalRanks=*/4,
          config),
      commSuccess);
  EXPECT_EQ(config.pipelineDepth, 3);
  EXPECT_EQ(config.maxChannels, 4);
  EXPECT_EQ(config.maxBlocks, 2);
  EXPECT_EQ(config.perChannelSize, 1008);
  EXPECT_EQ(config.userSignalCount, 1);
}

TEST(CtranBuildMultimemConfigTest, RejectsZeroBufferSize) {
  const auto savedMultimemDepth = MCCL_NVL_MULTIMEM_PIPELINE_DEPTH;
  auto restoreCvars = folly::makeGuard(
      [&] { MCCL_NVL_MULTIMEM_PIPELINE_DEPTH = savedMultimemDepth; });
  MCCL_NVL_MULTIMEM_PIPELINE_DEPTH = 4;

  comms::prims::MultimemNvlTransportConfig config{};
  EXPECT_EQ(
      ctranBuildMultimemNvlTransportConfig(
          ctranPrimsConfig{.maxChannels = 8, .maxBlocks = 3},
          /*bufferSize=*/0,
          /*nLocalRanks=*/4,
          config),
      commInvalidArgument);
}

TEST(CtranBuildMultimemConfigTest, RejectsInvalidDedicatedDepth) {
  const auto savedMultimemDepth = MCCL_NVL_MULTIMEM_PIPELINE_DEPTH;
  auto restoreCvars = folly::makeGuard(
      [&] { MCCL_NVL_MULTIMEM_PIPELINE_DEPTH = savedMultimemDepth; });

  comms::prims::MultimemNvlTransportConfig config{};
  for (const size_t invalidDepth :
       {size_t{0},
        static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1}) {
    MCCL_NVL_MULTIMEM_PIPELINE_DEPTH = invalidDepth;
    EXPECT_EQ(
        ctranBuildMultimemNvlTransportConfig(
            ctranPrimsConfig{.maxChannels = 4, .maxBlocks = 2},
            /*bufferSize=*/4096,
            /*nLocalRanks=*/4,
            config),
        commInvalidArgument);
  }
}
#endif // defined(ENABLE_PRIMS)

} // namespace ctran::testing
