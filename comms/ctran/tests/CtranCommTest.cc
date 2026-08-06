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

  CtranComm mcclComm(abort, ctranConfig{.pipesConfig = {.enablePrims = 1}});
  CtranComm ncclxComm(abort);

  EXPECT_TRUE(ctranPrimsEnabled(&mcclComm));
  EXPECT_FALSE(ctranPrimsEnabled(&ncclxComm));
}

TEST(CtranCommTest, ExplicitPrimsDisableDoesNotAffectProcessDefault) {
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

#if defined(ENABLE_PRIMS)
TEST(CtranApplyMultimemConfigTest, UsesFallbackConfig) {
  const comms::prims::MultimemNvlTransportConfig fallback{
      .dataBufferSize = 4096,
      .userSignalCount = 1,
      .pipelineDepth = 2,
      .maxGroups = 4,
  };
  comms::prims::MultiPeerNvlTransportConfig nvlConfig{};

  EXPECT_EQ(
      ctranApplyMultimemConfig(
          ctranPipesConfig{}, fallback, 8192, 4, nvlConfig),
      commSuccess);
  EXPECT_TRUE(nvlConfig.enableMultimem);
  EXPECT_EQ(nvlConfig.multimem, fallback);
}

TEST(CtranApplyMultimemConfigTest, UsesExplicitConfigExactly) {
  const comms::prims::MultimemNvlTransportConfig fallback{
      .dataBufferSize = 4096,
      .userSignalCount = 1,
      .pipelineDepth = 2,
      .maxGroups = 4,
  };
  const comms::prims::MultimemNvlTransportConfig explicitConfig{
      .dataBufferSize = 8192,
      .userSignalCount = 3,
      .pipelineDepth = 4,
      .maxGroups = 8,
  };
  const ctranPipesConfig pipesConfig{.multimemConfig = explicitConfig};
  comms::prims::MultiPeerNvlTransportConfig nvlConfig{};

  EXPECT_EQ(
      ctranApplyMultimemConfig(pipesConfig, fallback, 16384, 4, nvlConfig),
      commSuccess);
  EXPECT_TRUE(nvlConfig.enableMultimem);
  EXPECT_EQ(nvlConfig.multimem, explicitConfig);
}

TEST(CtranApplyMultimemConfigTest, ResolvesTopologyDataBufferSize) {
  const comms::prims::MultimemNvlTransportConfig explicitConfig{
      .dataBufferSize = 0,
      .userSignalCount = 1,
      .pipelineDepth = 2,
      .maxGroups = 4,
  };
  const ctranPipesConfig pipesConfig{.multimemConfig = explicitConfig};
  comms::prims::MultiPeerNvlTransportConfig nvlConfig{};

  EXPECT_EQ(
      ctranApplyMultimemConfig(pipesConfig, explicitConfig, 8192, 4, nvlConfig),
      commSuccess);
  EXPECT_EQ(nvlConfig.multimem.dataBufferSize, 8192);
}

TEST(CtranApplyMultimemConfigTest, RejectsInvalidConfigWithoutMutation) {
  const comms::prims::MultimemNvlTransportConfig invalidConfig{
      .dataBufferSize = 4096,
      .userSignalCount = 1,
      .pipelineDepth = 2,
      .maxGroups = 0,
  };
  const ctranPipesConfig pipesConfig{.multimemConfig = invalidConfig};
  comms::prims::MultiPeerNvlTransportConfig nvlConfig{};
  const auto original = nvlConfig;

  EXPECT_EQ(
      ctranApplyMultimemConfig(pipesConfig, invalidConfig, 4096, 4, nvlConfig),
      commInvalidArgument);
  EXPECT_EQ(nvlConfig.enableMultimem, original.enableMultimem);
  EXPECT_EQ(nvlConfig.multimem, original.multimem);
}

TEST(CtranApplyMultimemConfigTest, LeavesTwoRankTransportDisabled) {
  const comms::prims::MultimemNvlTransportConfig invalidConfig{
      .dataBufferSize = 4096,
      .userSignalCount = 1,
      .pipelineDepth = 2,
      .maxGroups = 0,
  };
  const ctranPipesConfig pipesConfig{.multimemConfig = invalidConfig};
  comms::prims::MultiPeerNvlTransportConfig nvlConfig{};

  EXPECT_EQ(
      ctranApplyMultimemConfig(pipesConfig, invalidConfig, 4096, 2, nvlConfig),
      commSuccess);
  EXPECT_FALSE(nvlConfig.enableMultimem);
}
#endif // defined(ENABLE_PRIMS)

} // namespace ctran::testing
