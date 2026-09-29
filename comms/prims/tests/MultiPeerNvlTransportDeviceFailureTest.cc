// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/futures/Future.h>

#include <memory>
#include <stdexcept>

#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims::tests {
namespace {

using StrictMockBootstrap =
    ::testing::StrictMock<meta::comms::testing::MockBootstrap>;

TEST(MultiPeerNvlTransportDeviceFailureTest, SelectionFailureRestoresDevice) {
  using ::testing::_;

  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "test requires a CUDA device";
  }

  int originalDevice = -1;
  CUDACHECK_TEST(cudaGetDevice(&originalDevice));
  auto mock = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
      .WillOnce([](void* buf, int, int, int) {
        auto* eligibility = static_cast<int*>(buf);
        EXPECT_EQ(eligibility[0], -1);
        eligibility[1] = 1;
        eligibility[2] = 1;
        return folly::makeSemiFuture(0);
      });

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/deviceCount,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  int currentDevice = -1;
  CUDACHECK_TEST(cudaGetDevice(&currentDevice));
  EXPECT_EQ(currentDevice, originalDevice);
  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  CUDACHECK_TEST(cudaGetDevice(&currentDevice));
  EXPECT_EQ(currentDevice, originalDevice);
}

TEST(MultiPeerNvlTransportDeviceFailureTest, PrepareIsIdempotentAndLocal) {
  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "test requires a CUDA device";
  }

  auto mock = std::make_shared<StrictMockBootstrap>();
  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/0,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_NO_THROW(transport.prepareExchange());
  EXPECT_NO_THROW(transport.prepareExchange());
}

TEST(
    MultiPeerNvlTransportDeviceFailureTest,
    PreparedExchangeRejectsLegacyExternalStagingWithoutBootstrap) {
  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "test requires a CUDA device";
  }

  auto mock = std::make_shared<StrictMockBootstrap>();
  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/0,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);
  ExternalStagingBuffers buffers;
  buffers.localBuffers.resize(3);
  buffers.remoteBuffers.resize(3);
  transport.setExternalDataBuffers(std::move(buffers));

  EXPECT_THROW(transport.prepareExchange(), std::invalid_argument);
}

TEST(
    MultiPeerNvlTransportDeviceFailureTest,
    PreparedExchangeFailurePoisonsWithoutRetryingBootstrap) {
  using ::testing::_;

  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "test requires a CUDA device";
  }

  auto mock = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*mock, allGather(_, _, 0, 3)).WillOnce([] {
    return folly::makeSemiFuture<int>(-1);
  });
  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/0,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  ASSERT_NO_THROW(transport.prepareExchange());
  EXPECT_THROW(transport.exchangePrepared(), std::runtime_error);
  EXPECT_THROW(transport.prepareExchange(), std::runtime_error);
  EXPECT_THROW(transport.exchangePrepared(), std::runtime_error);
}

TEST(
    MultiPeerNvlTransportDeviceFailureTest,
    PreparedExchangeRejectsMissingPreparation) {
  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "test requires a CUDA device";
  }

  auto mock = std::make_shared<StrictMockBootstrap>();
  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/0,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_THROW(transport.exchangePrepared(), std::logic_error);
}

TEST(
    MultiPeerNvlTransportDeviceFailureTest,
    WorkspaceRejectsMismatchedIdentityWithoutBootstrap) {
  auto mock = std::make_shared<StrictMockBootstrap>();
  int localStorage = 0;
  int otherStorage = 0;
  NvlMemExchangeWorkspace workspace(
      /*rank=*/0, /*nRanks=*/2, &localStorage);
  cudaIpcMemHandle_t handle{};

  EXPECT_THROW(
      static_cast<void>(nvlMemExchangeCudaIpcPrepared(
          *mock,
          /*rank=*/1,
          /*nRanks=*/2,
          &localStorage,
          handle,
          workspace)),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(nvlMemExchangeVmmPrepared(
          *mock,
          /*rank=*/0,
          /*nRanks=*/3,
          /*cuDev=*/0,
          /*localHandle=*/0,
          &localStorage,
          /*allocatedSize=*/4096,
          /*preferFabric=*/false,
          workspace)),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(nvlMemExchangeCudaIpcPrepared(
          *mock,
          /*rank=*/0,
          /*nRanks=*/2,
          &otherStorage,
          handle,
          workspace)),
      std::invalid_argument);
}

} // namespace
} // namespace comms::prims::tests
