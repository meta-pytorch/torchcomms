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

} // namespace
} // namespace comms::prims::tests
