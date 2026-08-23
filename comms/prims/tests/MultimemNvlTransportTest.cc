// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <folly/futures/Future.h>
#include <folly/init/Init.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/core/SignalState.cuh"
#include "comms/prims/memory/GpuMemHandler.h"
#include "comms/prims/memory/MultimemHandler.h"
#include "comms/prims/tests/MultimemNvlTransportTest.cuh"
#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"
#include "comms/prims/transport/nvl/MultimemNvlTransport.h"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims::tests {

namespace {

std::shared_ptr<meta::comms::IBootstrap> makeBootstrap(
    const std::string& prefix) {
  return std::shared_ptr<meta::comms::IBootstrap>(
      meta::comms::createBootstrap(prefix));
}

std::vector<int> identityRankMap(int size) {
  std::vector<int> rankMap(static_cast<std::size_t>(size));
  for (int rank = 0; rank < size; ++rank) {
    rankMap[static_cast<std::size_t>(rank)] = rank;
  }
  return rankMap;
}

std::vector<int> rotatedRankMap(int size) {
  auto rankMap = identityRankMap(size);
  std::rotate(rankMap.begin(), rankMap.begin() + 1, rankMap.end());
  return rankMap;
}

// Collective check: returns true iff every rank reports that multimem is
// eligible on the given CUDA device. Tests use this to skip cleanly on
// non-NVLS hosts without leaving stragglers blocked in downstream collectives.
bool allRanksMultimemEligible(
    const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
    int rank,
    int nRanks,
    int cudaDevice) {
  std::vector<int> eligible(static_cast<std::size_t>(nRanks));
  eligible[static_cast<std::size_t>(rank)] =
      MultimemNvlTransport::isEligible(nRanks, cudaDevice) ? 1 : 0;
  auto rc =
      bootstrap->allGather(eligible.data(), sizeof(int), rank, nRanks).get();
  EXPECT_EQ(rc, 0);
  if (rc != 0) {
    return false;
  }
  for (const int v : eligible) {
    if (v == 0) {
      return false;
    }
  }
  return true;
}

MultimemNvlTransportConfig makeConfig(
    std::size_t dataBufferSize,
    uint32_t userSignalCount = 1,
    std::size_t pipelineDepth = 0,
    std::size_t maxChannels = 0,
    bool enableUnicastPeerViews = false) {
  const std::size_t effectiveMaxChannels =
      std::max<std::size_t>(1, maxChannels);
  return make_multimem_nvl_transport_config({
      .perChannelSize = dataBufferSize / effectiveMaxChannels,
      .pipelineDepth = pipelineDepth,
      .maxChannels = effectiveMaxChannels,
      .maxBlocks = maxChannels,
      .userSignalCount = userSignalCount,
      .enableUnicastPeerViews = enableUnicastPeerViews,
  });
}

using meta::comms::testing::MockBootstrap;
using StrictMockBootstrap = ::testing::StrictMock<MockBootstrap>;

class ScopedTestCudaDevice {
 public:
  explicit ScopedTestCudaDevice(int device) {
    CUDACHECK_TEST(cudaGetDevice(&originalDevice_));
    CUDACHECK_TEST(cudaSetDevice(device));
  }

  ~ScopedTestCudaDevice() {
    EXPECT_EQ(cudaSetDevice(originalDevice_), cudaSuccess);
  }

 private:
  int originalDevice_{-1};
};

// Builds a StrictMockBootstrap that, by default, forwards every IBootstrap
// API to `real`. Individual tests override specific methods via EXPECT_CALL
// to inject failures. StrictMock ensures that if MultimemNvlTransport (or
// its GpuMemHandler/MultimemHandler) ever grows a new bootstrap dependency
// we didn't account for, the test fails immediately with an "uninteresting
// mock function call" error -- locking down the bootstrap API surface.
std::shared_ptr<StrictMockBootstrap> makeDelegatingMock(
    const std::shared_ptr<meta::comms::IBootstrap>& real) {
  using ::testing::_;

  auto mock = std::make_shared<StrictMockBootstrap>();

  ON_CALL(*mock, allGather(_, _, _, _))
      .WillByDefault([real](void* buf, int len, int rank, int nRanks) {
        return real->allGather(buf, len, rank, nRanks);
      });
  ON_CALL(*mock, barrier(_, _)).WillByDefault([real](int rank, int nRanks) {
    return real->barrier(rank, nRanks);
  });
  ON_CALL(*mock, allGatherNvlDomain(_, _, _, _, _))
      .WillByDefault([real](
                         void* buf,
                         int len,
                         int r,
                         int n,
                         const std::vector<int>& rankMap) {
        return real->allGatherNvlDomain(buf, len, r, n, rankMap);
      });
  ON_CALL(*mock, barrierNvlDomain(_, _, _))
      .WillByDefault([real](int r, int n, const std::vector<int>& rankMap) {
        return real->barrierNvlDomain(r, n, rankMap);
      });
  ON_CALL(*mock, send(_, _, _, _))
      .WillByDefault([real](void* buf, int len, int peer, int tag) {
        return real->send(buf, len, peer, tag);
      });
  ON_CALL(*mock, recv(_, _, _, _))
      .WillByDefault([real](void* buf, int len, int peer, int tag) {
        return real->recv(buf, len, peer, tag);
      });

  return mock;
}

} // namespace

class MultimemNvlTransportTestFixture : public ::testing::Test,
                                        public meta::comms::DistBaseTest {
 protected:
  void SetUp() override {
    distSetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    distTearDown();
  }
};

TEST_F(
    MultimemNvlTransportTestFixture,
    EligibilityRequiresSupportAndThreeRanks) {
  EXPECT_FALSE(MultimemNvlTransport::isEligible(1, localRank));
  EXPECT_FALSE(MultimemNvlTransport::isEligible(2, localRank));
  EXPECT_EQ(
      MultimemNvlTransport::isEligible(3, localRank),
      GpuMemHandler::isMultimemSupported(localRank));
}

TEST_F(MultimemNvlTransportTestFixture, SignalFreeConfigurationConstructs) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_signal_free_construction");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(/*dataBufferSize=*/4096, /*userSignalCount=*/0));
  transport.exchange();
  const auto handle = transport.getDeviceTransport();

  EXPECT_EQ(transport.getAllocatedDataBufferSize(), 4096);
  EXPECT_EQ(transport.getAllocatedSignalBufferSize(), 0);
  EXPECT_NE(handle.localData, nullptr);
  EXPECT_NE(handle.multimemData, nullptr);
  EXPECT_TRUE(handle.userLocalSignals.empty());
  EXPECT_TRUE(handle.userMultimemSignals.empty());
  EXPECT_TRUE(handle.internalLocalSignals.empty());
  EXPECT_TRUE(handle.internalMultimemSignals.empty());
  EXPECT_EQ(handle.pipelineDepth, 0);
  EXPECT_EQ(handle.maxChannels, 1);
  EXPECT_EQ(handle.maxBlocks, 0);
  EXPECT_EQ(handle.signalsPerChannel, 0);
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(MultimemNvlTransportTestFixture, MultiPeerMultimemDisabled) {
  // No multimem-eligibility skip here: with enableMultimem=false this test
  // exercises only the disabled-path API (hasMultimemNvlTransport() == false,
  // getMultimemNvlTransportDevice() throws) and never touches the multimem
  // setup, so it must run on non-NVLS hosts too (skipping would drop coverage).
  auto bootstrap = makeBootstrap("multimem_nvl_transport_multi_peer_disabled");

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      // Disable the tile P2P channels (a nonzero maxNumChannels requires
      // pipelineDepth >= 1); this test checks only the disabled-path API.
      .maxNumChannels = 0,
      .memSharingMode = MemSharingMode::kCudaIpc,
      .enableMultimem = false,
  };
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  EXPECT_FALSE(transport.hasMultimemNvlTransport());
  EXPECT_NO_THROW(transport.exchange());
  EXPECT_THROW(
      static_cast<void>(transport.getMultimemNvlTransportDevice()),
      std::runtime_error);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerCollectivelyRejectsAsymmetricEligibility) {
  using ::testing::_;

  auto mock = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 2))
      .WillOnce([](void* buf, int, int, int) {
        auto* eligible = static_cast<int*>(buf);
        eligible[0] = 0;
        eligible[1] = 1;
        return folly::makeSemiFuture(0);
      });

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = makeConfig(4096, 1, 1, 1),
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/2,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_FALSE(transport.hasMultimemNvlTransport());
  EXPECT_FALSE(transport.initializeMultimemNvlTransportIfEligible());
  EXPECT_FALSE(transport.hasMultimemNvlTransport());
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerCollectivelyRejectsAsymmetricEnablement) {
  using ::testing::_;

  auto mock = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 2))
      .WillOnce([](void* buf, int, int, int) {
        auto* eligible = static_cast<int*>(buf);
        EXPECT_EQ(eligible[0], 0);
        eligible[1] = 1;
        return folly::makeSemiFuture(0);
      });

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = false,
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/2,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_FALSE(transport.hasMultimemNvlTransport());
  EXPECT_FALSE(transport.initializeMultimemNvlTransportIfEligible());
  EXPECT_FALSE(transport.initializeMultimemNvlTransportIfEligible());
  EXPECT_FALSE(transport.hasMultimemNvlTransport());
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerRemoteEligibilityQueryErrorPoisonsRetry) {
  using ::testing::_;

  auto mock = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
      .WillOnce([](void* buf, int, int, int) {
        auto* eligibility = static_cast<int*>(buf);
        EXPECT_GE(eligibility[0], 0);
        eligibility[1] = -1;
        eligibility[2] = 1;
        return folly::makeSemiFuture(0);
      });

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = makeConfig(4096, 1, 1, 1),
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/localRank,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerGetterDoesNotInitializeOrTouchBootstrap) {
  auto mock = std::make_shared<StrictMockBootstrap>();
  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = makeConfig(4096, 1, 1, 1),
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/2,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_THROW(
      static_cast<void>(transport.getMultimemNvlTransportDevice()),
      std::runtime_error);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerEligibilityAgreementFailurePoisonsRetry) {
  using ::testing::_;

  auto mock = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
      .WillOnce([](void*, int, int, int) { return folly::makeSemiFuture(1); });

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = makeConfig(4096, 1, 1, 1),
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/localRank,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerEligibilityAgreementExceptionPoisonsRetry) {
  using ::testing::_;

  auto mock = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
      .WillOnce([](void*, int, int, int) {
        return folly::makeSemiFuture<int>(std::runtime_error("injected"));
      });

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = makeConfig(4096, 1, 1, 1),
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/localRank,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerCollectivelyRejectsAsymmetricConstructionFailure) {
  using ::testing::_;
  using ::testing::InSequence;

  auto mock = std::make_shared<StrictMockBootstrap>();
  {
    InSequence sequence;
    EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
        .WillOnce([](void* buf, int, int, int) {
          auto* eligible = static_cast<int*>(buf);
          eligible[0] = 1;
          eligible[1] = 1;
          eligible[2] = 1;
          return folly::makeSemiFuture(0);
        });
    EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
        .WillOnce([](void* buf, int, int, int) {
          auto* ready = static_cast<int*>(buf);
          EXPECT_EQ(ready[0], 0);
          ready[1] = 1;
          ready[2] = 1;
          return folly::makeSemiFuture(0);
        });
  }

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = make_multimem_nvl_transport_config({
          .perChannelSize = 0,
          .pipelineDepth = 1,
          .maxChannels = 1,
          .maxBlocks = 1,
          .userSignalCount = 1,
      }),
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/localRank,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  EXPECT_FALSE(transport.hasMultimemNvlTransport());
  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerMultimemUsesConfiguredDeviceAndRestoresCallerDevice) {
  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2 || numRanks < 3) {
    GTEST_SKIP() << "test requires at least two GPUs and three ranks";
  }

  auto bootstrap = makeBootstrap("multimem_nvl_transport_owned_device");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  const int callerDevice = (localRank + 1) % deviceCount;
  std::unique_ptr<ScopedTestCudaDevice> callerDeviceGuard;
  {
    MultiPeerNvlTransportConfig config{
        .pipelineDepth = 0,
        .p2pSignalCount = 1,
        .maxNumChannels = 0,
        .enableMultimem = true,
        .multimem = make_multimem_nvl_transport_config({
            .perChannelSize = 4096,
            .pipelineDepth = 1,
            .maxChannels = 1,
            .maxBlocks = 1,
            .userSignalCount = 1,
        }),
    };
    MultiPeerNvlTransport transport(
        globalRank, numRanks, localRank, bootstrap, config);
    ASSERT_NO_THROW(transport.exchange());

    callerDeviceGuard = std::make_unique<ScopedTestCudaDevice>(callerDevice);
    int currentDevice = -1;
    CUDACHECK_TEST(cudaGetDevice(&currentDevice));
    EXPECT_EQ(currentDevice, callerDevice);
    ASSERT_TRUE(transport.initializeMultimemNvlTransportIfEligible());
    CUDACHECK_TEST(cudaGetDevice(&currentDevice));
    EXPECT_EQ(currentDevice, callerDevice);
    EXPECT_NO_THROW(transport.getMultimemNvlTransportDevice());
  }

  int currentDevice = -1;
  CUDACHECK_TEST(cudaGetDevice(&currentDevice));
  EXPECT_EQ(currentDevice, callerDevice);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerConstructionAgreementFailurePoisonsRetry) {
  using ::testing::_;
  using ::testing::InSequence;

  auto mock = std::make_shared<StrictMockBootstrap>();
  {
    InSequence sequence;
    EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
        .WillOnce([](void* buf, int, int, int) {
          auto* eligible = static_cast<int*>(buf);
          eligible[0] = 1;
          eligible[1] = 1;
          eligible[2] = 1;
          return folly::makeSemiFuture(0);
        });
    EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
        .WillOnce(
            [](void*, int, int, int) { return folly::makeSemiFuture(1); });
  }

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = make_multimem_nvl_transport_config({
          .perChannelSize = 4096,
          .pipelineDepth = 1,
          .maxChannels = 1,
          .maxBlocks = 1,
          .userSignalCount = 1,
      }),
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/localRank,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerConstructionAgreementExceptionPoisonsRetry) {
  using ::testing::_;
  using ::testing::InSequence;

  auto mock = std::make_shared<StrictMockBootstrap>();
  {
    InSequence sequence;
    EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
        .WillOnce([](void* buf, int, int, int) {
          auto* eligible = static_cast<int*>(buf);
          eligible[0] = 1;
          eligible[1] = 1;
          eligible[2] = 1;
          return folly::makeSemiFuture(0);
        });
    EXPECT_CALL(*mock, allGather(_, sizeof(int), 0, 3))
        .WillOnce([](void*, int, int, int) {
          return folly::makeSemiFuture<int>(
              folly::make_exception_wrapper<std::runtime_error>(
                  "readiness agreement failed"));
        });
  }

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = make_multimem_nvl_transport_config({
          .perChannelSize = 4096,
          .pipelineDepth = 1,
          .maxChannels = 1,
          .maxBlocks = 1,
          .userSignalCount = 1,
      }),
  };
  MultiPeerNvlTransport transport(
      /*myRank=*/0,
      /*nRanks=*/3,
      /*multimemCudaDevice=*/localRank,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      config);

  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerCollectivelyInitializesMultimemTransport) {
  auto bootstrap = makeBootstrap("multimem_nvl_transport_multi_peer_test");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kBytesPerRank = 4096;
  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      // This test exercises only the multimem path; disable the tile P2P
      // channels (a nonzero maxNumChannels requires pipelineDepth >= 1).
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = make_multimem_nvl_transport_config({
          .perChannelSize = kBytesPerRank * static_cast<std::size_t>(numRanks),
          .pipelineDepth = 1,
          .maxChannels = 1,
          .maxBlocks = 1,
          .userSignalCount = 1,
      }),
  };
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  EXPECT_FALSE(transport.hasMultimemNvlTransport());
  ASSERT_NO_THROW(transport.exchange());
  ASSERT_TRUE(transport.initializeMultimemNvlTransportIfEligible());
  EXPECT_TRUE(transport.hasMultimemNvlTransport());
  EXPECT_NO_THROW(transport.getMultimemNvlTransportDevice());
}

TEST_F(
    MultimemNvlTransportTestFixture,
    MultiPeerConstructionFailureDoesNotStrandPeers) {
  auto bootstrap = makeBootstrap("multimem_nvl_transport_construction_failure");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 0,
      .p2pSignalCount = 1,
      .maxNumChannels = 0,
      .enableMultimem = true,
      .multimem = make_multimem_nvl_transport_config({
          .perChannelSize =
              globalRank == 0 ? std::size_t{0} : std::size_t{4096},
          .pipelineDepth = 1,
          .maxChannels = 1,
          .maxBlocks = 1,
          .userSignalCount = 1,
      }),
  };
  MultiPeerNvlTransport transport(
      globalRank, numRanks, localRank, bootstrap, config);
  ASSERT_NO_THROW(transport.exchange());

  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  EXPECT_FALSE(transport.hasMultimemNvlTransport());
  EXPECT_THROW(
      static_cast<void>(transport.initializeMultimemNvlTransportIfEligible()),
      std::runtime_error);
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// getDeviceTransport() must refuse to vend a handle before exchange() has
// installed the multicast overlay. Tested per-rank (no bootstrap needed for
// the throw path, so this runs even on hosts without NVLS).
TEST_F(
    MultimemNvlTransportTestFixture,
    GetDeviceTransportThrowsBeforeExchange) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_get_device_before_exchange");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(/*dataBufferSize=*/4096));
  EXPECT_THROW((void)transport.getDeviceTransport(), std::runtime_error);
}

// Happy-path exchange() from the primary (global-bootstrap +
// nvlRankToCommRank) constructor. Locks down the device handle shape:
// non-null local/multimem base pointers, distinct pointers, dataBufferSize
// echoed through, and user + internal signal spans sized to the requested
// slot counts. Also verifies the multimem pointer is stable across
// exchange() calls (idempotency) and that the two getAllocated* accessors
// report the configured / SignalState-aligned sizes.
TEST_F(MultimemNvlTransportTestFixture, ExchangeSetsUpDeviceHandle) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_exchange_happy_path");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kDataBytes = 8192;
  constexpr uint32_t kUserSignals = 2;
  // Keep these distinct so the test fails if handle.maxBlocks is accidentally
  // wired from maxChannels.
  constexpr std::size_t kMaxChannels = 2;
  constexpr std::size_t kMaxBlocks = 1;
  const uint64_t signalsPerChannel = multimem_staging_signals_per_channel(
      static_cast<uint32_t>(numRanks), /*pipelineDepth=*/1);
  const uint64_t internalSignals = kMaxChannels * signalsPerChannel;

  auto config =
      makeConfig(kDataBytes, kUserSignals, /*pipelineDepth=*/1, kMaxChannels);
  config.maxBlocks = kMaxBlocks;

  MultimemNvlTransport transport(
      bootstrap, globalRank, identityRankMap(numRanks), config);

  transport.exchange();
  auto handle = transport.getDeviceTransport();

  EXPECT_NE(handle.localData, nullptr);
  EXPECT_NE(handle.multimemData, nullptr);
  EXPECT_EQ(handle.nvlRank, globalRank);
  EXPECT_EQ(handle.nvlRanks, numRanks);
  EXPECT_NE(handle.localData, handle.multimemData)
      << "unicast and multicast VAs should be distinct";
  EXPECT_EQ(handle.dataBufferSize, kDataBytes);
  EXPECT_EQ(handle.userLocalSignals.size(), kUserSignals);
  EXPECT_EQ(handle.userMultimemSignals.size(), kUserSignals);
  EXPECT_EQ(handle.internalLocalSignals.size(), internalSignals);
  EXPECT_EQ(handle.internalMultimemSignals.size(), internalSignals);
  EXPECT_EQ(handle.pipelineDepth, 1);
  EXPECT_EQ(handle.maxChannels, kMaxChannels);
  EXPECT_EQ(handle.maxBlocks, kMaxBlocks);
  EXPECT_EQ(handle.signalsPerChannel, signalsPerChannel);
  EXPECT_TRUE(handle.internalUnicastSignalsByRank.empty());

  EXPECT_EQ(transport.getAllocatedDataBufferSize(), kDataBytes);
  EXPECT_EQ(
      transport.getAllocatedSignalBufferSize(),
      getSignalBufferSize(static_cast<int>(kUserSignals + internalSignals)));

  // Idempotency: a second exchange() must be a no-op.
  auto* firstMultimemBase = handle.multimemData;
  transport.exchange();
  auto handle2 = transport.getDeviceTransport();
  EXPECT_EQ(handle2.multimemData, firstMultimemBase);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(MultimemNvlTransportTestFixture, ExchangeSupportsDataOnlyConfiguration) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_exchange_data_only");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kDataBytes = 4096;
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(
          kDataBytes,
          /*userSignalCount=*/0,
          /*pipelineDepth=*/0,
          /*maxChannels=*/0));

  transport.exchange();
  const auto handle = transport.getDeviceTransport();

  EXPECT_EQ(handle.dataBufferSize, kDataBytes);
  EXPECT_TRUE(handle.userLocalSignals.empty());
  EXPECT_TRUE(handle.userMultimemSignals.empty());
  EXPECT_TRUE(handle.internalLocalSignals.empty());
  EXPECT_TRUE(handle.internalMultimemSignals.empty());
  EXPECT_EQ(handle.pipelineDepth, 0);
  EXPECT_EQ(handle.maxChannels, 1);
  EXPECT_EQ(handle.maxBlocks, 0);
  EXPECT_EQ(handle.signalsPerChannel, 0);
  EXPECT_EQ(transport.getAllocatedSignalBufferSize(), 0);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(MultimemNvlTransportTestFixture, PeerViewKernelsCoverNvl72) {
  constexpr int kNvlRanks = 72;
  constexpr int kSourceRank = kNvlRanks - 1;
  constexpr uint64_t kValue = 0x123456789ABCDEF0ULL;
  const std::size_t signalCount =
      static_cast<std::size_t>(kNvlRanks) * kNvlRanks;

  SignalState* signals = nullptr;
  SignalState** pointerTable = nullptr;
  uint64_t* output = nullptr;
  CUDACHECK_TEST(cudaMalloc(&signals, signalCount * sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signals, 0, signalCount * sizeof(SignalState)));
  CUDACHECK_TEST(cudaMalloc(&pointerTable, kNvlRanks * sizeof(SignalState*)));
  CUDACHECK_TEST(cudaMalloc(&output, kNvlRanks * sizeof(uint64_t)));
  CUDACHECK_TEST(cudaMemset(output, 0, kNvlRanks * sizeof(uint64_t)));

  std::vector<SignalState*> hostPointerTable(kNvlRanks);
  for (int destination = 0; destination < kNvlRanks; ++destination) {
    hostPointerTable[static_cast<std::size_t>(destination)] =
        signals + static_cast<std::size_t>(destination) * kNvlRanks;
  }
  CUDACHECK_TEST(cudaMemcpy(
      pointerTable,
      hostPointerTable.data(),
      hostPointerTable.size() * sizeof(SignalState*),
      cudaMemcpyHostToDevice));

  MultimemNvlTransportDevice transport{
      .internalLocalSignals = DeviceSpan<SignalState>(
          signals + static_cast<std::size_t>(kSourceRank) * kNvlRanks,
          kNvlRanks),
      .nvlRank = kSourceRank,
      .nvlRanks = kNvlRanks,
      .internalUnicastSignalsByRank =
          DeviceSpan<SignalState*>(pointerTable, kNvlRanks),
  };
  test::launchSetAllPeerInternalSignals(transport, kValue);
  test::launchReadPeerInternalSignals(transport, output);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<uint64_t> values(kNvlRanks);
  CUDACHECK_TEST(cudaMemcpy(
      values.data(),
      output,
      values.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  std::vector<uint64_t> expected(kNvlRanks);
  expected.back() = kValue;
  EXPECT_EQ(values, expected);

  CUDACHECK_TEST(cudaFree(output));
  CUDACHECK_TEST(cudaFree(pointerTable));
  CUDACHECK_TEST(cudaFree(signals));
}

TEST_F(
    MultimemNvlTransportTestFixture,
    ExchangeRejectsMismatchedStagingGeometry) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_exchange_mismatched_geometry");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  const std::size_t pipelineDepth = globalRank == 0 ? 2 : 4;
  const std::size_t maxChannels = globalRank == 0 ? 4 : 2;
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(8192, 1, pipelineDepth, maxChannels));
  try {
    transport.exchange();
    FAIL() << "expected setup agreement to reject mismatched geometry";
  } catch (const std::runtime_error& ex) {
    const std::string message = ex.what();
    EXPECT_NE(
        message.find("ranks disagree on multicast setup"), std::string::npos)
        << message;
    EXPECT_NE(
        message.find("parameters=[2048, 2, 4, 4, 1, 0]"), std::string::npos)
        << message;
    EXPECT_NE(
        message.find("parameters=[4096, 4, 2, 2, 1, 0]"), std::string::npos)
        << message;
  }

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    ExchangeRejectsMismatchedUserSignalLayout) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_exchange_mismatched_user_signals");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  const uint32_t userSignalCount = globalRank == 0 ? 1 : 2;
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(8192, userSignalCount, 1, 1));
  try {
    transport.exchange();
    FAIL() << "expected setup agreement to reject mismatched signal layout";
  } catch (const std::runtime_error& ex) {
    const std::string message = ex.what();
    EXPECT_NE(
        message.find("ranks disagree on multicast setup"), std::string::npos)
        << message;
    EXPECT_NE(
        message.find("parameters=[8192, 1, 1, 1, 1, 0]"), std::string::npos)
        << message;
    EXPECT_NE(
        message.find("parameters=[8192, 1, 1, 1, 2, 0]"), std::string::npos)
        << message;
  }

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    ExchangeRejectsMismatchedUnicastPeerViewEnablement) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_exchange_mismatched_peer_views");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  const bool enableUnicastPeerViews = globalRank == 0;
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(4096, 0, 1, 1, enableUnicastPeerViews));
  try {
    transport.exchange();
    FAIL() << "expected setup agreement to reject mismatched peer views";
  } catch (const std::runtime_error& ex) {
    const std::string message = ex.what();
    EXPECT_NE(
        message.find("ranks disagree on multicast setup"), std::string::npos)
        << message;
    EXPECT_NE(
        message.find("parameters=[4096, 1, 1, 1, 0, 0]"), std::string::npos)
        << message;
    EXPECT_NE(
        message.find("parameters=[4096, 1, 1, 1, 0, 1]"), std::string::npos)
        << message;
  }

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// The user and internal signal spans must be disjoint contiguous regions,
// with the internal region starting immediately after the user region on
// both the local and the multimem sides. This is what lets the transport
// reserve internal slots for its own protocols without leaking into the
// user-visible SignalState indices.
TEST_F(MultimemNvlTransportTestFixture, UserAndInternalSignalSpansAreDisjoint) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_signal_span_layout");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr uint32_t kUserSignals = 4;
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(/*dataBufferSize=*/4096, kUserSignals, 1, 1));
  transport.exchange();
  auto handle = transport.getDeviceTransport();

  // Internal region starts at userSignalCount elements past the user base,
  // on both mirrors.
  EXPECT_EQ(
      handle.internalLocalSignals.data(),
      handle.userLocalSignals.data() + kUserSignals);
  EXPECT_EQ(
      handle.internalMultimemSignals.data(),
      handle.userMultimemSignals.data() + kUserSignals);

  // User and internal regions do not overlap on either mirror.
  EXPECT_LE(
      handle.userLocalSignals.data() + handle.userLocalSignals.size(),
      handle.internalLocalSignals.data());
  EXPECT_LE(
      handle.userMultimemSignals.data() + handle.userMultimemSignals.size(),
      handle.internalMultimemSignals.data());

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(MultimemNvlTransportTestFixture, StageLayoutUsesTransportGeometry) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_stage_layout_geometry");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kDataBytes = 12 * 1024;
  constexpr uint32_t kPipelineDepth = 2;
  constexpr uint32_t kMaxChannels = 4;
  constexpr uint32_t kActiveGroups = 3;
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(kDataBytes, 0, kPipelineDepth, kMaxChannels));
  transport.exchange();

  test::StageLayoutResult* deviceResults = nullptr;
  CUDACHECK_TEST(cudaMalloc(
      &deviceResults, kMaxChannels * sizeof(test::StageLayoutResult)));
  test::launchStageLayout(
      transport.getDeviceTransport(), deviceResults, kActiveGroups);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<test::StageLayoutResult> results(kActiveGroups);
  CUDACHECK_TEST(cudaMemcpy(
      results.data(),
      deviceResults,
      results.size() * sizeof(test::StageLayoutResult),
      cudaMemcpyDeviceToHost));
  const uint64_t signalsPerChannel = multimem_staging_signals_per_channel(
      static_cast<uint32_t>(numRanks), kPipelineDepth);
  for (uint32_t group = 0; group < kActiveGroups; ++group) {
    const uint64_t channelBase = group * signalsPerChannel;
    const uint64_t laneBase = channelBase + 3 * numRanks;
    EXPECT_EQ(results[group].channelBeginBytes, group * 4096);
    EXPECT_EQ(results[group].stagingBytes, 2048);
    EXPECT_EQ(results[group].signalBase, channelBase);
    EXPECT_EQ(results[group].signalsPerChannel, signalsPerChannel);
    EXPECT_EQ(results[group].readyFirst, channelBase);
    EXPECT_EQ(results[group].readyLast, channelBase + numRanks - 1);
    EXPECT_EQ(results[group].ackFirst, channelBase + numRanks);
    EXPECT_EQ(results[group].ackLast, channelBase + 2 * numRanks - 1);
    EXPECT_EQ(results[group].consumedFirst, channelBase + 2 * numRanks);
    EXPECT_EQ(results[group].consumedLast, channelBase + 3 * numRanks - 1);
    EXPECT_EQ(results[group].lane0ReadyCounter, laneBase);
    EXPECT_EQ(results[group].lane0ReadyEpoch, laneBase + 1);
    EXPECT_EQ(results[group].lane0AckCounter, laneBase + 2);
    EXPECT_EQ(results[group].lane0AckEpoch, laneBase + 3);
    EXPECT_EQ(results[group].lane1ReadyCounter, laneBase + 4);
    EXPECT_EQ(results[group].lane1ReadyEpoch, laneBase + 5);
    EXPECT_EQ(results[group].lane1AckCounter, laneBase + 6);
    EXPECT_EQ(results[group].lane1AckEpoch, laneBase + 7);
    EXPECT_EQ(results[group].pipelineDepth, kPipelineDepth);
  }

  test::launchStageLayout(
      transport.getDeviceTransport(), deviceResults, kMaxChannels);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  results.resize(kMaxChannels);
  CUDACHECK_TEST(cudaMemcpy(
      results.data(),
      deviceResults,
      results.size() * sizeof(test::StageLayoutResult),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaFree(deviceResults));
  for (uint32_t group = 0; group < kMaxChannels; ++group) {
    EXPECT_EQ(results[group].channelBeginBytes, group * 3072);
    EXPECT_EQ(results[group].stagingBytes, 1536);
    EXPECT_EQ(results[group].signalBase, group * signalsPerChannel);
  }

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// The compat constructor must also drive a real exchange to completion.
// Uses NVL-local (rank, size) coordinates against the same underlying NVL
// team.
TEST_F(MultimemNvlTransportTestFixture, CompatCtorExchangeSucceeds) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_compat_ctor_exchange");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  MultimemNvlTransport transport(
      /*nvlRank=*/globalRank,
      /*nvlRanks=*/numRanks,
      bootstrap,
      makeConfig(/*dataBufferSize=*/4096));
  ASSERT_NO_THROW(transport.exchange());
  auto handle = transport.getDeviceTransport();
  EXPECT_NE(handle.localData, nullptr);
  EXPECT_NE(handle.multimemData, nullptr);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// Poison-on-failure: inject a bootstrap failure symmetrically on every rank
// via a delegating StrictMock, verify exchange() throws, and verify a
// second exchange() on the same object also throws (the poisoned-object
// contract). The recovery path -- constructing a fresh MultimemNvlTransport
// and exchange()-ing successfully -- is proven in ExchangeSetsUpDeviceHandle
// (fresh transport per test) and by the underlying MultimemHandler failure
// tests, so we intentionally do not repeat the fresh-object recovery here.
TEST_F(MultimemNvlTransportTestFixture, ExchangePoisonsAfterFailure) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto real = makeBootstrap("mmnvl_exchange_poison");
  if (!allRanksMultimemEligible(real, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  using ::testing::_;
  using ::testing::AnyNumber;
  auto mock = makeDelegatingMock(real);
  // Inject failure on the very first NVL-domain allGather (agreeOnHandleType
  // inside MultimemHandler::exchange). Every rank throws symmetrically so
  // there's no need to shorten the store timeout.
  EXPECT_CALL(*mock, allGatherNvlDomain(_, _, _, _, _))
      .WillOnce([](void*, int, int, int, const std::vector<int>&) {
        return folly::makeSemiFuture(-1);
      });
  EXPECT_CALL(*mock, barrierNvlDomain(_, _, _)).Times(AnyNumber());

  MultimemNvlTransport transport(
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      globalRank,
      identityRankMap(numRanks),
      makeConfig(/*dataBufferSize=*/4096));

  EXPECT_THROW(transport.exchange(), std::runtime_error);

  // Second call must throw the poisoned-object error without touching the
  // bootstrap. The StrictMock has no further EXPECT_CALL, so any recovered
  // bootstrap call would fail the test.
  try {
    transport.exchange();
    FAIL() << "expected poisoned transport to throw on second exchange()";
  } catch (const std::runtime_error& ex) {
    EXPECT_NE(
        std::string(ex.what()).find("previous exchange() failed"),
        std::string::npos)
        << ex.what();
  }

  // getDeviceTransport must also refuse: the transport was never marked
  // exchanged.
  EXPECT_THROW((void)transport.getDeviceTransport(), std::runtime_error);

  ASSERT_EQ(real->barrier(globalRank, numRanks).get(), 0);
}

// -----------------------------------------------------------------------------
// Device signal API tests
// -----------------------------------------------------------------------------
// These tests launch the kernels declared in MultimemNvlTransportTest.cuh
// against a fully-exchanged transport. Each verifies one behavior of the
// device signal API end-to-end: multimem PTX store propagation, multimem
// atomic-add accumulation, user/internal span isolation, wait_until, and
// read_signal / read_internal_signal.

namespace {

// Device buffer for a single uint64_t output slot. Reset to a distinctive
// sentinel between tests so a missing kernel write is obvious.
struct DeviceUint64Slot {
  DeviceUint64Slot() {
    CUDACHECK_TEST(cudaMalloc(&ptr_, sizeof(uint64_t)));
    reset(kSentinel);
  }
  ~DeviceUint64Slot() {
    if (ptr_) {
      (void)cudaFree(ptr_);
    }
  }
  DeviceUint64Slot(const DeviceUint64Slot&) = delete;
  DeviceUint64Slot& operator=(const DeviceUint64Slot&) = delete;
  DeviceUint64Slot(DeviceUint64Slot&&) = delete;
  DeviceUint64Slot& operator=(DeviceUint64Slot&&) = delete;

  void reset(uint64_t v) {
    CUDACHECK_TEST(
        cudaMemcpy(ptr_, &v, sizeof(uint64_t), cudaMemcpyHostToDevice));
  }
  uint64_t read() const {
    uint64_t v = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&v, ptr_, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return v;
  }
  uint64_t* device_ptr() {
    return ptr_;
  }

  static constexpr uint64_t kSentinel = 0xDEADBEEFCAFEBABEULL;

 private:
  uint64_t* ptr_{nullptr};
};

template <typename Launch>
std::vector<uint64_t> runSignalProtocol(std::size_t valueCount, Launch launch) {
  uint64_t* deviceValues = nullptr;
  CUDACHECK_TEST(cudaMalloc(&deviceValues, valueCount * sizeof(uint64_t)));
  CUDACHECK_TEST(cudaMemset(deviceValues, 0, valueCount * sizeof(uint64_t)));
  launch(deviceValues);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  std::vector<uint64_t> values(valueCount);
  CUDACHECK_TEST(cudaMemcpy(
      values.data(),
      deviceValues,
      values.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaFree(deviceValues));
  return values;
}

// Convenience: construct + exchange a transport with the given signal
// counts. Skips the caller test if the NVL team isn't multimem-eligible.
std::unique_ptr<MultimemNvlTransport> makeExchangedTransport(
    const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
    int globalRank,
    int numRanks,
    int localRank,
    uint32_t userSignalCount,
    bool needsInternalSignals) {
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    return nullptr;
  }
  auto config = makeConfig(
      /*dataBufferSize=*/4096,
      userSignalCount,
      needsInternalSignals ? 1 : 0,
      needsInternalSignals ? 1 : 0);
  auto transport = std::make_unique<MultimemNvlTransport>(
      bootstrap, globalRank, identityRankMap(numRanks), config);
  transport->exchange();
  return transport;
}

void verifyUnicastPeerViews(
    const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
    int globalRank,
    int numRanks,
    const std::vector<int>& rankMap,
    MultimemNvlTransport& transport) {
  const auto handle = transport.getDeviceTransport();
  ASSERT_EQ(handle.internalUnicastSignalsByRank.size(), numRanks);

  const auto nvlRankIt = std::find(rankMap.begin(), rankMap.end(), globalRank);
  ASSERT_NE(nvlRankIt, rankMap.end());
  const int nvlRank = static_cast<int>(nvlRankIt - rankMap.begin());
  EXPECT_EQ(handle.nvlRank, nvlRank);

  test::launchSetAllPeerInternalSignals(
      handle, static_cast<uint64_t>(nvlRank + 1));
  CUDACHECK_TEST(cudaDeviceSynchronize());
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  uint64_t* deviceValues = nullptr;
  CUDACHECK_TEST(cudaMalloc(
      &deviceValues, static_cast<std::size_t>(numRanks) * sizeof(uint64_t)));
  test::launchReadPeerInternalSignals(handle, deviceValues);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  std::vector<uint64_t> values(static_cast<std::size_t>(numRanks));
  CUDACHECK_TEST(cudaMemcpy(
      values.data(),
      deviceValues,
      values.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaFree(deviceValues));

  std::vector<uint64_t> expected(static_cast<std::size_t>(numRanks));
  for (int rank = 0; rank < numRanks; ++rank) {
    expected[static_cast<std::size_t>(rank)] = static_cast<uint64_t>(rank + 1);
  }
  EXPECT_EQ(values, expected);
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

} // namespace

TEST_F(
    MultimemNvlTransportTestFixture,
    DeviceUnicastPeerViewsUseIdentityNvlRankOrder) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_unicast_peer_views_identity");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  const auto rankMap = identityRankMap(numRanks);
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      rankMap,
      makeConfig(
          /*dataBufferSize=*/4096,
          /*userSignalCount=*/0,
          1,
          1,
          /*enableUnicastPeerViews=*/true));
  transport.exchange();
  verifyUnicastPeerViews(bootstrap, globalRank, numRanks, rankMap, transport);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    DeviceUnicastPeerViewsUseConfiguredNvlRankOrder) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_unicast_peer_views_rotated");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  const auto rankMap = rotatedRankMap(numRanks);
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      rankMap,
      makeConfig(
          /*dataBufferSize=*/4096,
          /*userSignalCount=*/0,
          1,
          1,
          /*enableUnicastPeerViews=*/true));
  transport.exchange();
  verifyUnicastPeerViews(bootstrap, globalRank, numRanks, rankMap, transport);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalAggregateSupportsBothAccessPathsAndScopes) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  constexpr uint32_t kPipelineDepth = 4;
  uint32_t caseIndex = 0;
  for (const auto access :
       {NvlSignalAccess::Unicast, NvlSignalAccess::Multimem}) {
    for (const bool fanIn : {false, true}) {
      auto bootstrap = makeBootstrap(
          "mmnvl_uniform_aggregate_matrix_" + std::to_string(caseIndex++));
      if (!allRanksMultimemEligible(
              bootstrap, globalRank, numRanks, localRank)) {
        GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
      }
      MultimemNvlTransport transport(
          bootstrap,
          globalRank,
          identityRankMap(numRanks),
          makeConfig(
              4096,
              0,
              kPipelineDepth,
              1,
              /*enableUnicastPeerViews=*/true));
      transport.exchange();
      const auto values =
          runSignalProtocol(2 * kPipelineDepth, [&](uint64_t* output) {
            test::launchAggregateSignalProtocol(
                transport.getDeviceTransport(),
                access,
                NvlSignalPhase::Ready,
                fanIn,
                /*roundValue=*/1,
                output);
          });
      if (!fanIn || globalRank == 0) {
        const std::vector<uint64_t> expected(
            values.size(),
            static_cast<uint64_t>(fanIn ? numRanks - 1 : numRanks));
        EXPECT_EQ(values, expected);
      }
      ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
    }
  }
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalPerPeerCoversPhasePolicyAccessAndScope) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_uniform_per_peer_matrix");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(
          4096,
          0,
          /*pipelineDepth=*/1,
          /*maxChannels=*/1,
          /*enableUnicastPeerViews=*/true));
  transport.exchange();

  struct PerPeerProtocolCase {
    NvlSignalAccess access;
    NvlSignalPhase phase;
    NvlPerPeerWaitPolicy waitPolicy;
  };
  // Access and phase affect publication and signal selection. Wait policy
  // affects only observation. Cover those dimensions orthogonally instead of
  // instantiating their full Cartesian product.
  constexpr PerPeerProtocolCase kProtocols[] = {
      {
          NvlSignalAccess::Unicast,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::WaitAll,
      },
      {
          NvlSignalAccess::Unicast,
          NvlSignalPhase::Ack,
          NvlPerPeerWaitPolicy::WaitAll,
      },
      {
          NvlSignalAccess::Unicast,
          NvlSignalPhase::Consumed,
          NvlPerPeerWaitPolicy::WaitAll,
      },
      {
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::WaitAll,
      },
      {
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Ack,
          NvlPerPeerWaitPolicy::WaitAll,
      },
      {
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Consumed,
          NvlPerPeerWaitPolicy::WaitAll,
      },
      {
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::SerialMin,
      },
      {
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::TreeMin,
      },
      {
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::ButterflyMin,
      },
  };

  uint64_t roundValue = 1;
  for (const auto& protocol : kProtocols) {
    for (const bool fanIn : {false, true}) {
      const auto values = runSignalProtocol(
          static_cast<std::size_t>(numRanks), [&](uint64_t* output) {
            if (protocol.waitPolicy == NvlPerPeerWaitPolicy::WaitAll) {
              test::launchPerPeerWaitAllSignalProtocol(
                  transport.getDeviceTransport(),
                  protocol.access,
                  protocol.phase,
                  fanIn,
                  roundValue,
                  output);
            } else {
              test::launchMultimemReadyPerPeerSignalProtocol(
                  transport.getDeviceTransport(),
                  protocol.waitPolicy,
                  fanIn,
                  roundValue,
                  output);
            }
          });
      if (!fanIn || globalRank == 0) {
        std::vector<uint64_t> expected(static_cast<std::size_t>(numRanks), 0);
        if (fanIn) {
          expected[0] = roundValue - 1;
        }
        for (int source = fanIn ? 1 : 0; source < numRanks; ++source) {
          expected[static_cast<std::size_t>(source)] = roundValue;
        }
        EXPECT_EQ(values, expected);
      }
      ++roundValue;
      ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
    }
  }
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalRoundTripUsesOneMultimemAggregateAckPublisher) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  constexpr uint32_t kPipelineDepth = 4;
  auto bootstrap = makeBootstrap("mmnvl_uniform_rtt");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(4096, 0, kPipelineDepth, 1));
  transport.exchange();
  const auto handle = transport.getDeviceTransport();
  const auto values =
      runSignalProtocol(4 * kPipelineDepth, [&](uint64_t* output) {
        test::launchAggregateSignalProtocol(
            handle,
            NvlSignalAccess::Multimem,
            NvlSignalPhase::Ready,
            /*fanIn=*/true,
            /*roundValue=*/1,
            output);
        test::launchAggregateAckSignalProtocol(
            handle, /*roundValue=*/1, output + 2 * kPipelineDepth);
      });
  const std::vector<uint64_t> expectedAck(2 * kPipelineDepth, 1);
  EXPECT_EQ(
      std::vector<uint64_t>(values.begin() + 2 * kPipelineDepth, values.end()),
      expectedAck);
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalAggregateSupportsEveryPipelineDepth) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  for (const uint32_t pipelineDepth : {1, 2, 4, 8, 16, 32}) {
    auto bootstrap =
        makeBootstrap("mmnvl_uniform_depth_" + std::to_string(pipelineDepth));
    if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
      GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
    }
    MultimemNvlTransport transport(
        bootstrap,
        globalRank,
        identityRankMap(numRanks),
        makeConfig(4096, 0, pipelineDepth, 1));
    transport.exchange();
    const auto values =
        runSignalProtocol(2 * pipelineDepth, [&](uint64_t* output) {
          test::launchAggregateSignalProtocol(
              transport.getDeviceTransport(),
              NvlSignalAccess::Multimem,
              NvlSignalPhase::Ready,
              /*fanIn=*/false,
              /*roundValue=*/1,
              output);
        });
    const std::vector<uint64_t> expected(
        values.size(), static_cast<uint64_t>(numRanks));
    EXPECT_EQ(values, expected);
    ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
  }
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalAggregatePublishesRelaxedPayloadFromAnotherLane) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  for (const uint32_t pipelineDepth : {1u, 4u}) {
    auto bootstrap = makeBootstrap(
        "mmnvl_uniform_relaxed_payload_depth_" + std::to_string(pipelineDepth));
    if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
      GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
    }
    MultimemNvlTransport transport(
        bootstrap,
        globalRank,
        identityRankMap(numRanks),
        makeConfig(
            4096,
            0,
            pipelineDepth,
            /*maxChannels=*/1,
            /*enableUnicastPeerViews=*/true));
    transport.exchange();
    ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

    const auto values = runSignalProtocol(1, [&](uint64_t* output) {
      test::launchAggregateMultimemRelaxedPayload(
          transport.getDeviceTransport(), output);
    });
    if (globalRank == 0) {
      EXPECT_EQ(values[0], 11) << "pipelineDepth=" << pipelineDepth;
    }
    ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
  }
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalPerPeerPublishesRelaxedPayloadFromAnotherWarp) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_uniform_per_peer_relaxed_payload");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(
          4096,
          0,
          /*pipelineDepth=*/1,
          /*maxChannels=*/1,
          /*enableUnicastPeerViews=*/true));
  transport.exchange();
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  const auto values = runSignalProtocol(1, [&](uint64_t* output) {
    test::launchPerPeerMultimemRelaxedPayload(
        transport.getDeviceTransport(), output);
  });
  if (globalRank == 0) {
    EXPECT_EQ(values[0], 22);
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    BlockAggregateBarrierOrdersEveryWarpAcrossRepeatedEpochs) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  constexpr uint32_t kChannels = 2;
  constexpr uint32_t kPipelineDepth = 4;
  constexpr uint32_t kEpochs = 3;
  constexpr uint32_t kThreads = 128;
  constexpr std::size_t kElementCount = kChannels * kEpochs * kThreads;
  auto bootstrap = makeBootstrap("mmnvl_block_aggregate_barrier");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(
          kElementCount * sizeof(int32_t),
          /*userSignalCount=*/0,
          kPipelineDepth,
          kChannels));
  transport.exchange();

  int32_t* deviceReducedValues = nullptr;
  uint64_t* deviceSignalValues = nullptr;
  constexpr std::size_t kSignalValueCount = 2 * kChannels * kPipelineDepth;
  CUDACHECK_TEST(
      cudaMalloc(&deviceReducedValues, kElementCount * sizeof(int32_t)));
  CUDACHECK_TEST(
      cudaMalloc(&deviceSignalValues, kSignalValueCount * sizeof(uint64_t)));
  test::launchBlockAggregateBarrier(
      transport.getDeviceTransport(),
      kChannels,
      kEpochs,
      deviceReducedValues,
      deviceSignalValues);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int32_t> reducedValues(kElementCount);
  std::vector<uint64_t> signalValues(kSignalValueCount);
  CUDACHECK_TEST(cudaMemcpy(
      reducedValues.data(),
      deviceReducedValues,
      kElementCount * sizeof(int32_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      signalValues.data(),
      deviceSignalValues,
      kSignalValueCount * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaFree(deviceReducedValues));
  CUDACHECK_TEST(cudaFree(deviceSignalValues));

  const int32_t rankSum = numRanks * (numRanks + 1) / 2;
  for (uint32_t epoch = 0; epoch < kEpochs; ++epoch) {
    for (uint32_t channel = 0; channel < kChannels; ++channel) {
      const int32_t expected =
          rankSum + numRanks * static_cast<int32_t>(10 * epoch + 100 * channel);
      const std::size_t offset =
          (static_cast<std::size_t>(epoch) * kChannels + channel) * kThreads;
      EXPECT_EQ(
          std::vector<int32_t>(
              reducedValues.begin() + offset,
              reducedValues.begin() + offset + kThreads),
          std::vector<int32_t>(kThreads, expected));
    }
  }
  const uint64_t expectedSignalValue =
      static_cast<uint64_t>(kEpochs * numRanks);
  for (uint32_t channel = 0; channel < kChannels; ++channel) {
    const std::size_t outputBase =
        static_cast<std::size_t>(channel) * 2 * kPipelineDepth;
    EXPECT_EQ(signalValues[outputBase], expectedSignalValue);
    EXPECT_EQ(signalValues[outputBase + kPipelineDepth], expectedSignalValue);
    for (uint32_t lane = 1; lane < kPipelineDepth; ++lane) {
      EXPECT_EQ(signalValues[outputBase + lane], 0);
      EXPECT_EQ(signalValues[outputBase + kPipelineDepth + lane], 0);
    }
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalPerPeerRoundTripUsesAggregateAck) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_uniform_per_peer_rtt");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(4096, 0, /*pipelineDepth=*/1, /*maxChannels=*/1));
  transport.exchange();

  constexpr uint64_t kRoundValue = 1;
  const auto handle = transport.getDeviceTransport();
  const auto values = runSignalProtocol(
      static_cast<std::size_t>(numRanks + 2), [&](uint64_t* output) {
        test::launchPerPeerWaitAllSignalProtocol(
            handle,
            NvlSignalAccess::Multimem,
            NvlSignalPhase::Ready,
            /*fanIn=*/true,
            kRoundValue,
            output);
        test::launchAggregateAckSignalProtocol(
            handle, kRoundValue, output + numRanks);
      });
  EXPECT_EQ(values[static_cast<std::size_t>(numRanks)], kRoundValue);
  EXPECT_EQ(values[static_cast<std::size_t>(numRanks + 1)], kRoundValue);
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalKeepsChannelsAndLanesIndependent) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  constexpr uint32_t kChannels = 4;
  constexpr uint32_t kPipelineDepth = 4;
  auto bootstrap = makeBootstrap("mmnvl_uniform_channel_isolation");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(16384, 0, kPipelineDepth, kChannels));
  transport.exchange();
  const auto values =
      runSignalProtocol(2 * kChannels * kPipelineDepth, [&](uint64_t* output) {
        test::launchMultiChannelAggregateSignal(
            transport.getDeviceTransport(), kChannels, output);
      });
  const std::vector<uint64_t> expected(
      values.size(), static_cast<uint64_t>(numRanks));
  EXPECT_EQ(values, expected);
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalAggregateMultimemAccountsForWaiterTransitions) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  constexpr uint32_t kPipelineDepth = 4;
  auto bootstrap = makeBootstrap("mmnvl_uniform_waiter_transition");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(4096, 0, kPipelineDepth, /*maxChannels=*/1));
  transport.exchange();
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  const auto values =
      runSignalProtocol(2 * kPipelineDepth, [&](uint64_t* output) {
        test::launchAggregateMultimemWaiterTransition(
            transport.getDeviceTransport(), output);
      });
  const uint64_t expected = static_cast<uint64_t>(2 * numRanks - 1);
  EXPECT_EQ(values, std::vector<uint64_t>(values.size(), expected));
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalPublishAndWaitComposeAcrossRanks) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_uniform_separate_publish_wait");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(
          4096,
          0,
          /*pipelineDepth=*/1,
          /*maxChannels=*/1,
          /*enableUnicastPeerViews=*/true));
  transport.exchange();
  constexpr uint64_t kRound = 17;
  const auto values = runSignalProtocol(
      static_cast<std::size_t>(numRanks), [&](uint64_t* output) {
        test::launchSeparatePublishAndWait(
            transport.getDeviceTransport(), kRound, output);
      });
  if (globalRank == 0) {
    std::vector<uint64_t> expected(static_cast<std::size_t>(numRanks), kRound);
    expected[0] = 0;
    EXPECT_EQ(values, expected);
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalPerPeerRoundValuesSkipReservedZero) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_uniform_per_peer_wrap");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(
          4096,
          0,
          /*pipelineDepth=*/1,
          /*maxChannels=*/1,
          /*enableUnicastPeerViews=*/true));
  transport.exchange();
  test::launchSetAllPeerInternalSignals(
      transport.getDeviceTransport(), ~uint64_t{0} - 1);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
  for (const uint64_t roundValue : {~uint64_t{0}, uint64_t{1}}) {
    const auto values = runSignalProtocol(
        static_cast<std::size_t>(numRanks), [&](uint64_t* output) {
          test::launchPerPeerWaitAllSignalProtocol(
              transport.getDeviceTransport(),
              NvlSignalAccess::Multimem,
              NvlSignalPhase::Ready,
              /*fanIn=*/false,
              roundValue,
              output);
        });
    EXPECT_EQ(
        values,
        std::vector<uint64_t>(static_cast<std::size_t>(numRanks), roundValue));
    ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
  }
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalAggregateCountersAndEpochsWrapToZero) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  constexpr uint32_t kPipelineDepth = 4;
  auto bootstrap = makeBootstrap("mmnvl_uniform_aggregate_wrap");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(4096, 0, kPipelineDepth, /*maxChannels=*/1));
  transport.exchange();
  const uint64_t initial = ~uint64_t{0} - static_cast<uint64_t>(numRanks - 1);
  test::launchInitializeAggregateSignals(
      transport.getDeviceTransport(), initial, initial);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  const auto values =
      runSignalProtocol(2 * kPipelineDepth, [&](uint64_t* output) {
        test::launchAggregateSignalProtocol(
            transport.getDeviceTransport(),
            NvlSignalAccess::Multimem,
            NvlSignalPhase::Ready,
            /*fanIn=*/false,
            /*roundValue=*/1,
            output);
      });
  EXPECT_EQ(values, std::vector<uint64_t>(values.size(), 0));
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalWaitAcceptsAlreadyAdvancedPeerRounds) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_uniform_per_peer_advanced");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(
          4096,
          0,
          /*pipelineDepth=*/1,
          /*maxChannels=*/1,
          /*enableUnicastPeerViews=*/true));
  transport.exchange();
  constexpr uint64_t kExpectedRound = 41;
  constexpr uint64_t kObservedRound = kExpectedRound + 1;
  test::launchSetAllPeerInternalSignals(
      transport.getDeviceTransport(), kObservedRound);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
  const auto values = runSignalProtocol(
      static_cast<std::size_t>(numRanks), [&](uint64_t* output) {
        test::launchPerPeerWaitOnly(
            transport.getDeviceTransport(), kExpectedRound, output);
      });
  EXPECT_EQ(
      values,
      std::vector<uint64_t>(
          static_cast<std::size_t>(numRanks), kObservedRound));
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    MultimemNvlTransportTestFixture,
    UniformSignalWaitAcceptsAlreadyAdvancedAggregateCounters) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  constexpr uint32_t kPipelineDepth = 4;
  auto bootstrap = makeBootstrap("mmnvl_uniform_aggregate_advanced");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(4096, 0, kPipelineDepth, /*maxChannels=*/1));
  transport.exchange();
  const uint64_t arrivals = static_cast<uint64_t>(numRanks);
  test::launchInitializeAggregateSignals(
      transport.getDeviceTransport(), 2 * arrivals, /*epochValue=*/0);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
  const auto values =
      runSignalProtocol(2 * kPipelineDepth, [&](uint64_t* output) {
        test::launchAggregateSignalProtocol(
            transport.getDeviceTransport(),
            NvlSignalAccess::Multimem,
            NvlSignalPhase::Ready,
            /*fanIn=*/false,
            /*roundValue=*/1,
            output);
      });
  for (uint32_t lane = 0; lane < kPipelineDepth; ++lane) {
    EXPECT_GE(values[lane], 2 * arrivals);
    EXPECT_LE(values[lane], 3 * arrivals);
    EXPECT_EQ(values[kPipelineDepth + lane], arrivals);
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// signal(SET) from rank 0 must broadcast a value to every rank's local
// signal state; every rank observes it through wait_signal_until +
// read_signal.
TEST_F(MultimemNvlTransportTestFixture, DeviceUserSignalSetBroadcasts) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_user_signal_set");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*needsInternalSignals=*/false);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr uint64_t kSignalValue = 0xA5A50000ULL + 42;
  auto handle = transport->getDeviceTransport();

  if (globalRank == 0) {
    test::launchSetUserSignal(handle, /*signalId=*/0, kSignalValue);
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  DeviceUint64Slot out;
  test::launchWaitAndReadUserSignal(
      handle, /*signalId=*/0, CmpOp::CMP_EQ, kSignalValue, out.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  EXPECT_EQ(out.read(), kSignalValue);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// signal(ADD) from every rank must accumulate atomically through the
// multimem VA. Every rank waits until the sum arrives, then reads it.
TEST_F(MultimemNvlTransportTestFixture, DeviceUserSignalAddAccumulates) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_user_signal_add");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*needsInternalSignals=*/false);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  const uint64_t expected = static_cast<uint64_t>(numRanks);
  auto handle = transport->getDeviceTransport();

  test::launchAddUserSignal(handle, /*signalId=*/0, /*value=*/1);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  DeviceUint64Slot out;
  test::launchWaitAndReadUserSignal(
      handle, /*signalId=*/0, CmpOp::CMP_GE, expected, out.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  EXPECT_EQ(out.read(), expected);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// signal_internal(SET) from rank 0 must reach every rank's internal span
// via wait_internal_signal_until + read_internal_signal, exercising the
// internal-signal path independently of the user path.
TEST_F(MultimemNvlTransportTestFixture, DeviceInternalSignalSetBroadcasts) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_internal_signal_set");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*needsInternalSignals=*/true);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr uint64_t kValue = 0xC0DECAFE00000010ULL;
  auto handle = transport->getDeviceTransport();

  if (globalRank == 0) {
    test::launchSetInternalSignal(handle, /*signalId=*/0, kValue);
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  DeviceUint64Slot out;
  test::launchWaitAndReadInternalSignal(
      handle, /*signalId=*/0, CmpOp::CMP_EQ, kValue, out.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  EXPECT_EQ(out.read(), kValue);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// signal_internal(ADD) accumulates on the internal span, disjoint from
// whatever the user span may be doing.
TEST_F(MultimemNvlTransportTestFixture, DeviceInternalSignalAddAccumulates) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_internal_signal_add");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*needsInternalSignals=*/true);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  const uint64_t expected = static_cast<uint64_t>(numRanks);
  auto handle = transport->getDeviceTransport();

  test::launchAddInternalSignal(handle, /*signalId=*/0, /*value=*/1);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  DeviceUint64Slot out;
  test::launchWaitAndReadInternalSignal(
      handle, /*signalId=*/0, CmpOp::CMP_GE, expected, out.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  EXPECT_EQ(out.read(), expected);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// User vs. internal signal spans are isolated in device memory: writing
// only the user slot must NOT be observable through the internal read
// path, and vice versa. Uses the same signalId (0) in both spans on
// purpose, so a bug that merged the two spans would surface as one write
// leaking into the other reader.
TEST_F(MultimemNvlTransportTestFixture, DeviceUserAndInternalSignalsIsolated) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_user_internal_isolation");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*needsInternalSignals=*/true);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  // Distinct values so any aliasing between spans is obvious.
  constexpr uint64_t kUserValue = 0x11111111ULL;
  constexpr uint64_t kInternalValue = 0x22222222ULL;
  auto handle = transport->getDeviceTransport();

  if (globalRank == 0) {
    test::launchSetUserSignal(handle, /*signalId=*/0, kUserValue);
    test::launchSetInternalSignal(handle, /*signalId=*/0, kInternalValue);
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);

  // Each rank waits on both spans (broadcast from rank 0). Waiting on both
  // before reading ensures a stale sentinel doesn't slip through.
  DeviceUint64Slot userOut;
  test::launchWaitAndReadUserSignal(
      handle, /*signalId=*/0, CmpOp::CMP_EQ, kUserValue, userOut.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());

  DeviceUint64Slot internalOut;
  test::launchWaitAndReadInternalSignal(
      handle,
      /*signalId=*/0,
      CmpOp::CMP_EQ,
      kInternalValue,
      internalOut.device_ptr());
  CUDACHECK_TEST(cudaDeviceSynchronize());

  EXPECT_EQ(userOut.read(), kUserValue);
  EXPECT_EQ(internalOut.read(), kInternalValue);

  // Cross-check via the no-wait reader that reads both spans in one shot.
  // Uses a 2-slot device buffer so the kernel writes out[0]=user and
  // out[1]=internal in a single launch.
  uint64_t* pairOut = nullptr;
  CUDACHECK_TEST(cudaMalloc(&pairOut, 2 * sizeof(uint64_t)));
  test::launchReadUserAndInternal(
      handle, /*userId=*/0, /*internalId=*/0, pairOut);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  uint64_t hostPair[2] = {0, 0};
  CUDACHECK_TEST(
      cudaMemcpy(hostPair, pairOut, sizeof(hostPair), cudaMemcpyDeviceToHost));
  EXPECT_EQ(hostPair[0], kUserValue);
  EXPECT_EQ(hostPair[1], kInternalValue);
  CUDACHECK_TEST(cudaFree(pairOut));

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(MultimemNvlTransportTestFixture, DeviceLoadReduceCoversPublicTypes) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_device_load_reduce_types");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/1,
      /*needsInternalSignals=*/true);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kElems = 9;
  const float rankValue = static_cast<float>(globalRank + 1);
  const float expected = static_cast<float>(numRanks * (numRanks + 1) / 2);
  auto handle = transport->getDeviceTransport();

  auto run = [&](test::MultimemReductionTestType type,
                 bool accF32,
                 std::size_t elementSize,
                 std::size_t sourceOffsetElems,
                 auto verify) {
    void* output = nullptr;
    CUDACHECK_TEST(cudaMalloc(&output, kElems * elementSize));
    test::launchFillReductionInput(
        handle, type, rankValue, kElems, sourceOffsetElems);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
    test::launchLoadReduce(
        handle, type, accF32, output, kElems, sourceOffsetElems);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    verify(output, expected);
    CUDACHECK_TEST(cudaFree(output));
    ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
  };

  run(test::MultimemReductionTestType::Float,
      /*accF32=*/true,
      sizeof(float),
      /*sourceOffsetElems=*/0,
      [&](void* output, float expectedValue) {
        std::vector<float> values(kElems);
        CUDACHECK_TEST(cudaMemcpy(
            values.data(),
            output,
            values.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        for (float value : values) {
          EXPECT_EQ(value, expectedValue);
        }
      });
  run(test::MultimemReductionTestType::Float,
      /*accF32=*/true,
      sizeof(float),
      /*sourceOffsetElems=*/1,
      [&](void* output, float expectedValue) {
        std::vector<float> values(kElems);
        CUDACHECK_TEST(cudaMemcpy(
            values.data(),
            output,
            values.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        for (float value : values) {
          EXPECT_EQ(value, expectedValue);
        }
      });
  run(test::MultimemReductionTestType::Int32,
      /*accF32=*/true,
      sizeof(int32_t),
      /*sourceOffsetElems=*/0,
      [&](void* output, float expectedValue) {
        std::vector<int32_t> values(kElems);
        CUDACHECK_TEST(cudaMemcpy(
            values.data(),
            output,
            values.size() * sizeof(int32_t),
            cudaMemcpyDeviceToHost));
        for (int32_t value : values) {
          EXPECT_EQ(value, static_cast<int32_t>(expectedValue));
        }
      });

  for (const auto type :
       {test::MultimemReductionTestType::Float16,
        test::MultimemReductionTestType::Bfloat16}) {
    for (const bool accF32 : {false, true}) {
      for (const std::size_t sourceOffsetElems : {0, 1}) {
        run(type,
            accF32,
            sizeof(uint16_t),
            sourceOffsetElems,
            [&, type](void* output, float expectedValue) {
              std::vector<uint16_t> values(kElems);
              CUDACHECK_TEST(cudaMemcpy(
                  values.data(),
                  output,
                  values.size() * sizeof(uint16_t),
                  cudaMemcpyDeviceToHost));
              for (uint16_t bits : values) {
                float value = 0;
                if (type == test::MultimemReductionTestType::Float16) {
                  __half raw{};
                  std::memcpy(&raw, &bits, sizeof(raw));
                  value = __half2float(raw);
                } else {
                  __nv_bfloat16 raw{};
                  std::memcpy(&raw, &bits, sizeof(raw));
                  value = __bfloat162float(raw);
                  expectedValue =
                      __bfloat162float(__float2bfloat16(expectedValue));
                }
                EXPECT_EQ(value, expectedValue);
              }
            });
      }
    }
  }
}

TEST_F(
    MultimemNvlTransportTestFixture,
    PhasedReduceBlockPreservesOwnedFp16AndBf16Lanes) {
  if (numRanks < 3) {
    GTEST_SKIP() << "MultimemNvlTransport requires 3+ ranks";
  }
  auto bootstrap = makeBootstrap("mmnvl_phased_reduce_block");
  auto transport = makeExchangedTransport(
      bootstrap,
      globalRank,
      numRanks,
      localRank,
      /*userSignalCount=*/0,
      /*needsInternalSignals=*/true);
  if (!transport) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kElements = 8;
  const float expected = static_cast<float>(numRanks * (numRanks + 1) / 2);
  const std::size_t firstLane = kElements *
      static_cast<std::size_t>(globalRank) / static_cast<std::size_t>(numRanks);
  const std::size_t endLane = kElements *
      static_cast<std::size_t>(globalRank + 1) /
      static_cast<std::size_t>(numRanks);
  for (const auto type :
       {test::MultimemReductionTestType::Float16,
        test::MultimemReductionTestType::Bfloat16}) {
    for (const bool accF32 : {false, true}) {
      void* output = nullptr;
      CUDACHECK_TEST(cudaMalloc(&output, 16));
      CUDACHECK_TEST(cudaMemset(output, 0, 16));
      test::launchPhasedReduceBlock(
          transport->getDeviceTransport(), type, accF32, output);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      std::vector<uint16_t> values(kElements);
      CUDACHECK_TEST(
          cudaMemcpy(values.data(), output, 16, cudaMemcpyDeviceToHost));
      CUDACHECK_TEST(cudaFree(output));
      for (std::size_t lane = 0; lane < kElements; ++lane) {
        float value = 0;
        if (type == test::MultimemReductionTestType::Float16) {
          __half raw{};
          std::memcpy(&raw, &values[lane], sizeof(raw));
          value = __half2float(raw);
        } else {
          __nv_bfloat16 raw{};
          std::memcpy(&raw, &values[lane], sizeof(raw));
          value = __bfloat162float(raw);
        }
        EXPECT_EQ(value, lane >= firstLane && lane < endLane ? expected : 0);
      }
      ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
    }
  }
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  // folly::Init consumes glog/gflags argv before other initializers see them.
  folly::Init init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::DistEnvironmentBase());
  return RUN_ALL_TESTS();
}
