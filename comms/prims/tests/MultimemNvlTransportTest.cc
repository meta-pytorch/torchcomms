// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <folly/futures/Future.h>
#include <folly/init/Init.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
    std::size_t maxGroups = 0) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = dataBufferSize;
  config.userSignalCount = userSignalCount;
  config.pipelineDepth = pipelineDepth;
  config.maxGroups = maxGroups;
  return config;
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

TEST(MultimemNvlTransportConfigTest, DerivesUserOnlyConfiguration) {
  const auto validation =
      validate_multimem_nvl_transport_config(makeConfig(256), 4);
  EXPECT_TRUE(validation);
  EXPECT_EQ(validation.internalSignalCount, 0);
}

TEST(MultimemNvlTransportConfigTest, DerivesStagingOnlyConfiguration) {
  const auto validation =
      validate_multimem_nvl_transport_config(makeConfig(256, 0, 2, 2), 4);
  EXPECT_TRUE(validation);
  EXPECT_EQ(validation.internalSignalCount, 48);
}

TEST(MultimemNvlTransportConfigTest, DerivesMixedConfiguration) {
  const auto validation =
      validate_multimem_nvl_transport_config(makeConfig(256, 7, 2, 2), 4);
  EXPECT_TRUE(validation);
  EXPECT_EQ(validation.internalSignalCount, 48);
}

TEST(MultimemNvlTransportConfigTest, RejectsPartialStagingGeometry) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(256, 1, 2, 0), 4).error,
      MultimemNvlTransportConfigError::PartialStagingGeometry);
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(256, 1, 0, 2), 4).error,
      MultimemNvlTransportConfigError::PartialStagingGeometry);
}

TEST(MultimemNvlTransportConfigTest, RejectsInvalidRankCount) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(256, 1, 1, 1), 0).error,
      MultimemNvlTransportConfigError::InvalidRankCount);
}

TEST(MultimemNvlTransportConfigTest, RejectsMissingDataBuffer) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(0, 1), 4).error,
      MultimemNvlTransportConfigError::MissingDataBuffer);
}

TEST(MultimemNvlTransportConfigTest, RejectsNoSignalSlots) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(256, 0), 4).error,
      MultimemNvlTransportConfigError::NoSignalSlots);
}

TEST(MultimemNvlTransportConfigTest, RejectsMaxGroupsOutOfRange) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(
          makeConfig(
              std::numeric_limits<std::size_t>::max(),
              1,
              1,
              static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) +
                  1),
          4)
          .error,
      MultimemNvlTransportConfigError::GeometryOutOfRange);
}

TEST(MultimemNvlTransportConfigTest, RejectsInsufficientDataCapacity) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(255, 1, 2, 2), 4).error,
      MultimemNvlTransportConfigError::InsufficientDataCapacity);
}

TEST(MultimemNvlTransportConfigTest, RejectsInternalSignalOverflow) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(
          makeConfig(
              std::numeric_limits<std::size_t>::max(),
              1,
              std::numeric_limits<std::size_t>::max(),
              2),
          4)
          .error,
      MultimemNvlTransportConfigError::GeometryOutOfRange);
}

TEST(MultimemNvlTransportConfigTest, RejectsTotalSignalOverflow) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(
          makeConfig(
              std::numeric_limits<std::size_t>::max(),
              std::numeric_limits<uint32_t>::max(),
              1,
              1),
          4)
          .error,
      MultimemNvlTransportConfigError::SignalCountOverflow);
}

TEST(MultimemNvlTransportConfigTest, ResolvesExplicitOverride) {
  const auto overrideConfig = makeConfig(8192, 3, 2, 2);
  const auto fallbackConfig = makeConfig(4096, 1, 1, 1);
  const auto resolved = resolve_multimem_nvl_transport_config(
      overrideConfig, fallbackConfig, 2048, 4);

  EXPECT_TRUE(resolved);
  EXPECT_EQ(resolved.config, overrideConfig);
  EXPECT_EQ(resolved.internalSignalCount, 48);
}

TEST(MultimemNvlTransportConfigTest, ResolvesAbsentOverrideFromFallback) {
  const auto fallbackConfig = makeConfig(4096, 1, 1, 1);
  const auto resolved = resolve_multimem_nvl_transport_config(
      std::nullopt, fallbackConfig, 2048, 4);

  EXPECT_TRUE(resolved);
  EXPECT_EQ(resolved.config, fallbackConfig);
}

TEST(MultimemNvlTransportConfigTest, ResolvesTopologyDataBufferSize) {
  const auto overrideConfig = makeConfig(0, 1, 1, 1);
  const auto resolved = resolve_multimem_nvl_transport_config(
      overrideConfig, makeConfig(4096, 1, 1, 1), 2048, 4);

  EXPECT_TRUE(resolved);
  EXPECT_EQ(resolved.config.dataBufferSize, 2048);
}

TEST(MultimemNvlTransportConfigTest, ReturnsAttemptedConfigOnError) {
  const auto overrideConfig = makeConfig(8192, 1, 0, 1);
  const auto resolved = resolve_multimem_nvl_transport_config(
      overrideConfig, makeConfig(4096, 1, 1, 1), 2048, 4);

  EXPECT_FALSE(resolved);
  EXPECT_EQ(resolved.config, overrideConfig);
  EXPECT_EQ(
      resolved.error, MultimemNvlTransportConfigError::PartialStagingGeometry);
}

TEST(MultimemNvlTransportConfigTest, DescribesEveryError) {
  EXPECT_STREQ(
      multimem_nvl_transport_config_error_string(
          MultimemNvlTransportConfigError::None),
      "none");
  EXPECT_STREQ(
      multimem_nvl_transport_config_error_string(
          MultimemNvlTransportConfigError::MissingDataBuffer),
      "data buffer size must be non-zero");
  EXPECT_STREQ(
      multimem_nvl_transport_config_error_string(
          MultimemNvlTransportConfigError::InvalidRankCount),
      "NVL rank count must be positive");
  EXPECT_STREQ(
      multimem_nvl_transport_config_error_string(
          MultimemNvlTransportConfigError::PartialStagingGeometry),
      "pipeline depth and maximum groups must both be zero or non-zero");
  EXPECT_STREQ(
      multimem_nvl_transport_config_error_string(
          MultimemNvlTransportConfigError::GeometryOutOfRange),
      "pipeline depth or maximum groups exceeds UINT32_MAX");
  EXPECT_STREQ(
      multimem_nvl_transport_config_error_string(
          MultimemNvlTransportConfigError::InsufficientDataCapacity),
      "data buffer is too small for the staging geometry");
  EXPECT_STREQ(
      multimem_nvl_transport_config_error_string(
          MultimemNvlTransportConfigError::SignalCountOverflow),
      "signal count exceeds INT_MAX");
  EXPECT_STREQ(
      multimem_nvl_transport_config_error_string(
          MultimemNvlTransportConfigError::NoSignalSlots),
      "at least one signal slot is required");
  EXPECT_STREQ(
      multimem_nvl_transport_config_error_string(
          static_cast<MultimemNvlTransportConfigError>(-1)),
      "unknown configuration error");
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
      .multimem =
          MultimemNvlTransportConfig{
              .dataBufferSize = 0,
              .userSignalCount = 1,
              .pipelineDepth = 1,
              .maxGroups = 1,
          },
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
        .multimem =
            MultimemNvlTransportConfig{
                .dataBufferSize = 4096,
                .userSignalCount = 1,
                .pipelineDepth = 1,
                .maxGroups = 1,
            },
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
      .multimem =
          MultimemNvlTransportConfig{
              .dataBufferSize = 4096,
              .userSignalCount = 1,
              .pipelineDepth = 1,
              .maxGroups = 1,
          },
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
      .multimem =
          MultimemNvlTransportConfig{
              .dataBufferSize = 4096,
              .userSignalCount = 1,
              .pipelineDepth = 1,
              .maxGroups = 1,
          },
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
      .multimem =
          MultimemNvlTransportConfig{
              .dataBufferSize =
                  kBytesPerRank * static_cast<std::size_t>(numRanks),
              .userSignalCount = 1,
              .pipelineDepth = 1,
              .maxGroups = 1,
          },
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
      .multimem =
          MultimemNvlTransportConfig{
              .dataBufferSize =
                  globalRank == 0 ? std::size_t{0} : std::size_t{4096},
              .userSignalCount = 1,
              .pipelineDepth = 1,
              .maxGroups = 1,
          },
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
  const uint32_t internalSignals =
      multimem_staging_signals_per_lane(static_cast<uint32_t>(numRanks));

  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(kDataBytes, kUserSignals, 1, 1));

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
  EXPECT_EQ(handle.maxGroups, 1);
  EXPECT_EQ(handle.signalsPerLane, internalSignals);

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
  constexpr uint32_t kMaxGroups = 4;
  constexpr uint32_t kActiveGroups = 3;
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      makeConfig(kDataBytes, 0, kPipelineDepth, kMaxGroups));
  transport.exchange();

  test::StageLayoutResult* deviceResults = nullptr;
  CUDACHECK_TEST(
      cudaMalloc(&deviceResults, kMaxGroups * sizeof(test::StageLayoutResult)));
  test::launchStageLayout(
      transport.getDeviceTransport(), deviceResults, kActiveGroups);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<test::StageLayoutResult> results(kActiveGroups);
  CUDACHECK_TEST(cudaMemcpy(
      results.data(),
      deviceResults,
      results.size() * sizeof(test::StageLayoutResult),
      cudaMemcpyDeviceToHost));
  const uint64_t signalsPerLane =
      multimem_staging_signals_per_lane(static_cast<uint32_t>(numRanks));
  for (uint32_t group = 0; group < kActiveGroups; ++group) {
    EXPECT_EQ(results[group].groupBeginBytes, group * 4096);
    EXPECT_EQ(results[group].stagingBytes, 2048);
    EXPECT_EQ(
        results[group].signalBase, group * kPipelineDepth * signalsPerLane);
    EXPECT_EQ(results[group].signalsPerLane, signalsPerLane);
    EXPECT_EQ(results[group].pipelineDepth, kPipelineDepth);
  }

  test::launchStageLayout(
      transport.getDeviceTransport(), deviceResults, kMaxGroups);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  results.resize(kMaxGroups);
  CUDACHECK_TEST(cudaMemcpy(
      results.data(),
      deviceResults,
      results.size() * sizeof(test::StageLayoutResult),
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaFree(deviceResults));
  for (uint32_t group = 0; group < kMaxGroups; ++group) {
    EXPECT_EQ(results[group].groupBeginBytes, group * 3072);
    EXPECT_EQ(results[group].stagingBytes, 1536);
    EXPECT_EQ(
        results[group].signalBase, group * kPipelineDepth * signalsPerLane);
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

} // namespace

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

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  // folly::Init consumes glog/gflags argv before other initializers see them.
  folly::Init init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::DistEnvironmentBase());
  return RUN_ALL_TESTS();
}
