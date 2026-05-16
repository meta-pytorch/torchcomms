// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "comms/pipes/MultimemHandler.h"
#include "comms/pipes/MultimemNvlTransport.h"
#include "comms/pipes/tests/MultimemNvlTransportTest.cuh"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes::tests {

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

std::vector<int> identityRankMap(int size) {
  std::vector<int> rankMap(static_cast<std::size_t>(size));
  for (int rank = 0; rank < size; ++rank) {
    rankMap[static_cast<std::size_t>(rank)] = rank;
  }
  return rankMap;
}

std::shared_ptr<meta::comms::IBootstrap> makeBootstrap(
    const std::string& prefix) {
  return std::shared_ptr<meta::comms::IBootstrap>(
      meta::comms::createBootstrap(prefix));
}

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
  for (const auto value : eligible) {
    if (value == 0) {
      return false;
    }
  }
  return true;
}

void verifyMultimemSegments(
    char* localData,
    std::size_t bytesPerRank,
    int nRanks) {
  std::vector<uint8_t> host(bytesPerRank * static_cast<std::size_t>(nRanks));
  CUDACHECK_TEST(
      cudaMemcpy(host.data(), localData, host.size(), cudaMemcpyDeviceToHost));

  for (int rank = 0; rank < nRanks; ++rank) {
    const auto expected = static_cast<uint8_t>(0x40 + rank);
    const std::size_t offset = static_cast<std::size_t>(rank) * bytesPerRank;
    for (std::size_t i = 0; i < bytesPerRank; ++i) {
      ASSERT_EQ(host[offset + i], expected) << "rank=" << rank << " byte=" << i;
    }
  }
}

test::MultimemNvlTransportTestResult runStoreSignalKernel(
    MultimemNvlTransportDevice deviceTransport,
    int rank,
    int nRanks,
    std::size_t bytesPerRank) {
  DeviceBuffer src(bytesPerRank);
  DeviceBuffer result(sizeof(test::MultimemNvlTransportTestResult));
  CUDACHECK_TEST(cudaMemset(src.get(), 0x40 + rank, bytesPerRank));
  CUDACHECK_TEST(cudaMemset(
      result.get(), 0, sizeof(test::MultimemNvlTransportTestResult)));

  test::launchMultimemStoreSignalTest(
      deviceTransport,
      static_cast<const char*>(src.get()),
      bytesPerRank,
      rank,
      nRanks,
      static_cast<test::MultimemNvlTransportTestResult*>(result.get()));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  test::MultimemNvlTransportTestResult hostResult;
  CUDACHECK_TEST(cudaMemcpy(
      &hostResult,
      result.get(),
      sizeof(test::MultimemNvlTransportTestResult),
      cudaMemcpyDeviceToHost));
  return hostResult;
}

TEST_F(
    MultimemNvlTransportTestFixture,
    EligibilityRequiresSupportAndThreeRanks) {
  EXPECT_FALSE(MultimemNvlTransport::isEligible(1, localRank));
  EXPECT_FALSE(MultimemNvlTransport::isEligible(2, localRank));
  EXPECT_EQ(
      MultimemNvlTransport::isEligible(3, localRank),
      MultimemHandler::isMultimemSupported(localRank));
}

TEST_F(MultimemNvlTransportTestFixture, StridedStoreHelperBroadcasts) {
  auto bootstrap = makeBootstrap("multimem_nvl_transport_strided_store_test");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kBytesPerRank = 4096;
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      MultimemNvlTransportConfig{
          .dataBufferSize = kBytesPerRank * static_cast<std::size_t>(numRanks),
          .userSignalCount = 1,
      });
  transport.exchange();
  const auto deviceTransport = transport.getDeviceTransport();

  DeviceBuffer src(kBytesPerRank);
  CUDACHECK_TEST(cudaMemset(src.get(), 0x40 + globalRank, kBytesPerRank));
  test::launchStridedMultimemStoreTest(
      deviceTransport,
      static_cast<const char*>(src.get()),
      kBytesPerRank,
      globalRank,
      numRanks);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  verifyMultimemSegments(deviceTransport.localData, kBytesPerRank, numRanks);
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(MultimemNvlTransportTestFixture, StoreSignalAddAndReadSignal) {
  auto bootstrap = makeBootstrap("multimem_nvl_transport_test");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kBytesPerRank = 4096;
  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      MultimemNvlTransportConfig{
          .dataBufferSize = kBytesPerRank * static_cast<std::size_t>(numRanks),
          .userSignalCount = 2,
          .internalSignalCount = 3,
      });
  transport.exchange();
  const auto deviceTransport = transport.getDeviceTransport();

  EXPECT_EQ(deviceTransport.userLocalSignals.size(), 2);
  EXPECT_EQ(deviceTransport.userMultimemSignals.size(), 2);
  EXPECT_EQ(deviceTransport.internalLocalSignals.size(), 3);
  EXPECT_EQ(deviceTransport.internalMultimemSignals.size(), 3);
  EXPECT_EQ(
      deviceTransport.internalLocalSignals.data(),
      deviceTransport.userLocalSignals.data() + 2);
  EXPECT_EQ(
      deviceTransport.internalMultimemSignals.data(),
      deviceTransport.userMultimemSignals.data() + 2);

  const auto result = runStoreSignalKernel(
      deviceTransport, globalRank, numRanks, kBytesPerRank);
  EXPECT_EQ(result.user_add_signal_value, static_cast<uint64_t>(numRanks));
  EXPECT_EQ(result.user_set_signal_value, 0x1234);
  EXPECT_EQ(result.internal_add_signal_value, static_cast<uint64_t>(numRanks));
  EXPECT_EQ(result.internal_set_signal_value, 0x5678);
  EXPECT_EQ(result.user_signal_count, 2);
  EXPECT_EQ(result.internal_signal_count, 3);
  EXPECT_EQ(
      result.internal_signal_addr,
      result.user_signal_addr + 2 * sizeof(SignalState));

  verifyMultimemSegments(deviceTransport.localData, kBytesPerRank, numRanks);
  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::DistEnvironmentBase());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
