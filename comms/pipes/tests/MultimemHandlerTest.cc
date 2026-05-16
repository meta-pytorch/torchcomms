// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "comms/pipes/MultimemHandler.h"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes::tests {

class MultimemHandlerTestFixture : public ::testing::Test,
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

std::vector<int> reverseRankMap(int size) {
  std::vector<int> rankMap(static_cast<std::size_t>(size));
  for (int rank = 0; rank < size; ++rank) {
    rankMap[static_cast<std::size_t>(rank)] = size - 1 - rank;
  }
  return rankMap;
}

std::shared_ptr<meta::comms::IBootstrap> makeBootstrap(
    const std::string& prefix) {
  return std::shared_ptr<meta::comms::IBootstrap>(
      meta::comms::createBootstrap(prefix));
}

bool allRanksMultimemSupported(
    const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
    int rank,
    int nRanks,
    int cudaDevice) {
  std::vector<int> supported(static_cast<std::size_t>(nRanks));
  supported[static_cast<std::size_t>(rank)] =
      MultimemHandler::isMultimemSupported(cudaDevice) ? 1 : 0;
  auto supportRc =
      bootstrap->allGather(supported.data(), sizeof(int), rank, nRanks).get();
  EXPECT_EQ(supportRc, 0);
  if (supportRc != 0) {
    return false;
  }
  bool allRanksSupported = true;
  for (const auto value : supported) {
    allRanksSupported = allRanksSupported && value != 0;
  }
  return allRanksSupported;
}

TEST_F(MultimemHandlerTestFixture, ExchangeSetsUpStableMappings) {
  if (numRanks < 3) {
    GTEST_SKIP() << "CUDA multimem transport is only useful for 3+ ranks";
  }
  auto bootstrap = makeBootstrap("multimem_handler_test");
  if (!allRanksMultimemSupported(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not supported";
  }

  constexpr std::size_t kRequestedBytes = 4096;
  MultimemHandler handler(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      localRank,
      kRequestedBytes);
  EXPECT_THROW(handler.getLocalDeviceMemPtr(), std::runtime_error);
  EXPECT_THROW(handler.getMultimemDeviceMemPtr(), std::runtime_error);

  handler.exchange();

  auto* localPtr = handler.getLocalDeviceMemPtr();
  auto* multimemPtr = handler.getMultimemDeviceMemPtr();

  EXPECT_NE(localPtr, nullptr);
  EXPECT_NE(multimemPtr, nullptr);
  EXPECT_NE(localPtr, multimemPtr);
  EXPECT_EQ(localPtr, handler.getLocalDeviceMemPtr());
  EXPECT_EQ(multimemPtr, handler.getMultimemDeviceMemPtr());
  EXPECT_GE(handler.getAllocatedSize(), kRequestedBytes);

  std::vector<std::uintptr_t> multimemAddrs(static_cast<std::size_t>(numRanks));
  multimemAddrs[static_cast<std::size_t>(globalRank)] =
      reinterpret_cast<std::uintptr_t>(multimemPtr);
  auto rc = bootstrap
                ->allGather(
                    multimemAddrs.data(),
                    sizeof(std::uintptr_t),
                    globalRank,
                    numRanks)
                .get();
  ASSERT_EQ(rc, 0);
  for (int rank = 0; rank < numRanks; ++rank) {
    EXPECT_NE(multimemAddrs[static_cast<std::size_t>(rank)], 0);
  }

  handler.exchange();
  EXPECT_EQ(localPtr, handler.getLocalDeviceMemPtr());
  EXPECT_EQ(multimemPtr, handler.getMultimemDeviceMemPtr());

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(MultimemHandlerTestFixture, ExchangeSupportsNonIdentityRankMap) {
  if (numRanks < 3) {
    GTEST_SKIP() << "CUDA multimem transport is only useful for 3+ ranks";
  }
  auto bootstrap = makeBootstrap("multimem_handler_non_identity_rank_map_test");
  if (!allRanksMultimemSupported(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not supported";
  }

  constexpr std::size_t kRequestedBytes = 4096;
  MultimemHandler handler(
      bootstrap,
      globalRank,
      reverseRankMap(numRanks),
      localRank,
      kRequestedBytes);
  handler.exchange();

  EXPECT_NE(handler.getLocalDeviceMemPtr(), nullptr);
  EXPECT_NE(handler.getMultimemDeviceMemPtr(), nullptr);
  EXPECT_GE(handler.getAllocatedSize(), kRequestedBytes);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::DistEnvironmentBase());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
