// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/window/WindowMemory.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using comms::pipes::MemSharingMode;
using comms::pipes::WindowMemory;
using comms::pipes::WindowMemoryConfig;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::tests {

class WindowMemoryTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }
};

/**
 * Test construction with valid parameters.
 */
TEST_F(WindowMemoryTestFixture, Construction) {
  auto bootstrap = std::make_shared<MpiBootstrap>();
  WindowMemoryConfig config{.signalCount = 4};
  WindowMemory signals(globalRank, numRanks, bootstrap, config);

  EXPECT_EQ(signals.rank(), globalRank);
  EXPECT_EQ(signals.nRanks(), numRanks);
  EXPECT_EQ(signals.signalCount(), 4);
  EXPECT_FALSE(signals.isExchanged());
}

/**
 * Test exchange and state verification.
 */
TEST_F(WindowMemoryTestFixture, ExchangeAndStateVerification) {
  auto bootstrap = std::make_shared<MpiBootstrap>();
  WindowMemoryConfig config{.signalCount = 2};
  WindowMemory signals(globalRank, numRanks, bootstrap, config);

  signals.exchange();

  EXPECT_TRUE(signals.isExchanged());
  // Note: getDeviceWindowSignal() is tested via MultiPeerNvlTransport
  // integration tests since it requires CUDA compilation
}

/**
 * Test various signal counts.
 */
TEST_F(WindowMemoryTestFixture, VariousSignalCounts) {
  auto bootstrap = std::make_shared<MpiBootstrap>();

  for (std::size_t signalCount : {1, 2, 4, 8, 16}) {
    WindowMemoryConfig config{.signalCount = signalCount};
    WindowMemory signals(globalRank, numRanks, bootstrap, config);

    signals.exchange();

    EXPECT_EQ(signals.signalCount(), signalCount);
    EXPECT_TRUE(signals.isExchanged());
  }
}

/**
 * Test explicit memory sharing modes.
 */
TEST_F(WindowMemoryTestFixture, ExplicitCudaIpcMode) {
  auto bootstrap = std::make_shared<MpiBootstrap>();
  WindowMemoryConfig config{.signalCount = 2};

  // Explicitly request cudaIpc mode
  WindowMemory signals(
      globalRank, numRanks, bootstrap, config, MemSharingMode::kCudaIpc);

  EXPECT_EQ(signals.getMemSharingMode(), MemSharingMode::kCudaIpc);

  signals.exchange();

  EXPECT_TRUE(signals.isExchanged());
}

/**
 * Test single rank scenario.
 */
TEST_F(WindowMemoryTestFixture, SingleRankExchange) {
  // This test creates WindowMemory with nRanks=1, which requires a single-rank
  // MPI environment to avoid bootstrap/rank mismatch issues
  if (numRanks > 1) {
    GTEST_SKIP() << "Single rank test requires single MPI rank";
  }

  auto bootstrap = std::make_shared<MpiBootstrap>();
  WindowMemoryConfig config{.signalCount = 2};

  WindowMemory signals(0 /* myRank */, 1 /* nRanks */, bootstrap, config);

  signals.exchange();

  EXPECT_TRUE(signals.isExchanged());
}

// Note: getDeviceWindowSignal() error path (calling before exchange()) and
// success path are tested via MultiPeerNvlTransport integration tests since
// DeviceWindowSignal requires CUDA compilation (.cuh file with __device__
// methods).

} // namespace comms::pipes::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
