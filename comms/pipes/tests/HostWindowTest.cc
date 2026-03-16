// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/window/HostWindow.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using comms::pipes::HostWindow;
using comms::pipes::MultiPeerTransport;
using comms::pipes::MultiPeerTransportConfig;
using comms::pipes::WindowConfig;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::tests {

class HostWindowTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }

  std::unique_ptr<MultiPeerTransport> createTransport() {
    auto bootstrap = std::make_shared<MpiBootstrap>();
    MultiPeerTransportConfig config;
    config.nvlConfig.dataBufferSize = 1024 * 1024;
    config.nvlConfig.chunkSize = 64 * 1024;
    config.nvlConfig.pipelineDepth = 2;
    auto transport = std::make_unique<MultiPeerTransport>(
        globalRank, numRanks, localRank, bootstrap, config);
    transport->exchange();
    return transport;
  }
};

TEST_F(HostWindowTestFixture, Construction) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 4, .barrierCount = 2};
  HostWindow window(*transport, config);

  EXPECT_EQ(window.rank(), globalRank);
  EXPECT_EQ(window.nRanks(), numRanks);
  EXPECT_EQ(window.config().peerSignalCount, 4);
  EXPECT_EQ(window.config().barrierCount, 2);
  EXPECT_FALSE(window.isExchanged());
}

TEST_F(HostWindowTestFixture, ExchangeAndStateVerification) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 2};
  HostWindow window(*transport, config);

  window.exchange();

  EXPECT_TRUE(window.isExchanged());
}

TEST_F(HostWindowTestFixture, VariousSignalCounts) {
  auto transport = createTransport();

  for (std::size_t signalCount : {1, 2, 4, 8, 16}) {
    WindowConfig config{.peerSignalCount = signalCount};
    HostWindow window(*transport, config);

    window.exchange();

    EXPECT_EQ(window.config().peerSignalCount, signalCount);
    EXPECT_TRUE(window.isExchanged());
  }
}

TEST_F(HostWindowTestFixture, ZeroCountConfig) {
  auto transport = createTransport();
  WindowConfig config{};
  HostWindow window(*transport, config);

  window.exchange();

  EXPECT_TRUE(window.isExchanged());
  EXPECT_GE(window.numNvlPeers() + window.numIbgdaPeers(), numRanks - 1);
}

TEST_F(HostWindowTestFixture, WithBarriersAndCounters) {
  auto transport = createTransport();
  WindowConfig config{
      .peerSignalCount = 2, .peerCounterCount = 1, .barrierCount = 4};
  HostWindow window(*transport, config);

  window.exchange();

  EXPECT_TRUE(window.isExchanged());
  EXPECT_EQ(window.config().peerSignalCount, 2);
  EXPECT_EQ(window.config().peerCounterCount, 1);
  EXPECT_EQ(window.config().barrierCount, 4);
}

// Note: getDeviceWindow() returns a CUDA type (DeviceWindow) defined in a .cuh
// header that requires CUDA compilation. Its success and error paths are
// validated by the DeviceWindow unit tests and integration tests.

// =============================================================================
// registerLocalBuffer Tests
// =============================================================================

TEST_F(HostWindowTestFixture, RegisterLocalBufferBeforeExchange) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);

  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, 1024));

  EXPECT_EQ(window.registerLocalBuffer(buf, 1024), 0);

  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(HostWindowTestFixture, RegisterLocalBufferAfterExchange) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, 4096));

  int regIdx = window.registerLocalBuffer(buf, 4096);
  EXPECT_EQ(regIdx, 0);

  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(HostWindowTestFixture, RegisterMultipleLocalBuffers) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  void* buf0 = nullptr;
  void* buf1 = nullptr;
  void* buf2 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf0, 1024));
  CUDACHECK_TEST(cudaMalloc(&buf1, 2048));
  CUDACHECK_TEST(cudaMalloc(&buf2, 4096));

  EXPECT_EQ(window.registerLocalBuffer(buf0, 1024), 0);
  EXPECT_EQ(window.registerLocalBuffer(buf1, 2048), 1);
  EXPECT_EQ(window.registerLocalBuffer(buf2, 4096), 2);

  CUDACHECK_TEST(cudaFree(buf0));
  CUDACHECK_TEST(cudaFree(buf1));
  CUDACHECK_TEST(cudaFree(buf2));
}

// =============================================================================
// registerAndExchangeBuffer Tests
// =============================================================================

TEST_F(HostWindowTestFixture, RegisterAndExchangeBufferBeforeExchange) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);

  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, 1024));

  EXPECT_EQ(window.registerAndExchangeBuffer(buf, 1024), 0);

  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(HostWindowTestFixture, RegisterAndExchangeBufferAfterExchange) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, 4096));

  int regIdx = window.registerAndExchangeBuffer(buf, 4096);
  EXPECT_EQ(regIdx, 0);

  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(HostWindowTestFixture, RegisterAndExchangeBufferCalledTwiceThrows) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  void* buf0 = nullptr;
  void* buf1 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf0, 1024));
  CUDACHECK_TEST(cudaMalloc(&buf1, 1024));

  window.registerAndExchangeBuffer(buf0, 1024);
  EXPECT_THROW(
      window.registerAndExchangeBuffer(buf1, 1024), std::runtime_error);

  CUDACHECK_TEST(cudaFree(buf0));
  CUDACHECK_TEST(cudaFree(buf1));
}

// =============================================================================
// Mixed registration Tests
// =============================================================================

TEST_F(HostWindowTestFixture, LocalThenExchangeBufferIndicesAreSequential) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  void* localBuf = nullptr;
  void* dstBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&localBuf, 1024));
  CUDACHECK_TEST(cudaMalloc(&dstBuf, 2048));

  EXPECT_EQ(window.registerLocalBuffer(localBuf, 1024), 0);
  EXPECT_EQ(window.registerAndExchangeBuffer(dstBuf, 2048), 1);

  CUDACHECK_TEST(cudaFree(localBuf));
  CUDACHECK_TEST(cudaFree(dstBuf));
}

TEST_F(HostWindowTestFixture, ExchangeThenLocalBufferIndicesAreSequential) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  void* dstBuf = nullptr;
  void* localBuf0 = nullptr;
  void* localBuf1 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&dstBuf, 2048));
  CUDACHECK_TEST(cudaMalloc(&localBuf0, 1024));
  CUDACHECK_TEST(cudaMalloc(&localBuf1, 4096));

  EXPECT_EQ(window.registerAndExchangeBuffer(dstBuf, 2048), 0);
  EXPECT_EQ(window.registerLocalBuffer(localBuf0, 1024), 1);
  EXPECT_EQ(window.registerLocalBuffer(localBuf1, 4096), 2);

  CUDACHECK_TEST(cudaFree(dstBuf));
  CUDACHECK_TEST(cudaFree(localBuf0));
  CUDACHECK_TEST(cudaFree(localBuf1));
}

} // namespace comms::pipes::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
