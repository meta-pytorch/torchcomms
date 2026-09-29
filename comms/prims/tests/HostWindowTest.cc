// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <memory>
#include <stdexcept>

#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/prims/window/HostWindow.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using comms::prims::HostWindow;
using comms::prims::MultiPeerTransport;
using comms::prims::MultiPeerTransportConfig;
using comms::prims::WindowConfig;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::prims::tests {

class HostWindowTestPeer {
 public:
  static void setCallerBufferLifetimeQuarantineRequired(
      HostWindow& window,
      bool quarantined) {
    window.callerBufferLifetimeQuarantineRequired_ = quarantined;
  }

  static bool callerBufferPossiblyExposed(const HostWindow& window) {
    return window.callerBufferPossiblyExposed_;
  }

  static std::shared_ptr<void>* lastCallerBufferKeepAlive(HostWindow& window) {
    return window.callerBufferKeepAlives_
        .at(window.callerBufferKeepAlives_.size() - 1)
        .get();
  }
};

std::shared_ptr<void> makeCudaBuffer(std::size_t size) {
  void* ptr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&ptr, size));
  return std::shared_ptr<void>(
      ptr, [](void* allocation) { static_cast<void>(cudaFree(allocation)); });
}

class HostWindowTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }

  std::unique_ptr<MultiPeerTransport> createTransport(bool disableIb = false) {
    auto bootstrap = std::make_shared<MpiBootstrap>();
    MultiPeerTransportConfig config;
    config.nvlConfig.pipelineDepth = 2;
    config.nvlConfig.maxNumChannels = 64;
    config.nvlConfig.perChannelSize = 16 * 1024;
    config.disableIb = disableIb;
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

  auto buffer = makeCudaBuffer(1024);

  // No IBGDA peers in mock transport → returns nullopt
  EXPECT_FALSE(window.registerLocalBuffer(buffer, 1024).has_value());
}

TEST_F(HostWindowTestFixture, RegisterLocalBufferAfterExchange) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  auto buffer = makeCudaBuffer(4096);

  EXPECT_FALSE(window.registerLocalBuffer(buffer, 4096).has_value());
}

TEST_F(HostWindowTestFixture, RegisterMultipleLocalBuffers) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  auto buffer0 = makeCudaBuffer(1024);
  auto buffer1 = makeCudaBuffer(2048);
  auto buffer2 = makeCudaBuffer(4096);

  EXPECT_FALSE(window.registerLocalBuffer(buffer0, 1024).has_value());
  EXPECT_FALSE(window.registerLocalBuffer(buffer1, 2048).has_value());
  EXPECT_FALSE(window.registerLocalBuffer(buffer2, 4096).has_value());
}

// =============================================================================
// registerAndExchangeBuffer Tests
// =============================================================================

TEST_F(HostWindowTestFixture, RegisterAndExchangeBufferBeforeExchange) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);

  auto buffer = makeCudaBuffer(1024);

  window.registerAndExchangeBuffer(buffer, 1024);
  EXPECT_TRUE(HostWindowTestPeer::callerBufferPossiblyExposed(window));
}

TEST_F(HostWindowTestFixture, RegisterAndExchangeBufferAfterExchange) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  auto buffer = makeCudaBuffer(4096);

  window.registerAndExchangeBuffer(buffer, 4096);
}

TEST_F(HostWindowTestFixture, RegisterAndExchangeBufferCalledTwiceThrows) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  auto buffer0 = makeCudaBuffer(1024);
  auto buffer1 = makeCudaBuffer(1024);

  window.registerAndExchangeBuffer(buffer0, 1024);
  EXPECT_THROW(
      window.registerAndExchangeBuffer(buffer1, 1024), std::runtime_error);
}

TEST_F(HostWindowTestFixture, RejectsInvalidOwnedBuffers) {
  auto transport = createTransport();
  WindowConfig config{};
  auto buffer = makeCudaBuffer(1024);

  EXPECT_THROW(
      (HostWindow(
          *transport, config, std::shared_ptr<void>{}, /*userBufferSize=*/1)),
      std::invalid_argument);
  EXPECT_THROW(
      (HostWindow(*transport, config, buffer, /*userBufferSize=*/0)),
      std::invalid_argument);

  HostWindow window(*transport, config);
  EXPECT_THROW(
      window.registerLocalBuffer(std::shared_ptr<void>{}, 1),
      std::invalid_argument);
  EXPECT_THROW(window.registerLocalBuffer(buffer, 0), std::invalid_argument);
  EXPECT_THROW(
      window.registerAndExchangeBuffer(std::shared_ptr<void>{}, 1),
      std::invalid_argument);
  EXPECT_THROW(
      window.registerAndExchangeBuffer(buffer, 0), std::invalid_argument);
}

TEST_F(HostWindowTestFixture, RetainsCallerBufferUntilWindowDestruction) {
  auto transport = createTransport();
  std::weak_ptr<void> weakOwner;
  {
    WindowConfig config{};
    HostWindow window(*transport, config);
    window.exchange();

    auto buffer = makeCudaBuffer(1024);
    weakOwner = buffer;
    window.registerAndExchangeBuffer(buffer, 1024);
    buffer.reset();
    EXPECT_FALSE(weakOwner.expired());
  }
  EXPECT_TRUE(weakOwner.expired());
}

TEST_F(HostWindowTestFixture, NoIbLocalRegistrationDoesNotRetainOwner) {
  auto transport = createTransport(/*disableIb=*/true);
  WindowConfig config{};
  HostWindow window(*transport, config);
  window.exchange();

  auto buffer = makeCudaBuffer(1024);
  std::weak_ptr<void> weakOwner = buffer;
  EXPECT_FALSE(window.registerLocalBuffer(buffer, 1024).has_value());
  buffer.reset();
  EXPECT_TRUE(weakOwner.expired());
}

TEST_F(
    HostWindowTestFixture,
    QuarantineDuringExceptionUnwindRetainsCallerOwner) {
  auto transport = createTransport(/*disableIb=*/true);
  std::weak_ptr<void> weakOwner;
  std::shared_ptr<void>* retainedHolder = nullptr;

  try {
    WindowConfig config{};
    HostWindow window(*transport, config);
    window.exchange();

    auto buffer = makeCudaBuffer(1024);
    weakOwner = buffer;
    window.registerAndExchangeBuffer(buffer, 1024);
    retainedHolder = HostWindowTestPeer::lastCallerBufferKeepAlive(window);
    buffer.reset();
    HostWindowTestPeer::setCallerBufferLifetimeQuarantineRequired(
        window, /*quarantined=*/true);
    throw std::runtime_error("force HostWindow exception unwind");
  } catch (const std::runtime_error& ex) {
    EXPECT_STREQ(ex.what(), "force HostWindow exception unwind");
  }

  ASSERT_NE(retainedHolder, nullptr);
  EXPECT_FALSE(weakOwner.expired());
  delete retainedHolder;
  EXPECT_TRUE(weakOwner.expired());
}

// =============================================================================
// Mixed registration Tests
// =============================================================================

TEST_F(HostWindowTestFixture, LocalThenExchangeBuffer) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  auto localBuffer = makeCudaBuffer(1024);
  auto dstBuffer = makeCudaBuffer(2048);

  EXPECT_FALSE(window.registerLocalBuffer(localBuffer, 1024).has_value());
  window.registerAndExchangeBuffer(dstBuffer, 2048);
}

TEST_F(HostWindowTestFixture, ExchangeThenLocalBuffer) {
  auto transport = createTransport();
  WindowConfig config{.peerSignalCount = 1};
  HostWindow window(*transport, config);
  window.exchange();

  auto dstBuffer = makeCudaBuffer(2048);
  auto localBuffer0 = makeCudaBuffer(1024);
  auto localBuffer1 = makeCudaBuffer(4096);

  window.registerAndExchangeBuffer(dstBuffer, 2048);
  EXPECT_FALSE(window.registerLocalBuffer(localBuffer0, 1024).has_value());
  EXPECT_FALSE(window.registerLocalBuffer(localBuffer1, 4096).has_value());
}

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
