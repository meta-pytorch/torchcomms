// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <memory>
#include <vector>

#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/TopologyDiscovery.h"
#include "comms/pipes/tests/DeviceWindowIbgdaIntegrationTest.cuh"
#include "comms/pipes/window/DeviceWindow.cuh"
#include "comms/pipes/window/HostWindow.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::tests {

// =============================================================================
// Test Fixture: forces all peers to IBGDA via TopologyConfig{p2pDisable=true}
// so DeviceWindow::put dispatches to the IBGDA path (and exercises ringDb).
// =============================================================================

class DeviceWindowIbgdaIntegrationTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MpiBaseTestFixture::TearDown();
  }

  struct TransportBundle {
    std::unique_ptr<MultiPeerTransport> transport;
    std::unique_ptr<HostWindow> window;
    DeviceWindow dw;
  };

  TransportBundle createIbgdaTransport(
      const WindowConfig& wmConfig,
      void* userBuffer,
      std::size_t userBufferSize) {
    MultiPeerTransportConfig config{
        .topoConfig = TopologyConfig{.p2pDisable = true},
        .ibgdaConfig = {.cudaDevice = localRank},
    };
    auto bootstrap = std::make_shared<MpiBootstrap>();
    auto transport = std::make_unique<MultiPeerTransport>(
        globalRank, numRanks, localRank, bootstrap, config);
    transport->exchange();

    auto window = std::make_unique<HostWindow>(
        *transport, wmConfig, userBuffer, userBufferSize);
    window->exchange();

    DeviceWindow dw = window->getDeviceWindow();
    return {std::move(transport), std::move(window), dw};
  }
};

// =============================================================================
// PutNoDbBatching — DeviceWindow::put(ringDb=false) + signal_peer
// =============================================================================

TEST_F(DeviceWindowIbgdaIntegrationTestFixture, PutNoDbBatching) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  const std::size_t nbytes = 64 * 1024;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t testPattern = 0x55;
  const int signalId = 0;

  try {
    DeviceBuffer userBuffer(nbytes);
    CUDACHECK_TEST(cudaMemset(userBuffer.get(), 0, nbytes));

    WindowConfig wmConfig{.peerSignalCount = 1};
    auto [transport, window, dw] =
        createIbgdaTransport(wmConfig, userBuffer.get(), nbytes);

    DeviceBuffer srcBuffer(nbytes);
    auto lkey = window->registerLocalBuffer(srcBuffer.get(), nbytes);
    ASSERT_TRUE(lkey.has_value())
        << "registerLocalBuffer should succeed when IBGDA peers exist";
    LocalBufferRegistration src{
        .base = srcBuffer.get(),
        .size = nbytes,
        .lkey = *lkey,
    };

    if (globalRank == 0) {
      std::vector<uint8_t> hostPattern(nbytes, testPattern);
      CUDACHECK_TEST(cudaMemcpy(
          srcBuffer.get(), hostPattern.data(), nbytes, cudaMemcpyHostToDevice));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testDeviceWindowPutNoDbAndSignal(
          dw, peerRank, src, nbytes, signalId);
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testDeviceWindowWaitSignal(dw, signalId, /*expectedValue=*/1);
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      std::vector<uint8_t> hostBuf(nbytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), userBuffer.get(), nbytes, cudaMemcpyDeviceToHost));
      std::size_t mismatches = 0;
      for (auto b : hostBuf) {
        if (b != testPattern) {
          mismatches++;
        }
      }
      EXPECT_EQ(mismatches, 0u)
          << "DeviceWindow::put(ringDb=false) + signal_peer: " << mismatches
          << " byte mismatches out of " << nbytes;
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(
      INFO, "Rank {}: DeviceWindow PutNoDbBatching test completed", globalRank);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
