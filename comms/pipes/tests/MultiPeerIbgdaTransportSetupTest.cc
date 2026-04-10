// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <mpi.h>

#include <cuda_runtime.h>
#include <memory>
#include <set>

#include <folly/init/Init.h>

#include "comms/pipes/MultiPeerIbgdaTransportSetup.h"
#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace comms::pipes;
using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

class MultiPeerIbgdaTransportSetupTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  std::unique_ptr<MultipeerIbgdaTransport> createTransport() {
    MultipeerIbgdaTransportConfig config{
        .cudaDevice = localRank,
    };
    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    auto transport = std::make_unique<MultipeerIbgdaTransport>(
        globalRank, numRanks, bootstrap, config);
    transport->exchange();
    return transport;
  }
};

TEST_F(MultiPeerIbgdaTransportSetupTestFixture, ConstructAndExchange) {
  std::unique_ptr<MultipeerIbgdaTransport> transport;
  try {
    transport = createTransport();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 2048,
      .chunkSize = 512,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  auto& hostPeerStates = setup.getHostPeerStates();
  EXPECT_EQ(hostPeerStates.size(), static_cast<size_t>(numRanks));

  auto* iterCounter = setup.getIterationCounter();
  EXPECT_NE(iterCounter, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerIbgdaTransportSetupTestFixture, IterationCounterZeroInit) {
  std::unique_ptr<MultipeerIbgdaTransport> transport;
  try {
    transport = createTransport();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 1024,
      .chunkSize = 256,
      .pipelineDepth = 2,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  uint64_t* d_counter = setup.getIterationCounter();
  ASSERT_NE(d_counter, nullptr);

  uint64_t h_counter = 0xDEAD;
  CUDACHECK_TEST(cudaMemcpy(
      &h_counter, d_counter, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_counter, 0u);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerIbgdaTransportSetupTestFixture, PeerStateOffsets) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }

  std::unique_ptr<MultipeerIbgdaTransport> transport;
  try {
    transport = createTransport();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  const auto& h_states = setup.getHostPeerStates();

  // Verify config is propagated correctly
  for (int peer = 0; peer < numRanks; peer++) {
    if (peer == globalRank) {
      continue;
    }
    EXPECT_EQ(h_states[peer].dataBufferSize, setupConfig.dataBufferSize)
        << "peer=" << peer;
    EXPECT_EQ(h_states[peer].chunkSize, setupConfig.chunkSize)
        << "peer=" << peer;
    EXPECT_EQ(h_states[peer].pipelineDepth, setupConfig.pipelineDepth)
        << "peer=" << peer;
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerIbgdaTransportSetupTestFixture, SignalSlotAssignment) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }

  std::unique_ptr<MultipeerIbgdaTransport> transport;
  try {
    transport = createTransport();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 1024,
      .chunkSize = 256,
      .pipelineDepth = 2,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  const auto& h_states = setup.getHostPeerStates();

  // Verify each peer gets distinct signal slot IDs
  std::set<int> signalIds;
  for (int peer = 0; peer < numRanks; peer++) {
    if (peer == globalRank) {
      continue;
    }
    int localSignalId = h_states[peer].localSignalId;
    EXPECT_TRUE(signalIds.find(localSignalId) == signalIds.end())
        << "Duplicate localSignalId " << localSignalId << " for peer " << peer;
    signalIds.insert(localSignalId);

    // Local signal ID should be peer * 2 (completion + back-pressure)
    constexpr int kExpectedSignalSlotsPerPeer = 2;
    EXPECT_EQ(localSignalId, peer * kExpectedSignalSlotsPerPeer)
        << "peer=" << peer;

    // Remote signal ID should be myRank * 2 (same for all peers)
    EXPECT_EQ(
        h_states[peer].remoteSignalId, globalRank * kExpectedSignalSlotsPerPeer)
        << "peer=" << peer;
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerIbgdaTransportSetupTestFixture, MultipleExchanges) {
  struct TestConfig {
    size_t dataBufferSize;
    size_t chunkSize;
    int pipelineDepth;
  };

  const std::vector<TestConfig> configs = {
      {1024, 256, 2},
      {4096, 1024, 4},
      {8192, 2048, 8},
  };

  for (size_t ci = 0; ci < configs.size(); ci++) {
    const auto& tc = configs[ci];

    std::unique_ptr<MultipeerIbgdaTransport> transport;
    try {
      transport = createTransport();
    } catch (const std::exception& e) {
      GTEST_SKIP() << "IBGDA transport not available: " << e.what();
    }

    MultiPeerIbgdaTransportSetupConfig setupConfig{
        .dataBufferSize = tc.dataBufferSize,
        .chunkSize = tc.chunkSize,
        .pipelineDepth = tc.pipelineDepth,
    };

    MultiPeerIbgdaTransportSetup setup(
        *transport, globalRank, numRanks, setupConfig);
    setup.exchangeBuffers();

    auto& hostPeerStates = setup.getHostPeerStates();
    EXPECT_EQ(hostPeerStates.size(), static_cast<size_t>(numRanks))
        << "config index=" << ci;

    auto* iterCounter = setup.getIterationCounter();
    EXPECT_NE(iterCounter, nullptr) << "config index=" << ci;

    // Verify config propagation from host-side peer states
    const auto& h_states = hostPeerStates;

    for (int peer = 0; peer < numRanks; peer++) {
      if (peer == globalRank) {
        continue;
      }
      EXPECT_EQ(h_states[peer].dataBufferSize, tc.dataBufferSize)
          << "config index=" << ci << " peer=" << peer;
      EXPECT_EQ(h_states[peer].chunkSize, tc.chunkSize)
          << "config index=" << ci << " peer=" << peer;
      EXPECT_EQ(h_states[peer].pipelineDepth, tc.pipelineDepth)
          << "config index=" << ci << " peer=" << peer;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
