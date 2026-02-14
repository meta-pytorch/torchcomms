// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultiPeerTransportStates.h"
#include "comms/pipes/Transport.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

class MultiPeerTransportStatesTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  std::unique_ptr<MultiPeerTransportStates> create_transport_states() {
    MultiPeerTransportStatesConfig config{
        .nvlConfig =
            {
                .dataBufferSize = 256 * 1024,
                .chunkSize = 512,
                .pipelineDepth = 4,
                .signalCount = 4,
            },
        .ibgdaConfig =
            {
                .cudaDevice = localRank,
                .signalCount = 4,
            },
    };
    auto bootstrap = std::make_shared<MpiBootstrap>();
    return std::make_unique<MultiPeerTransportStates>(
        globalRank, numRanks, bootstrap, config);
  }
};

// Verify that topology discovery correctly classifies peers as NVL or IBGDA.
// With nnodes=1, ppn=2, both ranks are on the same node so the peer should
// be NVL (assuming GPUs support P2P access).
TEST_F(MultiPeerTransportStatesTestFixture, TopologyDiscovery) {
  if (numRanks < 2) {
    XLOGF(WARNING, "Skipping: requires >= 2 ranks, got {}", numRanks);
    return;
  }

  auto states = create_transport_states();

  // On same node, peer should be NVL
  int peerRank = (globalRank == 0) ? 1 : 0;
  EXPECT_TRUE(states->is_nvl_peer(peerRank))
      << "Rank " << globalRank << " expected peer " << peerRank
      << " to be NVL (same node)";

  // Self should be SELF
  EXPECT_EQ(states->get_transport_type(globalRank), TransportType::SELF);
  EXPECT_EQ(states->get_transport_type(peerRank), TransportType::P2P_NVL);

  // Check peer rank vectors
  EXPECT_FALSE(states->nvl_peer_ranks().empty());
  // IBGDA is universal — it covers all non-self peers
  EXPECT_EQ(static_cast<int>(states->ibgda_peer_ranks().size()), numRanks - 1);

  XLOGF(
      INFO,
      "Rank {}: {} NVL peers, {} IBGDA peers",
      globalRank,
      states->nvl_peer_ranks().size(),
      states->ibgda_peer_ranks().size());

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify self transport type is always SELF regardless of rank count.
TEST_F(MultiPeerTransportStatesTestFixture, SelfTransportType) {
  auto states = create_transport_states();
  EXPECT_EQ(states->get_transport_type(globalRank), TransportType::SELF);
  EXPECT_EQ(states->my_rank(), globalRank);
  EXPECT_EQ(states->n_ranks(), numRanks);

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify exchange() completes without error on all ranks.
TEST_F(MultiPeerTransportStatesTestFixture, ExchangeSucceeds) {
  if (numRanks < 2) {
    XLOGF(WARNING, "Skipping: requires >= 2 ranks, got {}", numRanks);
    return;
  }

  auto states = create_transport_states();
  EXPECT_NO_THROW(states->exchange());

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify host-side NVL transport accessor returns valid objects after exchange.
TEST_F(MultiPeerTransportStatesTestFixture, HostNvlAccessor) {
  if (numRanks < 2) {
    XLOGF(WARNING, "Skipping: requires >= 2 ranks, got {}", numRanks);
    return;
  }

  auto states = create_transport_states();
  states->exchange();

  int peerRank = (globalRank == 0) ? 1 : 0;
  auto p2p = states->get_p2p_nvl_transport_device(peerRank);

  // The returned device should have valid local/remote state pointers
  EXPECT_NE(p2p.getLocalState().dataBuffer, nullptr)
      << "Rank " << globalRank << ": NVL local data buffer is null";
  EXPECT_NE(p2p.getRemoteState().dataBuffer, nullptr)
      << "Rank " << globalRank << ": NVL remote data buffer is null";

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify the self transport accessor returns a valid (trivial) object.
TEST_F(MultiPeerTransportStatesTestFixture, SelfAccessor) {
  auto states = create_transport_states();
  auto selfTransport = states->get_p2p_self_transport_device();
  // P2pSelfTransportDevice is stateless, just verify it constructs
  (void)selfTransport;

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify getDeviceHandle() returns a handle with correct metadata
// after exchange.
TEST_F(MultiPeerTransportStatesTestFixture, DeviceHandleMetadata) {
  if (numRanks < 2) {
    XLOGF(WARNING, "Skipping: requires >= 2 ranks, got {}", numRanks);
    return;
  }

  auto states = create_transport_states();
  states->exchange();

  auto handle = states->get_device_handle();
  EXPECT_EQ(handle.myRank, globalRank);
  EXPECT_EQ(handle.nRanks, numRanks);

  // Type per rank array should be non-empty
  EXPECT_FALSE(handle.typePerRank.empty());
  EXPECT_EQ(handle.typePerRank.size(), static_cast<uint32_t>(numRanks));

  // On single-node with NVL peers, nvlTransports should be non-empty
  EXPECT_FALSE(handle.nvlTransports.empty());

  // Global-to-NVL index array should be allocated
  EXPECT_FALSE(handle.globalToNvlIndex.empty());
  EXPECT_EQ(handle.globalToNvlIndex.size(), static_cast<uint32_t>(numRanks));

  // IBGDA is universal — always populated for all peers
  EXPECT_FALSE(handle.ibgdaTransports.empty());
  EXPECT_EQ(handle.ibgdaTransports.size(), static_cast<uint32_t>(numRanks - 1));

  // Global-to-IBGDA index array should be allocated
  EXPECT_FALSE(handle.globalToIbgdaIndex.empty());
  EXPECT_EQ(handle.globalToIbgdaIndex.size(), static_cast<uint32_t>(numRanks));

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify getDeviceHandle() throws before exchange() is called.
TEST_F(MultiPeerTransportStatesTestFixture, DeviceHandleBeforeExchange) {
  auto states = create_transport_states();
  EXPECT_THROW(states->get_device_handle(), std::runtime_error);

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify that IBGDA transport is accessible even for an NVL peer.
// This is the key capability: IBGDA is universal, NVL is the preferred overlay.
TEST_F(MultiPeerTransportStatesTestFixture, HostIbgdaAccessorForNvlPeer) {
  if (numRanks < 2) {
    XLOGF(WARNING, "Skipping: requires >= 2 ranks, got {}", numRanks);
    return;
  }

  auto states = create_transport_states();
  states->exchange();

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Peer is NVL, but IBGDA should also be accessible
  ASSERT_TRUE(states->is_nvl_peer(peerRank));
  EXPECT_TRUE(states->has_ibgda(peerRank));

  auto* ibgdaDev = states->get_p2p_ibgda_transport_device(peerRank);
  EXPECT_NE(ibgdaDev, nullptr)
      << "IBGDA transport should be accessible for NVL peer " << peerRank;

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
