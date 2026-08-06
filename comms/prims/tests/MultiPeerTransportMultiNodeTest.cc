// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <algorithm>
#include <cstring>
#include <vector>

#include <unistd.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/prims/topology/NvmlFabricInfo.h"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/prims/transport/Transport.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using namespace meta::comms;

namespace comms::prims::tests {

/**
 * Multi-node test fixture for MultiPeerTransport (nnodes=2, ppn=2).
 *
 * Ranks span two hosts, creating a mixed topology: same-node peers are
 * NVLink-connected while cross-node peers fall back to IBGDA. This
 * fixture independently detects the platform (MNNVL vs H100) to verify
 * that MultiPeerTransport's topology discovery makes correct decisions.
 *
 * For single-node tests in a homogeneous NVL-only environment, see
 * MultiPeerTransportTest.cc.
 */
class MultiPeerTransportMultiNodeFixture : public MpiBaseTestFixture {
 protected:
  static constexpr int kIbgdaMaxGroups = 17;

  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
    detectPlatform();
  }

  /**
   * Independently detect the platform by querying NvmlFabricInfo and
   * gathering hostnames from all ranks.  This gives us ground truth to
   * verify that MultiPeerTransport made the correct topology
   * decisions.
   */
  void detectPlatform() {
    struct RankLocation {
      char hostname[64];
      NvmlFabricInfo fabricInfo;
    };

    RankLocation myLoc{};
    gethostname(myLoc.hostname, sizeof(myLoc.hostname));

    char busId[NvmlFabricInfo::kBusIdLen];
    CUDACHECK_TEST(
        cudaDeviceGetPCIBusId(busId, NvmlFabricInfo::kBusIdLen, localRank));
    myLoc.fabricInfo = NvmlFabricInfo::query(busId);

    std::vector<RankLocation> allLocs(numRanks);
    MPI_Allgather(
        &myLoc,
        sizeof(RankLocation),
        MPI_BYTE,
        allLocs.data(),
        sizeof(RankLocation),
        MPI_BYTE,
        MPI_COMM_WORLD);

    // Count same-hostname ranks (= local node size).
    localSize_ = 0;
    for (int r = 0; r < numRanks; ++r) {
      if (std::strcmp(myLoc.hostname, allLocs[r].hostname) == 0) {
        ++localSize_;
      }
    }

    // Check if ALL ranks share the same MNNVL fabric.
    isMnnvl_ = myLoc.fabricInfo.available;
    if (isMnnvl_) {
      for (int r = 0; r < numRanks; ++r) {
        if (!allLocs[r].fabricInfo.available ||
            std::memcmp(
                myLoc.fabricInfo.clusterUuid,
                allLocs[r].fabricInfo.clusterUuid,
                NvmlFabricInfo::kUuidLen) != 0 ||
            myLoc.fabricInfo.cliqueId != allLocs[r].fabricInfo.cliqueId) {
          isMnnvl_ = false;
          break;
        }
      }
    }

    XLOGF(
        INFO,
        "Rank {} platform detection: isMnnvl={}, localSize={}",
        globalRank,
        isMnnvl_,
        localSize_);
  }

  std::unique_ptr<MultiPeerTransport> create_transport_states(
      bool p2pDisable = false) {
    MultiPeerTransportConfig config{
        .nvlConfig =
            {
                .pipelineDepth = 4,
                .p2pSignalCount = 4,
                .maxNumChannels = 64,
                .perChannelSize = 4 * 1024,
            },
        .ibConfig =
            {
                .cudaDevice = localRank,
                .max_num_channels = kIbgdaMaxGroups,
            },
        .topoConfig =
            {
                .p2pDisable = p2pDisable,
            },
    };
    auto bootstrap = std::make_shared<MpiBootstrap>();
    return std::make_unique<MultiPeerTransport>(
        globalRank, numRanks, localRank, bootstrap, config);
  }

  bool isMnnvl_{false};
  int localSize_{0};
};

// MNNVL (GB200 NVL72): all peers in the same fabric, so NVL is preferred.
// Non-MNNVL (H100 / standalone GB200): same-node peers prefer NVL and
// cross-node peers prefer IBGDA.
TEST_F(MultiPeerTransportMultiNodeFixture, TopologyDiscoveryMultiNode) {
  if (numRanks < 4) {
    GTEST_SKIP() << "Requires >= 4 ranks (nnodes=2, ppn=2)";
  }

  auto states = create_transport_states();

  int nvlCount = states->nvl_peer_ranks().size();
  int ibgdaCount = states->ib_peer_ranks().size();

  if (isMnnvl_) {
    // All ranks share the same NVLink fabric, so no preferred-IB peers exist.
    EXPECT_EQ(nvlCount, numRanks - 1) << "MNNVL: all peers should be NVL";
    EXPECT_EQ(ibgdaCount, 0) << "MNNVL: no peers should prefer IBGDA";
  } else {
    EXPECT_EQ(nvlCount, localSize_ - 1)
        << "Non-MNNVL: NVL peers should be same-node only";
    EXPECT_EQ(ibgdaCount, numRanks - localSize_)
        << "Non-MNNVL: IBGDA peers should be cross-node only";
  }

  // Self should always be SELF.
  EXPECT_EQ(states->get_transport_type(globalRank), TransportType::SELF);

  EXPECT_EQ(nvlCount + ibgdaCount, numRanks - 1);

  XLOGF(
      INFO,
      "Rank {} (localRank {}): isMnnvl={}, {} NVL peers, {} IBGDA peers",
      globalRank,
      localRank,
      isMnnvl_,
      nvlCount,
      ibgdaCount);

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify that exchange() completes on both platforms.
TEST_F(MultiPeerTransportMultiNodeFixture, ExchangeMultiNode) {
  if (numRanks < 4) {
    GTEST_SKIP() << "Requires >= 4 ranks (nnodes=2, ppn=2)";
  }

  auto states = create_transport_states();
  EXPECT_NO_THROW(states->exchange());

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify the device handle reflects the platform-specific topology.
//
// Preferred transport counts vary by platform:
//   MNNVL: all NVL, Non-MNNVL: same-node NVL and cross-node IBGDA.
TEST_F(MultiPeerTransportMultiNodeFixture, DeviceHandleMultiNode) {
  if (numRanks < 4) {
    GTEST_SKIP() << "Requires >= 4 ranks (nnodes=2, ppn=2)";
  }

  auto states = create_transport_states();
  states->exchange();

  std::vector<PeerChannelDemand> demands;
  for (const int peer : states->ib_peer_ranks()) {
    demands.push_back({
        .peerRank = peer,
        .ibChannels = states->ib_channel_capacity(),
    });
  }
  auto handle = states->get_device_handle(demands);
  EXPECT_EQ(handle.myRank, globalRank);
  EXPECT_EQ(handle.nRanks, numRanks);
  EXPECT_EQ(handle.transports.size(), static_cast<uint32_t>(numRanks));

  // NVL peers should be present (at minimum same-node peers).
  EXPECT_GT(handle.numNvlPeers, 0);

  if (isMnnvl_) {
    EXPECT_EQ(handle.numNvlPeers, numRanks - 1)
        << "MNNVL: all peers should be NVL";
  } else {
    EXPECT_EQ(handle.numNvlPeers, localSize_ - 1)
        << "Non-MNNVL: NVL peers should be same-node only";
  }

  const int expectedIbPeers = isMnnvl_ ? 0 : numRanks - localSize_;
  EXPECT_EQ(handle.numIbPeers, expectedIbPeers);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MultiPeerTransportMultiNodeFixture, DeviceHandleAcrossPeerRounds) {
  if (numRanks < 4) {
    GTEST_SKIP() << "Requires >= 4 ranks (nnodes=2, ppn=2)";
  }

  auto states = create_transport_states(/*p2pDisable=*/true);
  states->exchange();

  const auto& ibPeers = states->ib_peer_ranks();
  ASSERT_EQ(ibPeers.size(), numRanks - 1);
  auto addIbPeer = [&ibPeers](std::vector<int>& peers, int peer) {
    if (std::find(ibPeers.begin(), ibPeers.end(), peer) != ibPeers.end() &&
        std::find(peers.begin(), peers.end(), peer) == peers.end()) {
      peers.push_back(peer);
    }
  };

  std::vector<int> ringPeers;
  addIbPeer(ringPeers, (globalRank + numRanks - 1) % numRanks);
  addIbPeer(ringPeers, (globalRank + 1) % numRanks);
  std::vector<PeerChannelDemand> ringDemands;
  ringDemands.reserve(ringPeers.size());
  for (const int peer : ringPeers) {
    ringDemands.push_back({
        .peerRank = peer,
        .ibChannels = states->ib_channel_capacity(),
    });
  }
  auto ringHandle = states->get_device_handle(ringDemands);

  std::vector<int> treePeers;
  if (globalRank > 0) {
    addIbPeer(treePeers, (globalRank - 1) / 2);
  }
  addIbPeer(treePeers, globalRank * 2 + 1);
  addIbPeer(treePeers, globalRank * 2 + 2);
  std::reverse(treePeers.begin(), treePeers.end());
  std::vector<PeerChannelDemand> treeDemands;
  treeDemands.reserve(treePeers.size());
  for (const int peer : treePeers) {
    treeDemands.push_back({
        .peerRank = peer,
        .ibChannels = states->ib_channel_capacity(),
    });
  }
  auto handle = states->get_device_handle(treeDemands);

  EXPECT_EQ(ringHandle.transports.data(), handle.transports.data());
  EXPECT_EQ(handle.myRank, globalRank);
  EXPECT_EQ(handle.nRanks, numRanks);
  EXPECT_EQ(handle.transports.size(), static_cast<uint32_t>(numRanks));
  EXPECT_EQ(handle.numIbPeers, numRanks - 1)
      << "IBGDA transports should cover all peers";

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify host-side NVL and IBGDA accessors for each platform.
TEST_F(MultiPeerTransportMultiNodeFixture, HostAccessorsMultiNode) {
  if (numRanks < 4) {
    GTEST_SKIP() << "Requires >= 4 ranks (nnodes=2, ppn=2)";
  }

  auto states = create_transport_states();
  EXPECT_EQ(states->ib_channel_capacity(), kIbgdaMaxGroups);
  states->exchange();

  // NVL peer accessor — always has at least same-node peers.
  // The returned pointers point to device memory inside the Transport array.
  // We can only verify they're non-null here; device-side tests verify
  // functionality.
  ASSERT_FALSE(states->nvl_peer_ranks().empty());
  for (int r : states->nvl_peer_ranks()) {
    auto p2p = states->get_p2p_nvl_transport_device(r);
    // Verify we can construct a device handle without throwing
    (void)p2p;
  }

  const int probePeer = (globalRank + 1) % numRanks;
  if (!states->has_ibgda(probePeer)) {
    GTEST_SKIP() << "Communicator has no underlying IBGDA transport";
  }

  // Once constructed, the underlying IBGDA transport can serve every
  // non-self peer, including NVL-preferred peers.
  std::vector<PeerChannelDemand> demands;
  demands.reserve(numRanks - 1);
  for (int peer = 0; peer < numRanks; ++peer) {
    if (peer == globalRank) {
      continue;
    }
    EXPECT_THROW(
        states->get_p2p_ibgda_transport_device(peer), std::runtime_error);
    demands.push_back({
        .peerRank = peer,
        .ibChannels = states->ib_channel_capacity(),
    });
  }
  (void)states->get_device_handle(demands);
  for (const auto& demand : demands) {
    auto* p2p = states->get_p2p_ibgda_transport_device(demand.peerRank);
    EXPECT_NE(p2p, nullptr)
        << "IBGDA transport device null for peer " << demand.peerRank;
  }

  XLOGF(
      INFO,
      "Rank {}: isMnnvl={}, validated {} NVL peers, {} underlying IBGDA peers",
      globalRank,
      isMnnvl_,
      states->nvl_peer_ranks().size(),
      demands.size());

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
