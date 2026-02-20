// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstring>
#include <vector>

#include <unistd.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/NvmlFabricInfo.h"
#include "comms/pipes/Transport.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

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

  std::unique_ptr<MultiPeerTransport> create_transport_states() {
    MultiPeerTransportConfig config{
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
    return std::make_unique<MultiPeerTransport>(
        globalRank, numRanks, localRank, bootstrap, config);
  }

  bool isMnnvl_{false};
  int localSize_{0};
};

// MNNVL (GB200 NVL72):  all peers in the same fabric → NVL preferred, IBGDA
// universal. Non-MNNVL (H100 / standalone GB200): same-node → NVL preferred,
// cross-node → IBGDA preferred. On both platforms, IBGDA covers ALL non-self
// peers.
TEST_F(MultiPeerTransportMultiNodeFixture, TopologyDiscoveryMultiNode) {
  ASSERT_GE(numRanks, 4) << "Requires >= 4 ranks (nnodes=2, ppn=2)";

  auto states = create_transport_states();

  int nvlCount = states->nvl_peer_ranks().size();
  int ibgdaCount = states->ibgda_peer_ranks().size();

  // IBGDA is universal — always covers all non-self peers.
  EXPECT_EQ(ibgdaCount, numRanks - 1)
      << "IBGDA should cover all non-self peers regardless of platform";

  if (isMnnvl_) {
    // All ranks share the same NVLink fabric → every peer also has NVL.
    EXPECT_EQ(nvlCount, numRanks - 1) << "MNNVL: all peers should be NVL";
  } else {
    // Same-node peers use NVL, cross-node peers use IBGDA as preferred.
    EXPECT_EQ(nvlCount, localSize_ - 1)
        << "Non-MNNVL: NVL peers should be same-node only";
  }

  // Self should always be SELF.
  EXPECT_EQ(states->get_transport_type(globalRank), TransportType::SELF);

  // Invariant: NVL peers are a subset of IBGDA peers.
  EXPECT_LE(nvlCount, ibgdaCount);

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
  ASSERT_GE(numRanks, 4) << "Requires >= 4 ranks (nnodes=2, ppn=2)";

  auto states = create_transport_states();
  EXPECT_NO_THROW(states->exchange());

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify the device handle reflects the platform-specific topology.
//
// IBGDA transports are always populated for all peers.
// NVL transports are populated based on platform:
//   MNNVL: all peers,  Non-MNNVL: same-node peers only.
TEST_F(MultiPeerTransportMultiNodeFixture, DeviceHandleMultiNode) {
  ASSERT_GE(numRanks, 4) << "Requires >= 4 ranks (nnodes=2, ppn=2)";

  auto states = create_transport_states();
  states->exchange();

  auto handle = states->get_device_handle();
  EXPECT_EQ(handle.myRank, globalRank);
  EXPECT_EQ(handle.nRanks, numRanks);
  EXPECT_EQ(handle.transports.size(), static_cast<uint32_t>(numRanks));

  // NVL peers are always present (at minimum same-node peers).
  EXPECT_GT(handle.numNvlPeers, 0);

  if (isMnnvl_) {
    EXPECT_EQ(handle.numNvlPeers, numRanks - 1)
        << "MNNVL: all peers should be NVL";
  } else {
    EXPECT_EQ(handle.numNvlPeers, localSize_ - 1)
        << "Non-MNNVL: NVL peers should be same-node only";
  }

  // IBGDA is universal — all non-self peers.
  EXPECT_EQ(handle.numIbPeers, numRanks - 1)
      << "IBGDA transports should cover all peers";

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify host-side NVL and IBGDA accessors for each platform.
TEST_F(MultiPeerTransportMultiNodeFixture, HostAccessorsMultiNode) {
  ASSERT_GE(numRanks, 4) << "Requires >= 4 ranks (nnodes=2, ppn=2)";

  auto states = create_transport_states();
  states->exchange();

  // NVL peer accessor — always has at least same-node peers.
  ASSERT_FALSE(states->nvl_peer_ranks().empty());
  for (int r : states->nvl_peer_ranks()) {
    auto p2p = states->get_p2p_nvl_transport_device(r);
    EXPECT_NE(p2p.getLocalState().dataBuffer, nullptr)
        << "NVL local data buffer null for peer " << r;
    EXPECT_NE(p2p.getRemoteState().dataBuffer, nullptr)
        << "NVL remote data buffer null for peer " << r;
  }

  // IBGDA is universal — accessor works for ALL non-self peers.
  ASSERT_EQ(static_cast<int>(states->ibgda_peer_ranks().size()), numRanks - 1);
  for (int r : states->ibgda_peer_ranks()) {
    auto* p2p = states->get_p2p_ibgda_transport_device(r);
    EXPECT_NE(p2p, nullptr) << "IBGDA transport device null for peer " << r;
  }

  XLOGF(
      INFO,
      "Rank {}: isMnnvl={}, validated {} NVL peers, {} IBGDA peers",
      globalRank,
      isMnnvl_,
      states->nvl_peer_ranks().size(),
      states->ibgda_peer_ranks().size());

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
