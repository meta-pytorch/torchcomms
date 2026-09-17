// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// End-to-end distributed tests for TopologyDiscovery using real CUDA, NVML,
// gethostname, and MPI-based bootstrap. Requires GPU hardware and MPI.

#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include "comms/utils/logger/SpdlogLogger.h"

#include "comms/prims/topology/NvmlFabricInfo.h"
#include "comms/prims/topology/TopologyDiscovery.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::prims::tests {

class TopologyDiscoveryE2eFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
    detect_platform();
  }

  /**
   * Independently detect the platform by querying NvmlFabricInfo and
   * gathering hostnames from all ranks. This gives us ground truth to
   * verify that TopologyDiscovery made the correct decisions.
   */
  void detect_platform() {
    struct RankLocation {
      char hostname[64]{};
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

    expectedNvlPeerRanks_.clear();
    for (int r = 0; r < numRanks; ++r) {
      if (r == globalRank) {
        continue;
      }
      const bool sameHost =
          std::strcmp(myLoc.hostname, allLocs[r].hostname) == 0;
      const bool sameFabric = myLoc.fabricInfo.available &&
          allLocs[r].fabricInfo.available &&
          std::equal(myLoc.fabricInfo.clusterUuid,
                     myLoc.fabricInfo.clusterUuid + NvmlFabricInfo::kUuidLen,
                     allLocs[r].fabricInfo.clusterUuid) &&
          myLoc.fabricInfo.cliqueId == allLocs[r].fabricInfo.cliqueId;
      if (sameHost || sameFabric) {
        expectedNvlPeerRanks_.push_back(r);
      }
    }

    COMMS_LOG(
        INFO,
        "Rank {} platform detection: expected NVL peers={}",
        globalRank,
        expectedNvlPeerRanks_.size());
  }

  TopologyResult run_discover() {
    auto bootstrap = std::make_shared<MpiBootstrap>();
    TopologyDiscovery topo;
    return topo.discover(globalRank, numRanks, localRank, *bootstrap);
  }

  CanonicalTopologyResult run_canonical_discover() {
    auto bootstrap = std::make_shared<MpiBootstrap>();
    TopologyDiscovery topo;
    return topo.discoverCanonical(globalRank, numRanks, localRank, *bootstrap);
  }

  std::vector<int> expectedNvlPeerRanks_;
};

// NVL peers should be populated and self should NOT appear in nvlPeerRanks.
TEST_F(TopologyDiscoveryE2eFixture, BasicTopologyClassification) {
  auto result = run_discover();

  // Self should not be in the peer list.
  for (int peer : result.nvlPeerRanks) {
    EXPECT_NE(peer, globalRank) << "Self should not appear in nvlPeerRanks";
  }

  // Self should be in the global-to-NVL-local mapping.
  EXPECT_NE(
      result.globalToNvlLocal.find(globalRank), result.globalToNvlLocal.end())
      << "Self should be in globalToNvlLocal";

  // NVL domain size = peers + self.
  int nvlNRanks = static_cast<int>(result.nvlPeerRanks.size()) + 1;
  EXPECT_EQ(static_cast<int>(result.globalToNvlLocal.size()), nvlNRanks);

  COMMS_LOG(
      INFO,
      "Rank {}: {} NVL peers, fabricAvailable={}",
      globalRank,
      result.nvlPeerRanks.size(),
      result.fabricAvailable);

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify NVL local rank indices are consistent across all ranks that share
// the same NVL domain.
TEST_F(TopologyDiscoveryE2eFixture, NvlLocalRankConsistency) {
  auto result = run_discover();

  // Broadcast each rank's globalToNvlLocal mapping via allGather and verify
  // consistency: if rank A thinks rank B has NVL-local index X, then rank B
  // should agree on its own index.
  int myNvlLocal = result.globalToNvlLocal.at(globalRank);

  std::vector<int> allNvlLocals(numRanks);
  MPI_Allgather(
      &myNvlLocal, 1, MPI_INT, allNvlLocals.data(), 1, MPI_INT, MPI_COMM_WORLD);

  // For each peer in our NVL domain, verify their self-reported NVL-local
  // index matches what we assigned.
  for (const auto& [gRank, expectedLocal] : result.globalToNvlLocal) {
    EXPECT_EQ(allNvlLocals[gRank], expectedLocal)
        << "Rank " << globalRank << " thinks rank " << gRank
        << " has NVL-local " << expectedLocal << ", but rank " << gRank
        << " reports " << allNvlLocals[gRank];
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify NVL local indices form a dense [0, N) range.
TEST_F(TopologyDiscoveryE2eFixture, NvlLocalIndicesDense) {
  auto result = run_discover();

  int nvlNRanks = static_cast<int>(result.globalToNvlLocal.size());
  ASSERT_GT(nvlNRanks, 0);

  std::vector<bool> seen(nvlNRanks, false);
  for (const auto& [gRank, nvlLocal] : result.globalToNvlLocal) {
    ASSERT_GE(nvlLocal, 0) << "NVL local index out of range for rank " << gRank;
    ASSERT_LT(nvlLocal, nvlNRanks)
        << "NVL local index out of range for rank " << gRank;
    EXPECT_FALSE(seen[nvlLocal]) << "Duplicate NVL local index " << nvlLocal;
    seen[nvlLocal] = true;
  }

  for (int i = 0; i < nvlNRanks; ++i) {
    EXPECT_TRUE(seen[i]) << "Missing NVL local index " << i;
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify NVL peer count matches the union of same-host and same-fabric peers.
TEST_F(TopologyDiscoveryE2eFixture, PlatformNvlPeerCount) {
  auto result = run_discover();

  EXPECT_EQ(result.nvlPeerRanks, expectedNvlPeerRanks_);

  COMMS_LOG(
      INFO,
      "Rank {} (localRank {}): {} NVL peers (expected {})",
      globalRank,
      localRank,
      result.nvlPeerRanks.size(),
      expectedNvlPeerRanks_.size());

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(TopologyDiscoveryE2eFixture, CanonicalSnapshotMatchesLocalProjection) {
  const auto legacy = run_discover();
  const auto result = run_canonical_discover();
  std::vector<int> domainByRank(static_cast<std::size_t>(numRanks), -1);
  for (std::size_t domain = 0; domain < result.ranksByDomain.size(); ++domain) {
    for (int rank : result.ranksByDomain[domain]) {
      ASSERT_GE(rank, 0);
      ASSERT_LT(rank, numRanks);
      ASSERT_EQ(domainByRank[static_cast<std::size_t>(rank)], -1);
      domainByRank[static_cast<std::size_t>(rank)] = static_cast<int>(domain);
    }
  }
  for (int rank = 0; rank < numRanks; ++rank) {
    ASSERT_GE(domainByRank[static_cast<std::size_t>(rank)], 0);
  }

  std::vector<int> allDomainByRank(
      static_cast<std::size_t>(numRanks) * numRanks);
  MPI_Allgather(
      domainByRank.data(),
      numRanks,
      MPI_INT,
      allDomainByRank.data(),
      numRanks,
      MPI_INT,
      MPI_COMM_WORLD);
  for (int rank = 0; rank < numRanks; ++rank) {
    const auto first =
        allDomainByRank.begin() + static_cast<std::size_t>(rank) * numRanks;
    EXPECT_TRUE(std::equal(first, first + numRanks, domainByRank.begin()))
        << "rank " << rank << " derived a different canonical partition";
  }

  const auto& localDomain = result.ranksByDomain.at(
      static_cast<std::size_t>(domainByRank[globalRank]));
  ASSERT_EQ(result.mptTopology.globalToNvlLocal.size(), localDomain.size());
  for (std::size_t local = 0; local < localDomain.size(); ++local) {
    EXPECT_EQ(
        result.mptTopology.globalToNvlLocal.at(localDomain[local]),
        static_cast<int>(local));
  }
  EXPECT_EQ(result.mptTopology.nvlPeerRanks, legacy.nvlPeerRanks);
  EXPECT_EQ(result.mptTopology.globalToNvlLocal, legacy.globalToNvlLocal);
  EXPECT_EQ(result.mptTopology.nvlPeerRanks, expectedNvlPeerRanks_);

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
