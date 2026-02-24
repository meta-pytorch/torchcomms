// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// End-to-end distributed tests for TopologyDiscovery using real CUDA, NVML,
// gethostname, and MPI-based bootstrap. Requires GPU hardware and MPI.

#include <unistd.h>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "comms/pipes/NvmlFabricInfo.h"
#include "comms/pipes/TopologyDiscovery.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::tests {

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

  TopologyResult run_discover() {
    auto bootstrap = std::make_shared<MpiBootstrap>();
    TopologyDiscovery topo;
    return topo.discover(globalRank, numRanks, localRank, *bootstrap);
  }

  bool isMnnvl_{false};
  int localSize_{0};
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

  XLOGF(
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

// Verify NVL peer count matches the platform: on MNNVL all ranks in the
// same clique are NVL peers, on non-MNNVL only same-host ranks are NVL peers.
TEST_F(TopologyDiscoveryE2eFixture, PlatformNvlPeerCount) {
  auto result = run_discover();

  int nvlPeerCount = static_cast<int>(result.nvlPeerRanks.size());

  if (isMnnvl_) {
    // All ranks share the same NVLink fabric → every peer is NVL.
    EXPECT_EQ(nvlPeerCount, numRanks - 1) << "MNNVL: all peers should be NVL";
  } else {
    // Same-host peers only.
    EXPECT_EQ(nvlPeerCount, localSize_ - 1)
        << "Non-MNNVL: NVL peers should be same-host only";
  }

  XLOGF(
      INFO,
      "Rank {} (localRank {}): isMnnvl={}, {} NVL peers (expected {})",
      globalRank,
      localRank,
      isMnnvl_,
      nvlPeerCount,
      isMnnvl_ ? numRanks - 1 : localSize_ - 1);

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
