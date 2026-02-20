// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstring>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/NvmlFabricInfo.h"
#include "comms/pipes/TopologyDiscovery.h"
#include "comms/pipes/Transport.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

class TopologyDiscoveryFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
    detectPlatform();
  }

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

    localSize_ = 0;
    for (int r = 0; r < numRanks; ++r) {
      if (std::strcmp(myLoc.hostname, allLocs[r].hostname) == 0) {
        ++localSize_;
      }
    }

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
  }

  bool isMnnvl_{false};
  int localSize_{0};
};

// Verify basic topology classification: NVL peers populated, self not in
// nvlPeerRanks, globalToNvlLocal contains self.
TEST_F(TopologyDiscoveryFixture, BasicTopologyClassification) {
  auto bootstrap = std::make_shared<MpiBootstrap>();
  auto topo =
      TopologyDiscovery::discover(globalRank, numRanks, localRank, bootstrap);

  // Self should be in the NVL local mapping but NOT in nvlPeerRanks.
  EXPECT_NE(
      topo.globalToNvlLocal.find(globalRank), topo.globalToNvlLocal.end());
  for (int r : topo.nvlPeerRanks) {
    EXPECT_NE(r, globalRank) << "Self should not appear in nvlPeerRanks";
  }

  // On same-node with >=2 GPUs, at least one peer should be NVL.
  if (numRanks >= 2 && localSize_ >= 2) {
    EXPECT_FALSE(topo.nvlPeerRanks.empty())
        << "Expected NVL peers on same node";
  }

  XLOGF(
      INFO,
      "Rank {}: {} NVL peers, isMnnvl={}",
      globalRank,
      topo.nvlPeerRanks.size(),
      isMnnvl_);

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify NVL local rank indices are consistent across all ranks.
TEST_F(TopologyDiscoveryFixture, NvlLocalRankConsistency) {
  auto bootstrap = std::make_shared<MpiBootstrap>();
  auto topo =
      TopologyDiscovery::discover(globalRank, numRanks, localRank, bootstrap);

  int nvlNRanks = static_cast<int>(topo.nvlPeerRanks.size()) + 1;
  int nvlLocalRank = topo.globalToNvlLocal.at(globalRank);

  // nvlLocalRank must be in [0, nvlNRanks)
  EXPECT_GE(nvlLocalRank, 0);
  EXPECT_LT(nvlLocalRank, nvlNRanks);

  // globalToNvlLocal must contain self
  auto it = topo.globalToNvlLocal.find(globalRank);
  ASSERT_NE(it, topo.globalToNvlLocal.end());
  EXPECT_EQ(it->second, nvlLocalRank);

  // All NVL peers should be in the mapping
  for (int r : topo.nvlPeerRanks) {
    EXPECT_NE(topo.globalToNvlLocal.find(r), topo.globalToNvlLocal.end())
        << "NVL peer " << r << " missing from globalToNvlLocal";
  }

  // Mapping size = nvlPeerRanks + self
  EXPECT_EQ(
      static_cast<int>(topo.globalToNvlLocal.size()),
      static_cast<int>(topo.nvlPeerRanks.size()) + 1);

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify NVL local indices form a dense [0, N) range.
TEST_F(TopologyDiscoveryFixture, NvlLocalIndicesDense) {
  auto bootstrap = std::make_shared<MpiBootstrap>();
  auto topo =
      TopologyDiscovery::discover(globalRank, numRanks, localRank, bootstrap);

  int nvlNRanks = static_cast<int>(topo.globalToNvlLocal.size());
  std::vector<bool> seen(nvlNRanks, false);

  for (const auto& [gRank, nvlLocal] : topo.globalToNvlLocal) {
    ASSERT_GE(nvlLocal, 0);
    ASSERT_LT(nvlLocal, nvlNRanks);
    EXPECT_FALSE(seen[nvlLocal]) << "Duplicate NVL local index " << nvlLocal;
    seen[nvlLocal] = true;
  }

  for (int i = 0; i < nvlNRanks; ++i) {
    EXPECT_TRUE(seen[i]) << "Missing NVL local index " << i;
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify NVL peer count matches platform expectations.
TEST_F(TopologyDiscoveryFixture, PlatformNvlPeerCount) {
  auto bootstrap = std::make_shared<MpiBootstrap>();
  auto topo =
      TopologyDiscovery::discover(globalRank, numRanks, localRank, bootstrap);

  if (isMnnvl_) {
    EXPECT_EQ(static_cast<int>(topo.nvlPeerRanks.size()), numRanks - 1)
        << "MNNVL: all peers should be NVL";
  } else {
    EXPECT_EQ(static_cast<int>(topo.nvlPeerRanks.size()), localSize_ - 1)
        << "Non-MNNVL: NVL peers should be same-node only";
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
