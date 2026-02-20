// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/bootstrap/NvlBootstrapAdapter.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using comms::pipes::NvlBootstrapAdapter;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::tests {

class NvlBootstrapAdapterTest : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    bootstrap_ = std::make_shared<MpiBootstrap>();
  }

  std::shared_ptr<MpiBootstrap> bootstrap_;
};

// =============================================================================
// IBootstrap NvlDomain method tests (on MpiBootstrap directly)
// =============================================================================

/**
 * Test allGatherNvlDomain on MpiBootstrap with all ranks in the NVL domain.
 *
 * Each rank writes its globalRank into its slot, then allGatherNvlDomain
 * exchanges data among all ranks. Verifies every rank received correct values.
 */
TEST_F(NvlBootstrapAdapterTest, AllGatherNvlDomainAllRanks) {
  std::vector<int> nvlRankToCommRank(numRanks);
  for (int i = 0; i < numRanks; ++i) {
    nvlRankToCommRank[i] = i;
  }

  std::vector<int> buf(numRanks, -1);
  buf[globalRank] = globalRank;

  auto rc =
      bootstrap_
          ->allGatherNvlDomain(
              buf.data(), sizeof(int), globalRank, numRanks, nvlRankToCommRank)
          .get();
  ASSERT_EQ(rc, 0);

  for (int i = 0; i < numRanks; ++i) {
    EXPECT_EQ(buf[i], i) << "Mismatch at index " << i;
  }
}

/**
 * Test allGatherNvlDomain with a subset of ranks (simulating MNNVL where
 * only even-ranked GPUs are in the same NVLink domain).
 */
TEST_F(NvlBootstrapAdapterTest, AllGatherNvlDomainSubset) {
  if (numRanks < 4) {
    XLOGF(
        WARNING, "Skipping subset test: requires >= 4 ranks, got {}", numRanks);
    return;
  }

  // NVL domain = even-ranked GPUs: {0, 2, 4, ...}
  std::vector<int> nvlRankToCommRank;
  for (int i = 0; i < numRanks; i += 2) {
    nvlRankToCommRank.push_back(i);
  }
  int nvlNranks = static_cast<int>(nvlRankToCommRank.size());

  int nvlLocalRank = -1;
  for (int i = 0; i < nvlNranks; ++i) {
    if (nvlRankToCommRank[i] == globalRank) {
      nvlLocalRank = i;
      break;
    }
  }

  if (nvlLocalRank < 0) {
    XLOGF(INFO, "Rank {} not in NVL domain, skipping", globalRank);
    return;
  }

  std::vector<int> buf(nvlNranks, -1);
  buf[nvlLocalRank] = globalRank + 100;

  auto rc = bootstrap_
                ->allGatherNvlDomain(
                    buf.data(),
                    sizeof(int),
                    nvlLocalRank,
                    nvlNranks,
                    nvlRankToCommRank)
                .get();
  ASSERT_EQ(rc, 0);

  for (int i = 0; i < nvlNranks; ++i) {
    EXPECT_EQ(buf[i], nvlRankToCommRank[i] + 100)
        << "Mismatch at NVL index " << i << " (global rank "
        << nvlRankToCommRank[i] << ")";
  }
}

/**
 * Test barrierNvlDomain on MpiBootstrap — all ranks participate
 * and the barrier completes without deadlock.
 */
TEST_F(NvlBootstrapAdapterTest, BarrierNvlDomainAllRanks) {
  std::vector<int> nvlRankToCommRank(numRanks);
  for (int i = 0; i < numRanks; ++i) {
    nvlRankToCommRank[i] = i;
  }

  auto rc =
      bootstrap_->barrierNvlDomain(globalRank, numRanks, nvlRankToCommRank)
          .get();
  ASSERT_EQ(rc, 0);
}

// =============================================================================
// NvlBootstrapAdapter wrapper tests
// =============================================================================

/**
 * Test NvlBootstrapAdapter end-to-end: adapter.allGather() routes through
 * allGatherNvlDomain on the underlying MpiBootstrap.
 */
TEST_F(NvlBootstrapAdapterTest, AdapterAllGatherEndToEnd) {
  std::vector<int> nvlRankToCommRank(numRanks);
  for (int i = 0; i < numRanks; ++i) {
    nvlRankToCommRank[i] = i;
  }

  NvlBootstrapAdapter adapter(bootstrap_, nvlRankToCommRank);

  std::vector<int> buf(numRanks, -1);
  buf[globalRank] = globalRank * 10 + 7;

  auto rc =
      adapter.allGather(buf.data(), sizeof(int), globalRank, numRanks).get();
  ASSERT_EQ(rc, 0);

  for (int i = 0; i < numRanks; ++i) {
    EXPECT_EQ(buf[i], i * 10 + 7) << "Mismatch at index " << i;
  }
}

/**
 * Test that adapter.barrier() routes through barrierNvlDomain and completes
 * without deadlock.
 */
TEST_F(NvlBootstrapAdapterTest, AdapterBarrierRouting) {
  std::vector<int> nvlRankToCommRank(numRanks);
  for (int i = 0; i < numRanks; ++i) {
    nvlRankToCommRank[i] = i;
  }

  NvlBootstrapAdapter adapter(bootstrap_, nvlRankToCommRank);

  auto rc = adapter.barrier(globalRank, numRanks).get();
  ASSERT_EQ(rc, 0);
}

/**
 * Test that adapter.send()/recv() correctly translates NVL-local peer indices
 * to global communicator ranks via localRankToCommRank mapping.
 *
 * Uses a reversed rank mapping: NVL local index i maps to global rank
 * (numRanks - 1 - i). Rank 0 sends to NVL-local peer (numRanks-1), which
 * maps to global rank 0. Rank (numRanks-1) receives from NVL-local peer 0,
 * which also maps to global rank 0. This verifies the mapping is applied.
 */
TEST_F(NvlBootstrapAdapterTest, AdapterSendRecvMapping) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  // Reversed mapping: NVL local index i → global rank (numRanks - 1 - i)
  std::vector<int> reversedMapping(numRanks);
  for (int i = 0; i < numRanks; ++i) {
    reversedMapping[i] = numRanks - 1 - i;
  }

  // Find this rank's NVL-local index in the reversed mapping
  int myNvlLocal = numRanks - 1 - globalRank;

  NvlBootstrapAdapter adapter(bootstrap_, reversedMapping);

  constexpr int kTag = 42;
  int sendVal = globalRank * 100 + 1;
  int recvVal = -1;

  // Each rank sends to NVL-local peer 0 and receives from NVL-local peer 0
  // NVL-local 0 maps to global rank (numRanks - 1)
  if (myNvlLocal != 0) {
    // Send to NVL-local peer 0 (which is global rank numRanks-1)
    auto rc = adapter.send(&sendVal, sizeof(int), 0, kTag + myNvlLocal).get();
    ASSERT_EQ(rc, 0);
  } else {
    // NVL-local rank 0 (= global rank numRanks-1) receives from all others
    for (int nvlPeer = 1; nvlPeer < numRanks; ++nvlPeer) {
      auto rc =
          adapter.recv(&recvVal, sizeof(int), nvlPeer, kTag + nvlPeer).get();
      ASSERT_EQ(rc, 0);
      // nvlPeer maps to global rank (numRanks - 1 - nvlPeer)
      int expectedSender = numRanks - 1 - nvlPeer;
      EXPECT_EQ(recvVal, expectedSender * 100 + 1)
          << "Expected value from global rank " << expectedSender
          << " (NVL peer " << nvlPeer << ")";
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Test adapter with a non-identity subset mapping.
 * Creates NVL domain with only odd-ranked GPUs: {1, 3, 5, ...}
 * and verifies allGather works correctly through the adapter.
 */
TEST_F(NvlBootstrapAdapterTest, AdapterSubsetMapping) {
  if (numRanks < 4) {
    XLOGF(
        WARNING,
        "Skipping subset mapping test: requires >= 4 ranks, got {}",
        numRanks);
    return;
  }

  // NVL domain = odd-ranked GPUs: {1, 3, 5, ...}
  std::vector<int> nvlRankToCommRank;
  for (int i = 1; i < numRanks; i += 2) {
    nvlRankToCommRank.push_back(i);
  }
  int nvlNranks = static_cast<int>(nvlRankToCommRank.size());

  int nvlLocalRank = -1;
  for (int i = 0; i < nvlNranks; ++i) {
    if (nvlRankToCommRank[i] == globalRank) {
      nvlLocalRank = i;
      break;
    }
  }

  if (nvlLocalRank < 0) {
    XLOGF(INFO, "Rank {} not in odd NVL domain, skipping", globalRank);
    return;
  }

  NvlBootstrapAdapter adapter(bootstrap_, nvlRankToCommRank);

  std::vector<int> buf(nvlNranks, -1);
  buf[nvlLocalRank] = globalRank + 200;

  auto rc =
      adapter.allGather(buf.data(), sizeof(int), nvlLocalRank, nvlNranks).get();
  ASSERT_EQ(rc, 0);

  for (int i = 0; i < nvlNranks; ++i) {
    EXPECT_EQ(buf[i], nvlRankToCommRank[i] + 200)
        << "Mismatch at NVL index " << i << " (global rank "
        << nvlRankToCommRank[i] << ")";
  }
}

} // namespace comms::pipes::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
