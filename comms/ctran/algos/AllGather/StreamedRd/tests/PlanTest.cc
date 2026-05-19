// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include "comms/ctran/algos/AllGather/StreamedRd/Plan.h"

namespace ctran::allgather::ctsrd {
namespace {

int log2i(const int n) {
  int r = 0;
  while ((int(1) << r) < n) {
    r++;
  }
  return r;
}

bool verbose() {
  static const bool v = std::getenv("CTSRD_PRINT_PLAN") != nullptr;
  return v;
}

void printPlan(
    const char* label,
    const int rank,
    const int nRanks,
    const Plan& plan) {
  if (!verbose()) {
    return;
  }
  std::cout << label << " rank=" << rank << " nRanks=" << nRanks << "\n";
  for (int step = 0; step < plan.nSteps(); step++) {
    std::cout << "  step " << step << ": [";
    const auto& c = plan.chunks(step);
    for (int k = 0; k < c.size(); k++) {
      if (k > 0) {
        std::cout << ", ";
      }
      std::cout << c[k];
    }
    std::cout << "]\n";
  }
}

int peerAt(const int rank, const int step, const int nRanks) {
  const auto dist = nRanks / (2 << step);
  const auto pos = (rank / dist) % 2;
  return pos == 0 ? rank + dist : rank - dist;
}

class PlanTest : public ::testing::TestWithParam<std::tuple<int, FcMode>> {};

TEST_P(PlanTest, CorrectSize) {
  const auto [nRanks, fcMode] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fcMode);
    EXPECT_EQ(plan.nSteps(), nSteps);
    for (int step = 0; step < nSteps; step++) {
      EXPECT_EQ(plan.chunks(step).size(), 1 << step)
          << "rank=" << rank << " step=" << step << " nRanks=" << nRanks;
    }
  }
}

TEST_P(PlanTest, UniquePositions) {
  const auto [nRanks, fcMode] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fcMode);
    for (int step = 0; step < nSteps; step++) {
      const auto& order = plan.chunks(step);
      const std::set<int> seen(order.begin(), order.end());
      EXPECT_EQ(seen.size(), order.size())
          << "rank=" << rank << " step=" << step;
    }
  }
}

TEST_P(PlanTest, CorrectPositionSet) {
  const auto [nRanks, fcMode] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fcMode);
    for (int step = 0; step < nSteps; step++) {
      const auto& order = plan.chunks(step);
      const auto sender = peerAt(rank, step, nRanks);
      const auto stride = nRanks >> step;
      const auto peerOffset = sender % stride;

      std::set<int> expected;
      for (int k = 0; k < (1 << step); k++) {
        expected.insert(k * stride + peerOffset);
      }
      const std::set<int> actual(order.begin(), order.end());
      EXPECT_EQ(actual, expected) << "rank=" << rank << " step=" << step;
    }
  }
}

TEST_P(PlanTest, FirstElementIsSenderOwnChunk) {
  const auto [nRanks, fcMode] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fcMode);
    for (int step = 0; step < nSteps; step++) {
      const auto sender = peerAt(rank, step, nRanks);
      EXPECT_EQ(plan.chunk(step, 0), sender)
          << "rank=" << rank << " step=" << step;
    }
  }
}

TEST_P(PlanTest, BothModesProduceSameSet) {
  const auto [nRanks, fcMode] = GetParam();
  if (fcMode != FcMode::kRecvOnly) {
    return;
  }
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto planA = createRecvPlan(rank, nRanks, FcMode::kRecvOnly);
    const auto planB = createRecvPlan(rank, nRanks, FcMode::kFull);
    for (int step = 0; step < nSteps; step++) {
      const auto& orderA = planA.chunks(step);
      const auto& orderB = planB.chunks(step);
      const std::set<int> setA(orderA.begin(), orderA.end());
      const std::set<int> setB(orderB.begin(), orderB.end());
      EXPECT_EQ(setA, setB) << "rank=" << rank << " step=" << step;
    }
  }
}

TEST_P(PlanTest, PeerPairDisjoint) {
  const auto [nRanks, fcMode] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fcMode);
    for (int step = 0; step < nSteps; step++) {
      const auto peer = peerAt(rank, step, nRanks);
      const auto peerPlan = createRecvPlan(peer, nRanks, fcMode);

      const auto& orderR = plan.chunks(step);
      const auto& orderP = peerPlan.chunks(step);
      const std::set<int> setR(orderR.begin(), orderR.end());
      const std::set<int> setP(orderP.begin(), orderP.end());

      for (const auto pos : setR) {
        EXPECT_EQ(setP.count(pos), 0u)
            << "rank=" << rank << " peer=" << peer << " step=" << step
            << " pos=" << pos << " appears in both";
      }

      std::set<int> combined;
      combined.insert(setR.begin(), setR.end());
      combined.insert(setP.begin(), setP.end());
      EXPECT_EQ(combined.size(), 1 << (step + 1))
          << "rank=" << rank << " step=" << step;
    }
  }
}

TEST_P(PlanTest, SendRecvPlanMatch) {
  const auto [nRanks, fcMode] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto recvPlan = createRecvPlan(rank, nRanks, fcMode);
    const auto sendPlan = createSendPlan(rank, nRanks, fcMode);
    printPlan("recv", rank, nRanks, recvPlan);
    printPlan("send", rank, nRanks, sendPlan);
    for (int step = 0; step < nSteps; step++) {
      const auto peer = peerAt(rank, step, nRanks);
      const auto peerSendPlan = createSendPlan(peer, nRanks, fcMode);
      EXPECT_EQ(recvPlan.chunks(step), peerSendPlan.chunks(step))
          << "rank=" << rank << " peer=" << peer << " step=" << step;
    }
  }
}

TEST(PlanKnownValues, RecvOnlyNRanks8) {
  const auto plan = createRecvPlan(1, 8, FcMode::kRecvOnly);
  EXPECT_EQ(plan.chunk(1, 0), 3u);
  EXPECT_EQ(plan.chunk(1, 1), 7u);

  const std::vector<int> step2 = {0, 4, 2, 6};
  EXPECT_EQ(plan.chunks(2), step2);
}

TEST(PlanKnownValues, FullNRanks8) {
  const auto plan = createRecvPlan(1, 8, FcMode::kFull);
  const std::vector<int> step2 = {0, 2, 6, 4};
  EXPECT_EQ(plan.chunks(2), step2);
}

TEST(PlanKnownValues, RecvOnlyNRanks16) {
  const auto plan = createRecvPlan(1, 16, FcMode::kRecvOnly);
  const std::vector<int> step3 = {0, 8, 4, 12, 2, 10, 6, 14};
  EXPECT_EQ(plan.chunks(3), step3);
}

TEST(PlanKnownValues, FullNRanks16) {
  const auto plan = createRecvPlan(1, 16, FcMode::kFull);
  const std::vector<int> step3 = {0, 2, 6, 14, 10, 8, 4, 12};
  EXPECT_EQ(plan.chunks(3), step3);
}

TEST(PlanKnownValues, Rank4Step1NRanks8) {
  const auto plan = createRecvPlan(4, 8, FcMode::kRecvOnly);
  const std::vector<int> step1 = {6, 2};
  EXPECT_EQ(plan.chunks(1), step1);
}

INSTANTIATE_TEST_SUITE_P(
    AllRanksAndModes,
    PlanTest,
    ::testing::Combine(
        ::testing::Values(
            int(2),
            int(4),
            int(8),
            int(16),
            int(32),
            int(64),
            int(128),
            int(256)),
        ::testing::Values(FcMode::kRecvOnly, FcMode::kFull)));

// nRanks=1 is valid (single rank, no-op for ctsrd) — should produce 0 steps.
TEST(PlanEdgeCaseTest, SingleRankProducesZeroSteps) {
  for (const auto fcMode : {FcMode::kRecvOnly, FcMode::kFull}) {
    const auto recv = createRecvPlan(0, 1, fcMode);
    EXPECT_EQ(recv.nSteps(), 0);
    const auto send = createSendPlan(0, 1, fcMode);
    EXPECT_EQ(send.nSteps(), 0);
  }
}

// nRanks=2 is the smallest non-trivial case — should produce 1 step.
TEST(PlanEdgeCaseTest, TwoRanksProducesOneStep) {
  for (const auto fcMode : {FcMode::kRecvOnly, FcMode::kFull}) {
    for (int rank = 0; rank < 2; rank++) {
      const auto recv = createRecvPlan(rank, 2, fcMode);
      EXPECT_EQ(recv.nSteps(), 1);
      EXPECT_EQ(recv.chunks(0).size(), 1);
      EXPECT_EQ(recv.peer(0), 1 - rank);

      const auto send = createSendPlan(rank, 2, fcMode);
      EXPECT_EQ(send.nSteps(), 1);
      EXPECT_EQ(send.chunks(0).size(), 1);
      EXPECT_EQ(send.chunk(0, 0), rank);
    }
  }
}

TEST(PlanEdgeCaseTest, NegativeNRanksAborts) {
  EXPECT_DEATH(
      createRecvPlan(0, -1, FcMode::kRecvOnly), "nRanks must be positive");
  EXPECT_DEATH(
      createSendPlan(0, -1, FcMode::kRecvOnly), "nRanks must be positive");
}

TEST(PlanEdgeCaseTest, ZeroNRanksAborts) {
  EXPECT_DEATH(
      createRecvPlan(0, 0, FcMode::kRecvOnly), "nRanks must be positive");
  EXPECT_DEATH(
      createSendPlan(0, 0, FcMode::kRecvOnly), "nRanks must be positive");
}

TEST(PlanEdgeCaseTest, NegativeRankAborts) {
  EXPECT_DEATH(
      createRecvPlan(-1, 4, FcMode::kRecvOnly), "myRank .* out of range");
  EXPECT_DEATH(
      createSendPlan(-1, 4, FcMode::kRecvOnly), "myRank .* out of range");
}

TEST(PlanEdgeCaseTest, NonPowerOfTwoNRanksAborts) {
  EXPECT_DEATH(
      createRecvPlan(0, 3, FcMode::kRecvOnly), "nRanks must be a power of 2");
  EXPECT_DEATH(
      createSendPlan(0, 6, FcMode::kRecvOnly), "nRanks must be a power of 2");
}

TEST(PlanEdgeCaseTest, RankOutOfRangeAborts) {
  EXPECT_DEATH(
      createRecvPlan(4, 4, FcMode::kRecvOnly), "myRank .* out of range");
  EXPECT_DEATH(
      createSendPlan(4, 4, FcMode::kRecvOnly), "myRank .* out of range");
}

} // namespace
} // namespace ctran::allgather::ctsrd
