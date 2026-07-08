// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include "comms/ctran/algos/AllGather/StreamedRd/Common.h"
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
  std::cout << label << " rank=" << rank << " nRanks=" << nRanks
            << " fwdPeers=" << plan.fwdPeers() << "\n";
  for (int step = 0; step < plan.nSteps(); step++) {
    std::cout << "  step " << step << ": [";
    const auto& c = plan.chunks(step);
    for (size_t k = 0; k < c.size(); k++) {
      if (k > 0) {
        std::cout << ", ";
      }
      std::cout << c[k];
    }
    std::cout << "]\n";
  }
}

struct FakeFlushMapper {
  bool localFlushEnabled{false};
  int iflushCalls{0};

  bool isLocalFlushEnabled() const {
    return localFlushEnabled;
  }

  commResult_t iflush(
      const void* /*buf*/,
      const void* /*regHdl*/,
      CtranMapperRequest** req) {
    iflushCalls++;
    *req = new CtranMapperRequest();
    return commSuccess;
  }
};

struct FakeFlushContext {
  FakeFlushMapper* mapper{nullptr};
  void* recvbuff{nullptr};
  void* memHdl{nullptr};
  std::vector<std::vector<std::unique_ptr<CtranMapperRequest>>> recvFlushReqs;
};

int peerAt(const int rank, const int step, const int nRanks) {
  const auto dist = nRanks / (2 << step);
  const auto pos = (rank / dist) % 2;
  return pos == 0 ? rank + dist : rank - dist;
}

// Param: (nRanks, fwdPeers)
class PlanTest : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(PlanTest, CorrectSize) {
  const auto [nRanks, fwdPeers] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fwdPeers);
    EXPECT_EQ(plan.nSteps(), nSteps);
    for (int step = 0; step < nSteps; step++) {
      EXPECT_EQ(plan.chunks(step).size(), 1u << step)
          << "rank=" << rank << " step=" << step << " nRanks=" << nRanks;
    }
  }
}

TEST_P(PlanTest, UniquePositions) {
  const auto [nRanks, fwdPeers] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fwdPeers);
    for (int step = 0; step < nSteps; step++) {
      const auto& order = plan.chunks(step);
      const std::set<int> seen(order.begin(), order.end());
      EXPECT_EQ(seen.size(), order.size())
          << "rank=" << rank << " step=" << step;
    }
  }
}

TEST_P(PlanTest, CorrectPositionSet) {
  const auto [nRanks, fwdPeers] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fwdPeers);
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

TEST_P(PlanTest, PeerPairDisjoint) {
  const auto [nRanks, fwdPeers] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fwdPeers);
    for (int step = 0; step < nSteps; step++) {
      const auto peer = peerAt(rank, step, nRanks);
      const auto peerPlan = createRecvPlan(peer, nRanks, fwdPeers);

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
      EXPECT_EQ(combined.size(), 1u << (step + 1))
          << "rank=" << rank << " step=" << step;
    }
  }
}

TEST_P(PlanTest, SendRecvPlanMatch) {
  const auto [nRanks, fwdPeers] = GetParam();
  const auto nSteps = log2i(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    const auto recvPlan = createRecvPlan(rank, nRanks, fwdPeers);
    const auto sendPlan = createSendPlan(rank, nRanks, fwdPeers);
    printPlan("recv", rank, nRanks, recvPlan);
    printPlan("send", rank, nRanks, sendPlan);
    for (int step = 0; step < nSteps; step++) {
      const auto peer = peerAt(rank, step, nRanks);
      const auto peerSendPlan = createSendPlan(peer, nRanks, fwdPeers);
      EXPECT_EQ(recvPlan.chunks(step), peerSendPlan.chunks(step))
          << "rank=" << rank << " peer=" << peer << " step=" << step;
    }
  }
}

// All fwdPeers values must produce the same chunk SET at every step (only the
// order may differ between fwdPeers values).
TEST_P(PlanTest, EquivalentSetAcrossFwdPeers) {
  const auto [nRanks, fwdPeers] = GetParam();
  const auto nSteps = log2i(nRanks);
  const auto reference = nSteps; // recvOnly-equivalent
  if (fwdPeers == reference) {
    return;
  }
  for (int rank = 0; rank < nRanks; rank++) {
    const auto planA = createRecvPlan(rank, nRanks, fwdPeers);
    const auto planB = createRecvPlan(rank, nRanks, reference);
    for (int step = 0; step < nSteps; step++) {
      const std::set<int> setA(
          planA.chunks(step).begin(), planA.chunks(step).end());
      const std::set<int> setB(
          planB.chunks(step).begin(), planB.chunks(step).end());
      EXPECT_EQ(setA, setB) << "rank=" << rank << " step=" << step;
    }
  }
}

// Spot-checks of expected orderings for the unified walker are intentionally
// omitted: the per-fwdPeers chunk ORDER is an implementation-internal detail
// that only affects when (not whether) chunks are delivered, and is robustly
// covered by the parametric invariant tests above (CorrectSize,
// CorrectPositionSet, PeerPairDisjoint, SendRecvPlanMatch,
// EquivalentSetAcrossFwdPeers). Adding hand-derived order spot-checks here
// is brittle and easy to get wrong relative to the actual walker.

// First element rule: for any fwdPeers, step 0 should be just [sender]
// (only 1 chunk to send). This holds independently of fwdPeers.
TEST_P(PlanTest, FirstStepIsSenderOwnChunk) {
  const auto [nRanks, fwdPeers] = GetParam();
  for (int rank = 0; rank < nRanks; rank++) {
    const auto plan = createRecvPlan(rank, nRanks, fwdPeers);
    if (plan.nSteps() == 0) {
      continue;
    }
    const auto sender = peerAt(rank, 0, nRanks);
    EXPECT_EQ(plan.chunk(0, 0), sender) << "rank=" << rank;
  }
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
        // fwdPeers values: 0 (ctrd), 1 (full), 2 (middle), large
        // (recvOnly-equivalent). 8 exceeds nSteps for the smaller cases,
        // exercising the clamping/saturation behavior in the planner.
        ::testing::Values(0, 1, 2, 8)));

// nRanks=1 is valid (single rank, no-op) — should produce 0 steps.
TEST(PlanEdgeCaseTest, SingleRankProducesZeroSteps) {
  for (const int fwdPeers : {0, 1, 8}) {
    const auto recv = createRecvPlan(0, 1, fwdPeers);
    EXPECT_EQ(recv.nSteps(), 0);
    const auto send = createSendPlan(0, 1, fwdPeers);
    EXPECT_EQ(send.nSteps(), 0);
  }
}

// nRanks=2 is the smallest non-trivial case — should produce 1 step.
TEST(PlanEdgeCaseTest, TwoRanksProducesOneStep) {
  for (const int fwdPeers : {0, 1, 8}) {
    for (int rank = 0; rank < 2; rank++) {
      const auto recv = createRecvPlan(rank, 2, fwdPeers);
      EXPECT_EQ(recv.nSteps(), 1);
      EXPECT_EQ(recv.chunks(0).size(), 1u);
      EXPECT_EQ(recv.peer(0), 1 - rank);

      const auto send = createSendPlan(rank, 2, fwdPeers);
      EXPECT_EQ(send.nSteps(), 1);
      EXPECT_EQ(send.chunks(0).size(), 1u);
      EXPECT_EQ(send.chunk(0, 0), rank);
    }
  }
}

TEST(PlanEdgeCaseTest, NegativeNRanksAborts) {
  EXPECT_DEATH(createRecvPlan(0, -1, 1), "nRanks must be positive");
  EXPECT_DEATH(createSendPlan(0, -1, 1), "nRanks must be positive");
}

TEST(PlanEdgeCaseTest, ZeroNRanksAborts) {
  EXPECT_DEATH(createRecvPlan(0, 0, 1), "nRanks must be positive");
  EXPECT_DEATH(createSendPlan(0, 0, 1), "nRanks must be positive");
}

TEST(PlanEdgeCaseTest, NegativeRankAborts) {
  EXPECT_DEATH(createRecvPlan(-1, 4, 1), "myRank .* out of range");
  EXPECT_DEATH(createSendPlan(-1, 4, 1), "myRank .* out of range");
}

TEST(PlanEdgeCaseTest, NonPowerOfTwoNRanksAborts) {
  EXPECT_DEATH(createRecvPlan(0, 3, 1), "nRanks must be a power of 2");
  EXPECT_DEATH(createSendPlan(0, 6, 1), "nRanks must be a power of 2");
}

TEST(PlanEdgeCaseTest, RankOutOfRangeAborts) {
  EXPECT_DEATH(createRecvPlan(4, 4, 1), "myRank .* out of range");
  EXPECT_DEATH(createSendPlan(4, 4, 1), "myRank .* out of range");
}

TEST(PlanEdgeCaseTest, NegativeFwdPeersAborts) {
  EXPECT_DEATH(createRecvPlan(0, 4, -1), "fwdPeers must be >= 0");
  EXPECT_DEATH(createSendPlan(0, 4, -1), "fwdPeers must be >= 0");
}

// The unified walker handles fwdPeers values as follows for own (root) chunk
// placement in sendPlan[s]:
//   - fp = 0 (burst): own deferred to every step, pushed FIRST at each
//     step start via postDeferredPuts.
//   - fp ∈ [1, nSteps-1] (intermediate): own pre-enqueued for steps
//     [0, fp), then deferred (pushed later, after receive-driven forwards)
//     for steps [fp, nSteps). At intermediate steps the per-step ORDER
//     differs from fp=0 because received chunks land in the queue before
//     own's deferred flush.
//   - fp = nSteps (max streaming): own pre-enqueued for every step at
//     startup, so it ends up FIRST at every step (same final order as fp=0,
//     reached via a different mechanism).
//
// These invariants pin both equalities and distinctions:
//   * Chunk SET at every step is identical across all fp values.
//   * fp = 0 and fp = nSteps produce the SAME per-step chunk ORDER.
//   * Intermediate fp values produce DIFFERENT orders at some steps.

TEST(PlanFwdPeersInvariantsTest, ChunkSetSameAcrossFwdPeers) {
  const int nRanks = 16;
  const int nSteps = 4;
  for (int myRank = 0; myRank < nRanks; myRank++) {
    Plan ref = createSendPlan(myRank, nRanks, 0);
    for (int fp = 1; fp <= nSteps + 2; fp++) {
      Plan alt = createSendPlan(myRank, nRanks, fp);
      ASSERT_EQ(alt.nSteps(), nSteps);
      for (int s = 0; s < nSteps; s++) {
        std::set<int> refSet(ref.chunks(s).begin(), ref.chunks(s).end());
        std::set<int> altSet(alt.chunks(s).begin(), alt.chunks(s).end());
        EXPECT_EQ(altSet, refSet)
            << "rank=" << myRank << " step=" << s << " fp=" << fp
            << ": chunk SET must be identical across fwd_peers values.";
      }
    }
  }
}

TEST(PlanFwdPeersInvariantsTest, BurstAndMaxFwdPeersHaveSameOrder) {
  // fp=0 (burst, all deferred) and fp=nSteps (max streaming, all immediate)
  // both put own first at every step — via different mechanisms — and so
  // produce the SAME per-step chunk order. This is why the perf comparison
  // shows fp=0 and fp=-1 (which clamps to nSteps) as indistinguishable on
  // this cluster.
  const int nRanks = 16;
  const int nSteps = 4;
  for (int myRank = 0; myRank < nRanks; myRank++) {
    Plan burst = createSendPlan(myRank, nRanks, 0);
    Plan maxFwd = createSendPlan(myRank, nRanks, nSteps);
    for (int s = 0; s < nSteps; s++) {
      EXPECT_EQ(burst.chunks(s), maxFwd.chunks(s))
          << "rank=" << myRank << " step=" << s
          << ": fp=0 and fp=nSteps must produce the same per-step order.";
    }
  }
}

TEST(PlanFwdPeersInvariantsTest, IntermediateFwdPeersHasDifferentOrder) {
  // Intermediate fp values (e.g., fp=1) must produce a DIFFERENT chunk order
  // at some step than fp=0/fp=nSteps; otherwise the cvar would have no
  // observable effect at intermediate values. This guards against
  // accidentally reducing the walker to ignore fp.
  const int nRanks = 16;
  const int nSteps = 4;
  bool anyDiff = false;
  for (int myRank = 0; myRank < nRanks; myRank++) {
    Plan burst = createSendPlan(myRank, nRanks, 0);
    Plan intermediate = createSendPlan(myRank, nRanks, 1);
    for (int s = 0; s < nSteps; s++) {
      if (burst.chunks(s) != intermediate.chunks(s)) {
        anyDiff = true;
      }
    }
  }
  EXPECT_TRUE(anyDiff)
      << "fp=0 and fp=1 must produce distinguishable per-step orders at some "
         "rank/step (otherwise the walker is ignoring fwd_peers).";
}

TEST(PlanFwdPeersInvariantsTest, BurstStepChunkCountIsPowerOfTwo) {
  // Burst mode: step S sends 2^S chunks (own + all received from prior steps).
  const int nRanks = 16;
  const int nSteps = 4;
  Plan burst = createSendPlan(0, nRanks, 0);
  for (int s = 0; s < nSteps; s++) {
    EXPECT_EQ(burst.chunks(s).size(), size_t(1) << s)
        << "fp=0 step " << s << " should send 2^" << s << " chunks";
    EXPECT_EQ(burst.chunks(s).front(), 0)
        << "fp=0 step " << s << ": own chunk must be at front";
  }
}

TEST(CommonFlushTest, SkipsFlushRequestWhenLocalFlushDisabled) {
  FakeFlushMapper mapper;
  mapper.localFlushEnabled = false;
  FakeFlushContext ctx;
  ctx.mapper = &mapper;
  ctx.recvFlushReqs.resize(1);
  CtranMapperRequest* flushReq = nullptr;

  EXPECT_EQ(common::postRecvFlush(ctx, 0, &flushReq), commSuccess);
  EXPECT_EQ(flushReq, nullptr);
  EXPECT_EQ(mapper.iflushCalls, 0);
  EXPECT_TRUE(ctx.recvFlushReqs.at(0).empty());
}

TEST(CommonFlushTest, StoresFlushRequestWhenLocalFlushEnabled) {
  FakeFlushMapper mapper;
  mapper.localFlushEnabled = true;
  FakeFlushContext ctx;
  ctx.mapper = &mapper;
  ctx.recvFlushReqs.resize(1);
  CtranMapperRequest* flushReq = nullptr;

  EXPECT_EQ(common::postRecvFlush(ctx, 0, &flushReq), commSuccess);
  ASSERT_NE(flushReq, nullptr);
  EXPECT_EQ(mapper.iflushCalls, 1);
  ASSERT_EQ(ctx.recvFlushReqs.at(0).size(), 1);
  EXPECT_EQ(ctx.recvFlushReqs.at(0).front().get(), flushReq);
}

} // namespace
} // namespace ctran::allgather::ctsrd
