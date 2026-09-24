// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Unit tests for P2pIbrcHostLanes::split(), the lane division a striped
// host-driven transfer relies on. Pure arithmetic, so no fabric and no GPU:
// the properties below are exactly what both endpoints depend on agreeing.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/prims/transport/ibrc/P2pIbrcHostLanes.h"

namespace {

using comms::prims::IbgdaLocalBuffer;
using comms::prims::IbgdaRemoteBuffer;
using comms::prims::IbrcDesc;
using comms::prims::IbrcNicStatus;
using comms::prims::NetworkLKey;
using comms::prims::NetworkLKeys;
using comms::prims::NetworkRKey;
using comms::prims::NetworkRKeys;
using comms::prims::P2pIbrcHostLanes;
using comms::prims::P2pIbrcHostWriter;
constexpr std::size_t kMin = P2pIbrcHostLanes::kDefaultMinLaneBytes;
constexpr std::size_t kAlign = P2pIbrcHostLanes::kLaneAlignment;

// Every split must cover [0, nbytes) exactly once, with no gap and no overlap.
// A gap silently drops payload; an overlap makes two lanes race on the same
// bytes. Neither would be caught by a sender-side check.
void expectCoversExactly(
    const std::vector<P2pIbrcHostLanes::LaneRange>& ranges,
    std::size_t nbytes) {
  ASSERT_FALSE(ranges.empty());
  EXPECT_EQ(ranges.front().offset, 0u);
  std::size_t total = 0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    EXPECT_GT(ranges[i].bytes, 0u) << "lane " << i << " is empty";
    EXPECT_EQ(ranges[i].offset, total) << "lane " << i << " leaves a gap";
    total += ranges[i].bytes;
  }
  EXPECT_EQ(total, nbytes);
}

TEST(P2pIbrcHostLanesSplit, SingleLaneIsWholeTransfer) {
  const auto r = P2pIbrcHostLanes::split(8 << 20, 1);
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].offset, 0u);
  EXPECT_EQ(r[0].bytes, std::size_t{8} << 20);
}

TEST(P2pIbrcHostLanesSplit, EvenSplitCoversTransfer) {
  const std::size_t n = 8 << 20;
  for (int lanes : {2, 4, 8}) {
    const auto r = P2pIbrcHostLanes::split(n, lanes);
    EXPECT_EQ(static_cast<int>(r.size()), lanes);
    expectCoversExactly(r, n);
  }
}

TEST(P2pIbrcHostLanesSplit, AllButLastAreAlignedAndEqual) {
  const auto r = P2pIbrcHostLanes::split(8 << 20, 4);
  ASSERT_EQ(r.size(), 4u);
  for (std::size_t i = 0; i + 1 < r.size(); ++i) {
    EXPECT_EQ(r[i].bytes, r[0].bytes);
    EXPECT_EQ(r[i].offset % kAlign, 0u);
  }
}

// The remainder rides the last lane rather than being spread, so the mapping
// stays reproducible from (nbytes, lanes) alone on both sides.
TEST(P2pIbrcHostLanesSplit, RemainderGoesToLastLane) {
  const std::size_t n = (4 << 20) + 17;
  const auto r = P2pIbrcHostLanes::split(n, 4);
  ASSERT_EQ(r.size(), 4u);
  expectCoversExactly(r, n);
  EXPECT_GT(r.back().bytes, r.front().bytes);
}

// Below the gate a transfer is latency-bound; striping it only adds
// cross-lane straggler cost, so it collapses to one lane.
TEST(P2pIbrcHostLanesSplit, BelowMinChannelBytesCollapsesToOneLane) {
  const auto r = P2pIbrcHostLanes::split(kMin, 4);
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].bytes, kMin);
}

// A transfer that cannot afford every lane should use as many as it CAN, not
// fall back to one. Collapsing here measured 17us slower at 512KB on GB200,
// because two lanes are plainly worth having even when four are not.
TEST(P2pIbrcHostLanesSplit, ClampsLaneCountInsteadOfCollapsing) {
  const std::size_t n = 512 * 1024; // affords 2 lanes at a 256KB floor, not 4
  const auto r = P2pIbrcHostLanes::split(n, 4);
  ASSERT_EQ(r.size(), 2u);
  expectCoversExactly(r, n);
}

TEST(P2pIbrcHostLanesSplit, ClampScalesWithSize) {
  EXPECT_EQ(P2pIbrcHostLanes::split(kMin * 3, 8).size(), 3u);
  EXPECT_EQ(P2pIbrcHostLanes::split(kMin * 6, 8).size(), 6u);
  EXPECT_EQ(P2pIbrcHostLanes::split(kMin * 99, 8).size(), 8u);
}

TEST(P2pIbrcHostLanesSplit, ExactlyAtThresholdSplits) {
  const std::size_t n = kMin * 4;
  const auto r = P2pIbrcHostLanes::split(n, 4);
  EXPECT_EQ(r.size(), 4u);
  expectCoversExactly(r, n);
}

// An empty lane would never signal, so a receiver waiting on it would hang
// rather than fail. Splits that cannot give every lane bytes must collapse.
TEST(P2pIbrcHostLanesSplit, NeverEmitsAnEmptyLane) {
  for (std::size_t n : {std::size_t{1}, kAlign - 1, kAlign, kMin * 2 + 1}) {
    for (int lanes : {1, 2, 3, 8, 16}) {
      const auto r = P2pIbrcHostLanes::split(n, lanes);
      expectCoversExactly(r, n);
    }
  }
}

TEST(P2pIbrcHostLanesSplit, TinyTransferWithManyLanesCollapses) {
  const auto r = P2pIbrcHostLanes::split(64, 16);
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].bytes, 64u);
}

// Both endpoints call split() independently and must agree, so it has to be a
// pure function of its arguments.
TEST(P2pIbrcHostLanesSplit, IsDeterministic) {
  const std::size_t n = (8 << 20) + 4096;
  const auto a = P2pIbrcHostLanes::split(n, 4);
  const auto b = P2pIbrcHostLanes::split(n, 4);
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].offset, b[i].offset);
    EXPECT_EQ(a[i].bytes, b[i].bytes);
  }
}

// A caller may lower the gate; the coverage properties must still hold.
TEST(P2pIbrcHostLanesSplit, HonorsCallerMinChannelBytes) {
  const auto r = P2pIbrcHostLanes::split(4096, 2, /*minLaneBytes=*/1024);
  ASSERT_EQ(r.size(), 2u);
  expectCoversExactly(r, 4096);
}

// The stride separating per-lane signal slots, as the hostring consumer uses.
constexpr std::size_t kStride = 128;

// A zero stride puts every lane's signal on one counter, so one arrival would
// satisfy the wait for all of them while the rest are still in flight. It has
// to be refused before anything is posted, because put_striped() cannot recall
// lanes already on the wire.
TEST(P2pIbrcHostLanesLayout, RejectsZeroStride) {
  EXPECT_THROW(
      P2pIbrcHostLanes::LaneLayout(kMin * 4, 4, /*signalStride=*/0),
      std::runtime_error);
}

TEST(P2pIbrcHostLanesLayout, RejectsMisalignedStride) {
  for (std::size_t stride :
       {std::size_t{4}, std::size_t{12}, std::size_t{129}}) {
    EXPECT_THROW(
        P2pIbrcHostLanes::LaneLayout(kMin * 4, 4, stride), std::runtime_error)
        << "stride " << stride;
  }
}

// A stride narrower than one counter overlaps adjacent slots, which aliases
// lanes the same way a zero stride does.
TEST(P2pIbrcHostLanesLayout, RejectsStrideNarrowerThanACounter) {
  EXPECT_THROW(
      P2pIbrcHostLanes::LaneLayout(kMin * 4, 4, sizeof(uint64_t) / 2),
      std::runtime_error);
}

// The layout is the single source of the lane count: it must report exactly
// what split() would have produced, including the collapse cases.
TEST(P2pIbrcHostLanesLayout, UsedMatchesSplit) {
  for (std::size_t n : {kMin, kMin * 2, kMin * 4, 512 * std::size_t{1024}}) {
    for (int lanes : {1, 2, 4, 8}) {
      const P2pIbrcHostLanes::LaneLayout layout(n, lanes, kStride);
      EXPECT_EQ(
          layout.used(),
          static_cast<int>(P2pIbrcHostLanes::split(n, lanes).size()))
          << "nbytes " << n << " lanes " << lanes;
      EXPECT_EQ(layout.signalStride(), kStride);
    }
  }
}

// Two lanes over throwaway rings. Nothing drains them, so any wait runs to its
// bound -- which is what makes the bound observable.
class P2pIbrcHostLanesBudget : public ::testing::Test {
 protected:
  static constexpr int kLanes = 2;
  static constexpr uint32_t kDepth = 4;

  P2pIbrcHostLanes makeLanes() {
    return P2pIbrcHostLanes(makeWriters());
  }

  std::vector<P2pIbrcHostWriter> makeWriters(
      const std::shared_ptr<void>& lease = nullptr) {
    std::vector<P2pIbrcHostWriter> writers;
    writers.reserve(kLanes);
    for (int l = 0; l < kLanes; ++l) {
      const auto i = static_cast<std::size_t>(l);
      writers.emplace_back(
          descs_[i].data(),
          &pi_[i],
          &ci_[i],
          &status_[i],
          kDepth,
          static_cast<uint32_t>(l),
          lease);
    }
    return writers;
  }

  // The reason a call refused, so a test can tell a real rejection from an
  // unrelated failure such as a wait simply timing out.
  static std::string refusal(const std::function<void()>& fn) {
    try {
      fn();
    } catch (const std::runtime_error& e) {
      return e.what();
    }
    return "";
  }

  // Why a wait on `lane` gave up.
  static std::string waitFailure(P2pIbrcHostLanes& lanes, int lane) {
    uint64_t counter = 0;
    try {
      lanes.writer(lane).poll_counter(&counter, 1);
    } catch (const std::runtime_error& e) {
      return e.what();
    }
    return "";
  }

  static IbgdaLocalBuffer localBuffer() {
    NetworkLKeys keys(kLanes);
    for (int i = 0; i < kLanes; ++i) {
      keys[i] = NetworkLKey{static_cast<uint32_t>(0x1000 + i)};
    }
    return IbgdaLocalBuffer(reinterpret_cast<void*>(0x100000), keys);
  }

  static IbgdaRemoteBuffer remoteBuffer() {
    NetworkRKeys keys(kLanes);
    for (int i = 0; i < kLanes; ++i) {
      keys[i] = NetworkRKey{static_cast<uint32_t>(0x2000 + i)};
    }
    return IbgdaRemoteBuffer(reinterpret_cast<void*>(0x200000), keys);
  }

  std::array<std::array<IbrcDesc, kDepth>, kLanes> descs_{};
  std::array<uint64_t, kLanes> pi_{};
  std::array<uint64_t, kLanes> ci_{};
  std::array<IbrcNicStatus, kLanes> status_{};
};

// The rings a lanes object holds are exclusive: getHostLanes() hands out
// [0, numLanes), so a second object for the same peer would put two producers
// on the same rings, and the writer's backpressure check is only sound with
// one. The claim must also be released when the object dies, or a
// communicator could never rebuild its cache.
TEST_F(P2pIbrcHostLanesBudget, RingClaimIsExclusiveAndReleased) {
  // Stands in for the transport's per-peer claim: getHostLanes() refuses while
  // the token is alive, and the weak_ptr expires when the holder dies.
  std::weak_ptr<void> issued;
  {
    auto owner = std::make_shared<char>();
    issued = owner;
    // Not std::move(owner): argument evaluation is unordered, so moving here
    // could hand makeWriters() an already-emptied token.
    auto held = P2pIbrcHostLanes(makeWriters(owner), owner);
    EXPECT_FALSE(issued.expired()) << "claim not held while lanes are alive";
  }
  EXPECT_TRUE(issued.expired()) << "claim outlived the lanes object";
}

/*
 * writer() hands out a movable writer, so a lane can be moved out and outlive
 * the object that was handed the claim. Every lane therefore shares the claim
 * rather than the lanes object holding it alone -- otherwise the peer reads as
 * free the moment the object dies, and the transport issues a second producer
 * onto a ring the escaped writer is still driving.
 */
TEST_F(P2pIbrcHostLanesBudget, ClaimSurvivesALaneMovedOutOfTheObject) {
  std::weak_ptr<void> issued;
  std::optional<P2pIbrcHostWriter> escaped;
  {
    auto owner = std::make_shared<char>();
    issued = owner;
    auto lanes = P2pIbrcHostLanes(makeWriters(owner), owner);
    escaped = std::move(lanes.writer(0));
  }
  EXPECT_FALSE(issued.expired())
      << "claim released while a lane writer can still drive the ring";

  escaped.reset();
  EXPECT_TRUE(issued.expired()) << "claim outlived the last lane writer";
}

/*
 * The sharing above is a precondition, not a convention: a future second way
 * to build lanes that forgot to pass the claim down would otherwise compile,
 * pass every test here, and silently restore the escape. Checked at
 * construction so the mistake cannot reach the wire.
 */
TEST_F(P2pIbrcHostLanesBudget, RejectsLanesThatDoNotShareTheClaim) {
  auto owner = std::make_shared<char>();

  EXPECT_THROW(
      P2pIbrcHostLanes(makeWriters(/*lease=*/nullptr), owner),
      std::runtime_error);

  // A different peer's claim is the same defect as none at all.
  EXPECT_THROW(
      P2pIbrcHostLanes(makeWriters(std::make_shared<char>()), owner),
      std::runtime_error);

  EXPECT_NO_THROW(P2pIbrcHostLanes(makeWriters(owner), owner));
}

// A layout built against a wider peer must be refused rather than indexed
// with: the wait side would otherwise run off the end of writers_.
TEST_F(P2pIbrcHostLanesBudget, RejectsLayoutWiderThanLanes) {
  auto lanes = makeLanes();
  // Keeps the test bounded if the guard ever regresses and the poll proceeds.
  lanes.set_timeout(std::chrono::milliseconds(1));

  const P2pIbrcHostLanes::LaneLayout wide(
      P2pIbrcHostLanes::kDefaultMinLaneBytes * 8, /*lanes=*/8, kStride);
  ASSERT_GT(wide.used(), lanes.lanes());

  std::array<uint64_t, 8> counters{};
  EXPECT_THROW(lanes.poll_all(wide, counters.data(), 1), std::runtime_error);
}

// A bad stride can no longer reach a post at all: every entry point takes a
// layout, and the layout that would carry the bad stride throws on
// construction. This pins that end of it -- the writers are never touched, so
// nothing can be on the wire when the stride is refused.
TEST_F(P2pIbrcHostLanesBudget, BadStrideCannotReachAPost) {
  auto lanes = makeLanes();
  for (std::size_t stride :
       {std::size_t{0}, std::size_t{4}, std::size_t{12}, std::size_t{129}}) {
    EXPECT_NE(
        refusal([&] {
          const P2pIbrcHostLanes::LaneLayout bad(kMin * kLanes, kLanes, stride);
          lanes.put_striped(bad, localBuffer(), remoteBuffer(), remoteBuffer());
        }).find("signal stride"),
        std::string::npos)
        << "accepted stride " << stride;
  }
  for (int l = 0; l < kLanes; ++l) {
    EXPECT_EQ(pi_[static_cast<std::size_t>(l)], 0u) << "lane " << l;
  }
}

// The transport arms every lane with the communicator's abort when it builds
// them. Adjusting only the deadline must not drop it, or an aborted job spins
// out the full timeout on every lane instead of unwinding.
TEST_F(P2pIbrcHostLanesBudget, TimeoutChangeKeepsAbortPredicate) {
  auto lanes = makeLanes();
  lanes.set_wait_policy(std::chrono::seconds(30), [] { return true; });
  lanes.set_timeout(std::chrono::seconds(30));

  const auto start = std::chrono::steady_clock::now();
  for (int l = 0; l < kLanes; ++l) {
    EXPECT_NE(waitFailure(lanes, l).find("aborted"), std::string::npos)
        << "lane " << l << " lost its abort predicate";
  }
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
}

// A striped transfer waits on every lane, so a lane left on its own per-wait
// bound would let the operation overrun the caller's timeout.
TEST_F(P2pIbrcHostLanesBudget, BudgetReachesEveryLane) {
  auto lanes = makeLanes();
  lanes.set_timeout(std::chrono::milliseconds(1));

  const auto budgets = lanes.begin_op_until(
      std::chrono::steady_clock::now() + std::chrono::milliseconds(1));
  for (int l = 0; l < kLanes; ++l) {
    EXPECT_NE(
        waitFailure(lanes, l).find("operation budget expired"),
        std::string::npos)
        << "lane " << l << " kept its per-wait bound";
  }
}

TEST_F(P2pIbrcHostLanesBudget, BudgetReleasesOnEveryLane) {
  auto lanes = makeLanes();
  lanes.set_timeout(std::chrono::milliseconds(1));

  {
    const auto budgets = lanes.begin_op_until(
        std::chrono::steady_clock::now() + std::chrono::milliseconds(1));
  }
  for (int l = 0; l < kLanes; ++l) {
    EXPECT_NE(
        waitFailure(lanes, l).find("timed out after 1ms"), std::string::npos)
        << "lane " << l << " kept the operation budget";
  }
}

} // namespace
