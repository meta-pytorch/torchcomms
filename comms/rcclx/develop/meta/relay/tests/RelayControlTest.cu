// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Protocol tests for the sharded-relay host control plane.
 *
 * WHY FORKED PROCESSES AND NOT RANKS
 *
 * The interesting failures here are not "does a plan arrive" -- they are what
 * happens when a segment is malformed, when its creator is dead, when a
 * consumer falls more than a ring behind, and when a reader lands in the middle
 * of a write. A rank layout cannot produce any of those on demand: every rank
 * runs the same code with the same environment. Forked children can, because
 * the parent chooses each child's geometry, its role, and when it dies.
 *
 * So this suite runs at ppn=1 and builds its own writer/reader processes. The
 * communicator-bound layer -- relayControlInit/Release and the bootstrap
 * unanimity vote -- is covered where a real 8-rank comm exists, in the
 * production-shape control plane test.
 */

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

#include "meta/relay/relay_control.h"
#include "nccl.h"

using namespace rcclx::relay;

namespace {

constexpr int64_t kMs = 1000LL * 1000LL;
constexpr int64_t kGenerousTimeoutNs = 60LL * 1000LL * kMs; // 60 s
constexpr int64_t kShortTimeoutNs = 150LL * kMs;

// Every test gets its own segment name. Tests share a process and the harness
// may run several binaries on one host, so a fixed name would make unrelated
// runs collide and fail for reasons that have nothing to do with the protocol.
uint64_t uniqueHash(uint32_t salt) {
  return (static_cast<uint64_t>(getpid()) << 24) ^ (salt * 0x9E3779B9ull) ^
      0xC0FFEEull;
}

RelayControlConfig makeConfig(
    uint64_t hash,
    uint32_t nRanks,
    uint32_t rank,
    uint32_t ringDepth,
    uint32_t maxCalls) {
  RelayControlConfig cfg;
  cfg.nRanks = nRanks;
  cfg.rank = rank;
  cfg.nActive = 0;
  cfg.commHash = hash;
  cfg.ringDepth = ringDepth;
  cfg.maxCalls = maxCalls;
  return cfg;
}

// A deterministic, epoch-dependent plan. The point of varying nCalls, opCode
// and every count with the epoch is that a torn or stale read cannot coincide
// with a correct one: any mismatch localizes to a field and an epoch.
uint32_t planCalls(uint64_t epoch, uint32_t maxCalls) {
  return 1u + static_cast<uint32_t>(epoch % maxCalls);
}

uint32_t planOp(uint64_t epoch) {
  return kRelayOpAllReduce +
      static_cast<uint32_t>(epoch % (kRelayOpCount - kRelayOpAllReduce));
}

size_t planCount(uint64_t epoch, uint32_t i) {
  return static_cast<size_t>(epoch * 1000003ull + i * 7ull + 1ull);
}

RelayPlanInfo makePlan(uint64_t epoch, uint32_t maxCalls) {
  RelayPlanInfo info{};
  info.nCalls = planCalls(epoch, maxCalls);
  info.opCode = planOp(epoch);
  info.dtype = static_cast<uint32_t>(ncclFloat32);
  info.redOp = static_cast<uint32_t>(ncclSum);
  return info;
}

void fillCounts(uint64_t epoch, uint32_t n, std::vector<size_t>& out) {
  for (uint32_t i = 0; i < n; i++) {
    out[i] = planCount(epoch, i);
  }
}

// Child exit codes. Distinct per failure so a red test says which invariant
// broke, in a context where gtest assertions cannot be reported upward.
enum ChildStatus : int {
  kChildOk = 0,
  kChildAttachFailed = 10,
  kChildConsumeFailed = 11,
  kChildWrongCallCount = 12,
  kChildWrongOpCode = 13,
  kChildWrongCounts = 14,
  kChildCreateFailed = 15,
};

void expectChildOk(pid_t pid) {
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status))
      << "child did not exit normally (signal "
      << (WIFSIGNALED(status) ? WTERMSIG(status) : -1) << ")";
  ASSERT_EQ(WEXITSTATUS(status), kChildOk)
      << "child reported failure code " << WEXITSTATUS(status);
}

/**
 * Wait until every listed rank has registered as a consumer.
 *
 * This is the precondition the real integration gets for free and these tests
 * must establish for themselves. In production, comm init is a bootstrap
 * all-gather that publisher and helpers leave together, and the publisher's
 * first forward then costs milliseconds of Python and GPU work while the
 * helper's very next action is its first consume() -- so the helper is always
 * registered long before the ring could wrap.
 *
 * Here the "helpers" are freshly forked processes that need milliseconds to
 * exist, racing a publisher whose publishes take 50 ns. Without this barrier
 * the publisher laps the ring before the last child registers, and that child
 * correctly reports a desync -- a real property, specified by
 * ALateConsumerGetsADesyncErrorNotCorruption, but not the property these tests
 * are about.
 */
void awaitConsumers(
    const RelayControlBlock& block,
    const std::vector<uint32_t>& ranks) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  for (;;) {
    size_t registered = 0;
    for (const uint32_t r : ranks) {
      if (block.consumerProgress(r) != 0) {
        registered++;
      }
    }
    if (registered == ranks.size()) {
      return;
    }
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "only " << registered << " of " << ranks.size()
        << " consumers ever registered";
    usleep(200);
  }
}

} // namespace

TEST(RelayControlBlockTest, CreateAndAttachRoundTripsOnePlan) {
  const uint64_t hash = uniqueHash(1);
  const uint32_t maxCalls = 8;
  auto cfg = makeConfig(hash, 2, 0, 4, maxCalls);

  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(cfg));
  EXPECT_EQ(writer.ringDepth(), 4u);
  EXPECT_EQ(writer.maxCalls(), maxCalls);

  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, 4, maxCalls)));

  std::vector<size_t> counts(maxCalls);
  RelayPlanInfo sent = makePlan(0, maxCalls);
  fillCounts(0, sent.nCalls, counts);
  ASSERT_EQ(
      writer.publish(0, sent, counts.data(), kGenerousTimeoutNs), ncclSuccess);

  RelayPlanInfo got{};
  std::vector<size_t> gotCounts(maxCalls, 0);
  ASSERT_EQ(
      reader.consume(0, &got, gotCounts.data(), maxCalls, kGenerousTimeoutNs),
      ncclSuccess);
  EXPECT_EQ(got.nCalls, sent.nCalls);
  EXPECT_EQ(got.opCode, sent.opCode);
  EXPECT_EQ(got.dtype, sent.dtype);
  EXPECT_EQ(got.redOp, sent.redOp);
  for (uint32_t i = 0; i < sent.nCalls; i++) {
    EXPECT_EQ(gotCounts[i], planCount(0, i)) << "count " << i;
  }
  EXPECT_EQ(writer.highWaterCalls(), sent.nCalls);
}

TEST(RelayControlBlockTest, SegmentBytesMatchesTheDocumentedLayout) {
  // 64-byte header + one 64-byte-aligned consumed block + ringDepth slots of
  // (8 seq + 32 info + 8 * maxCalls), each aligned to 64.
  EXPECT_EQ(RelayControlBlock::segmentBytes(8, 4, 128), 64u + 64u + 4u * 1088u);
  // Defaults stay comfortably inside a few pages.
  EXPECT_LT(RelayControlBlock::segmentBytes(8, 4, 128), 8u * 1024u);
}

TEST(RelayControlBlockTest, RingWrapsWithoutLosingOrCorruptingPlans) {
  const uint64_t hash = uniqueHash(2);
  const uint32_t maxCalls = 4;
  const uint32_t ringDepth = 2;
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, ringDepth, maxCalls)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, ringDepth, maxCalls)));

  std::vector<size_t> counts(maxCalls);
  std::vector<size_t> got(maxCalls);
  // Several times around the ring, so slot reuse is exercised rather than just
  // the first pass through fresh slots.
  for (uint64_t epoch = 0; epoch < 10; epoch++) {
    RelayPlanInfo sent = makePlan(epoch, maxCalls);
    fillCounts(epoch, sent.nCalls, counts);
    ASSERT_EQ(
        writer.publish(epoch, sent, counts.data(), kGenerousTimeoutNs),
        ncclSuccess)
        << "epoch " << epoch;
    RelayPlanInfo info{};
    ASSERT_EQ(
        reader.consume(epoch, &info, got.data(), maxCalls, kGenerousTimeoutNs),
        ncclSuccess)
        << "epoch " << epoch;
    ASSERT_EQ(info.nCalls, sent.nCalls) << "epoch " << epoch;
    for (uint32_t i = 0; i < sent.nCalls; i++) {
      ASSERT_EQ(got[i], planCount(epoch, i))
          << "epoch " << epoch << " count " << i;
    }
  }
}

TEST(RelayControlBlockTest, ConcurrentReadersNeverObserveATornPlan) {
  const uint32_t kReaders = 4;
  const uint32_t nRanks = kReaders + 1;
  const uint32_t maxCalls = 64;
  const uint32_t ringDepth = 4;
  const uint64_t kEpochs = 400;
  const uint64_t hash = uniqueHash(3);

  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, nRanks, 0, ringDepth, maxCalls)));

  std::vector<pid_t> readers;
  for (uint32_t r = 1; r <= kReaders; r++) {
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      RelayControlBlock reader;
      if (!reader.attach(makeConfig(hash, nRanks, r, ringDepth, maxCalls))) {
        _exit(kChildAttachFailed);
      }
      std::vector<size_t> counts(maxCalls, 0);
      for (uint64_t epoch = 0; epoch < kEpochs; epoch++) {
        RelayPlanInfo info{};
        if (reader.consume(
                epoch, &info, counts.data(), maxCalls, kGenerousTimeoutNs) !=
            ncclSuccess) {
          _exit(kChildConsumeFailed);
        }
        if (info.nCalls != planCalls(epoch, maxCalls)) {
          _exit(kChildWrongCallCount);
        }
        if (info.opCode != planOp(epoch)) {
          _exit(kChildWrongOpCode);
        }
        for (uint32_t i = 0; i < info.nCalls; i++) {
          if (counts[i] != planCount(epoch, i)) {
            _exit(kChildWrongCounts);
          }
        }
      }
      _exit(kChildOk);
    }
    readers.push_back(pid);
  }

  // Every reader must be in its consume loop before the ring can wrap.
  awaitConsumers(writer, {1, 2, 3, 4});

  std::vector<size_t> counts(maxCalls);
  for (uint64_t epoch = 0; epoch < kEpochs; epoch++) {
    RelayPlanInfo info = makePlan(epoch, maxCalls);
    fillCounts(epoch, info.nCalls, counts);
    ASSERT_EQ(
        writer.publish(epoch, info, counts.data(), kGenerousTimeoutNs),
        ncclSuccess)
        << "epoch " << epoch;
  }

  for (const pid_t pid : readers) {
    expectChildOk(pid);
  }
  EXPECT_EQ(writer.highWaterCalls(), maxCalls);
}

TEST(
    RelayControlBlockTest,
    PublisherWaitsForALaggingConsumerRatherThanDropping) {
  const uint32_t maxCalls = 4;
  const uint32_t ringDepth = 2;
  const uint64_t kEpochs = 12;
  const uint64_t hash = uniqueHash(4);

  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, ringDepth, maxCalls)));

  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    RelayControlBlock reader;
    if (!reader.attach(makeConfig(hash, 2, 1, ringDepth, maxCalls))) {
      _exit(kChildAttachFailed);
    }
    std::vector<size_t> counts(maxCalls, 0);
    for (uint64_t epoch = 0; epoch < kEpochs; epoch++) {
      RelayPlanInfo info{};
      if (reader.consume(
              epoch, &info, counts.data(), maxCalls, kGenerousTimeoutNs) !=
          ncclSuccess) {
        _exit(kChildConsumeFailed);
      }
      if (info.nCalls != planCalls(epoch, maxCalls)) {
        _exit(kChildWrongCallCount);
      }
      for (uint32_t i = 0; i < info.nCalls; i++) {
        if (counts[i] != planCount(epoch, i)) {
          _exit(kChildWrongCounts);
        }
      }
      // Lag AFTER registering, which is what a real helper doing work between
      // consumes looks like. Sleeping before the first consume would instead
      // violate the start-up contract tested separately below.
      usleep(20 * 1000);
    }
    _exit(kChildOk);
  }

  awaitConsumers(writer, {1});

  std::vector<size_t> counts(maxCalls);
  for (uint64_t epoch = 0; epoch < kEpochs; epoch++) {
    RelayPlanInfo info = makePlan(epoch, maxCalls);
    fillCounts(epoch, info.nCalls, counts);
    ASSERT_EQ(
        writer.publish(epoch, info, counts.data(), kGenerousTimeoutNs),
        ncclSuccess)
        << "epoch " << epoch;
  }
  expectChildOk(pid);
}

/**
 * The start-up contract, and what happens when it is broken.
 *
 * A rank's ROLE is not knowable to the segment: active ranks attach too, and
 * they never consume, so the publisher cannot simply wait on everyone who
 * attached. Consumers therefore register themselves on entry to consume(),
 * which means a consumer that has not yet reached its first consume() is not
 * protected, and the publisher is free to run a full ring ahead.
 *
 * In the real system that window does not exist in any practical sense: comm
 * init is a bootstrap barrier that both sides leave together, the helper's next
 * action is its first consume(), and the publisher would have to complete
 * ringDepth ENTIRE forwards -- GPU work included -- before the helper executes
 * one instruction of its loop.
 *
 * This test exists so that the behaviour when it IS broken is specified rather
 * than discovered: the late consumer gets a bounded, attributed desync error,
 * not corruption and not a hang.
 */
TEST(RelayControlBlockTest, ALateConsumerGetsADesyncErrorNotCorruption) {
  const uint32_t maxCalls = 4;
  const uint32_t ringDepth = 2;
  const uint64_t hash = uniqueHash(21);

  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, ringDepth, maxCalls)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, ringDepth, maxCalls)));

  // The consumer has attached but never entered consume(), so it is not
  // registered and the publisher runs away.
  std::vector<size_t> counts(maxCalls);
  for (uint64_t epoch = 0; epoch < 6; epoch++) {
    RelayPlanInfo info = makePlan(epoch, maxCalls);
    fillCounts(epoch, info.nCalls, counts);
    ASSERT_EQ(
        writer.publish(epoch, info, counts.data(), kGenerousTimeoutNs),
        ncclSuccess)
        << "epoch " << epoch;
  }

  RelayPlanInfo got{};
  std::vector<size_t> out(maxCalls, 0);
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(
      reader.consume(0, &got, out.data(), maxCalls, kGenerousTimeoutNs),
      ncclInternalError);
  // Bounded and immediate: waiting cannot bring epoch 0 back.
  EXPECT_LT(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count(),
      1000);
}

TEST(RelayControlBlockTest, PublisherTimesOutAndAttributesTheLaggingRank) {
  const uint32_t maxCalls = 4;
  const uint32_t ringDepth = 2;
  const uint64_t hash = uniqueHash(5);

  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, ringDepth, maxCalls)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, ringDepth, maxCalls)));

  std::vector<size_t> counts(maxCalls);
  std::vector<size_t> got(maxCalls);
  for (uint64_t epoch = 0; epoch < 2; epoch++) {
    RelayPlanInfo info = makePlan(epoch, maxCalls);
    fillCounts(epoch, info.nCalls, counts);
    ASSERT_EQ(
        writer.publish(epoch, info, counts.data(), kGenerousTimeoutNs),
        ncclSuccess);
  }
  // The consumer takes epoch 0 and then stops, so the slot epoch 3 needs is
  // still holding an unread epoch 1.
  RelayPlanInfo info{};
  ASSERT_EQ(
      reader.consume(0, &info, got.data(), maxCalls, kGenerousTimeoutNs),
      ncclSuccess);

  // Epoch 2 reuses epoch 0's slot, which has been consumed.
  RelayPlanInfo two = makePlan(2, maxCalls);
  fillCounts(2, two.nCalls, counts);
  ASSERT_EQ(
      writer.publish(2, two, counts.data(), kGenerousTimeoutNs), ncclSuccess);

  // Epoch 3 reuses epoch 1's slot, which has not been. This must block and then
  // fail, never silently overwrite.
  RelayPlanInfo three = makePlan(3, maxCalls);
  fillCounts(3, three.nCalls, counts);
  EXPECT_EQ(
      writer.publish(3, three, counts.data(), kShortTimeoutNs),
      ncclInternalError);
  EXPECT_EQ(writer.abortReason(), static_cast<uint32_t>(kRelayAbortTimeout));
  EXPECT_EQ(writer.abortRank(), 0u);
}

TEST(RelayControlBlockTest, ConsumeTimesOutAndPoisonsTheSegment) {
  const uint64_t hash = uniqueHash(6);
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 4, 8)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, 4, 8)));

  RelayPlanInfo info{};
  std::vector<size_t> counts(8, 0);
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(
      reader.consume(0, &info, counts.data(), 8, kShortTimeoutNs),
      ncclInternalError);
  const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
  // Bounded is the whole point: the store this replaces had a wait() timeout,
  // and removing the store must not remove that property.
  EXPECT_LT(elapsedMs, 5000);
  EXPECT_EQ(reader.abortReason(), static_cast<uint32_t>(kRelayAbortTimeout));
  EXPECT_EQ(reader.abortRank(), 1u);
  // And the failure is visible to everyone else, so one stuck rank produces one
  // attributed cause rather than N independent timeouts.
  EXPECT_EQ(writer.abortReason(), static_cast<uint32_t>(kRelayAbortTimeout));
  EXPECT_EQ(writer.abortRank(), 1u);
}

TEST(RelayControlBlockTest, PeerAbortStopsConsumersImmediately) {
  const uint64_t hash = uniqueHash(7);
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 4, 8)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, 4, 8)));

  writer.setAbort(kRelayAbortCaller);
  EXPECT_EQ(reader.abortReason(), static_cast<uint32_t>(kRelayAbortCaller));
  EXPECT_EQ(reader.abortRank(), 0u);

  RelayPlanInfo info{};
  std::vector<size_t> counts(8, 0);
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(
      reader.consume(0, &info, counts.data(), 8, kGenerousTimeoutNs),
      ncclInternalError);
  // Returns on the abort, not on the 60 s budget.
  EXPECT_LT(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count(),
      1000);
}

TEST(RelayControlBlockTest, RejectsAPlanLargerThanTheSegmentCapacity) {
  const uint64_t hash = uniqueHash(8);
  const uint32_t maxCalls = 4;
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 4, maxCalls)));

  std::vector<size_t> counts(maxCalls + 1, 1);
  RelayPlanInfo info{};
  info.nCalls = maxCalls + 1;
  info.opCode = kRelayOpAllReduce;
  EXPECT_EQ(
      writer.publish(0, info, counts.data(), kGenerousTimeoutNs),
      ncclInvalidArgument);

  // Exactly at capacity is fine: the rejection is > capacity, not >=.
  info.nCalls = maxCalls;
  EXPECT_EQ(
      writer.publish(0, info, counts.data(), kGenerousTimeoutNs), ncclSuccess);
}

TEST(RelayControlBlockTest, ConsumeRejectsATooSmallBufferAndLeavesTheSlotHeld) {
  const uint64_t hash = uniqueHash(9);
  const uint32_t maxCalls = 8;
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 4, maxCalls)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, 4, maxCalls)));

  std::vector<size_t> counts(maxCalls, 0);
  RelayPlanInfo info{};
  info.nCalls = 6;
  info.opCode = kRelayOpAllGather;
  fillCounts(0, info.nCalls, counts);
  ASSERT_EQ(
      writer.publish(0, info, counts.data(), kGenerousTimeoutNs), ncclSuccess);

  RelayPlanInfo got{};
  std::vector<size_t> small(2, 0);
  EXPECT_EQ(
      reader.consume(0, &got, small.data(), 2, kGenerousTimeoutNs),
      ncclInvalidArgument);
  // The caller is told the size it needed, so the error is actionable.
  EXPECT_EQ(got.nCalls, 6u);
  // And the plan was NOT taken, so the publisher must keep the slot reserved.
  EXPECT_EQ(reader.consumerProgress(1), 1u /* registered, nothing completed */);

  // A correctly sized buffer still gets the plan afterwards.
  std::vector<size_t> big(maxCalls, 0);
  EXPECT_EQ(
      reader.consume(0, &got, big.data(), maxCalls, kGenerousTimeoutNs),
      ncclSuccess);
  EXPECT_EQ(got.nCalls, 6u);
  for (uint32_t i = 0; i < 6; i++) {
    EXPECT_EQ(big[i], planCount(0, i)) << "count " << i;
  }
}

TEST(RelayControlBlockTest, DetectsDesyncWhenTheSlotHasAdvancedPastTheEpoch) {
  const uint64_t hash = uniqueHash(10);
  const uint32_t maxCalls = 4;
  const uint32_t ringDepth = 2;
  // nRanks=1: no consumer ever registers, so the publisher is free to run
  // ahead, which is exactly the state a late consumer would find.
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 1, 0, ringDepth, maxCalls)));

  std::vector<size_t> counts(maxCalls);
  for (uint64_t epoch = 0; epoch <= 2; epoch++) {
    RelayPlanInfo info = makePlan(epoch, maxCalls);
    fillCounts(epoch, info.nCalls, counts);
    ASSERT_EQ(
        writer.publish(epoch, info, counts.data(), kGenerousTimeoutNs),
        ncclSuccess);
  }

  // Epoch 0's slot now holds epoch 2. Waiting cannot bring epoch 0 back, so
  // this must fail immediately rather than burn the timeout.
  RelayPlanInfo got{};
  std::vector<size_t> out(maxCalls, 0);
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(
      writer.consume(0, &got, out.data(), maxCalls, kGenerousTimeoutNs),
      ncclInternalError);
  EXPECT_LT(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count(),
      1000);
}

TEST(RelayControlBlockTest, RejectsASecondPublisher) {
  const uint64_t hash = uniqueHash(11);
  const uint32_t maxCalls = 4;
  RelayControlBlock first;
  ASSERT_TRUE(first.create(makeConfig(hash, 4, 0, 4, maxCalls)));
  RelayControlBlock second;
  ASSERT_TRUE(second.attach(makeConfig(hash, 4, 1, 4, maxCalls)));

  std::vector<size_t> counts(maxCalls, 3);
  RelayPlanInfo info{};
  info.nCalls = 1;
  info.opCode = kRelayOpAllReduce;
  ASSERT_EQ(
      first.publish(0, info, counts.data(), kGenerousTimeoutNs), ncclSuccess);
  // One ring means one publisher. A second one would drive the seqlock
  // backwards, and since both plans are byte-identical the damage would be
  // invisible in the data.
  EXPECT_EQ(
      second.publish(1, info, counts.data(), kGenerousTimeoutNs),
      ncclInvalidArgument);
  // The rightful publisher is unaffected.
  EXPECT_EQ(
      first.publish(1, info, counts.data(), kGenerousTimeoutNs), ncclSuccess);
}

/**
 * Test: an epoch that does not advance is refused.
 *
 * waitForSlotDrain() returns immediately for any epoch below ringDepth, so a
 * publisher that restarts its counter walks back over slots holding plans no
 * consumer has taken and the overwrite is silent. Re-publishing an epoch is the
 * same defect seen from the other side: it drives the slot's seqlock backwards
 * from 2e+2 to 2e+1, which a concurrent reader observes as a tear on a slot
 * nobody is tearing.
 *
 * The class already enforces a single publisher so that two ranks cannot race
 * one slot; this is the other half of that invariant -- one publisher cannot
 * race itself across calls.
 */
TEST(RelayControlBlockTest, RejectsAnEpochThatDoesNotAdvance) {
  const uint64_t hash = uniqueHash(21);
  const uint32_t maxCalls = 4;
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 4, maxCalls)));

  std::vector<size_t> counts(maxCalls, 7);
  RelayPlanInfo info{};
  info.nCalls = 1;
  info.opCode = kRelayOpAllReduce;

  ASSERT_EQ(
      writer.publish(0, info, counts.data(), kGenerousTimeoutNs), ncclSuccess);
  // Same epoch again.
  EXPECT_EQ(
      writer.publish(0, info, counts.data(), kGenerousTimeoutNs),
      ncclInvalidArgument);

  ASSERT_EQ(
      writer.publish(1, info, counts.data(), kGenerousTimeoutNs), ncclSuccess);
  // Rewound below what has already been published.
  EXPECT_EQ(
      writer.publish(0, info, counts.data(), kGenerousTimeoutNs),
      ncclInvalidArgument);
  // Advancing still works, so the rejections above are not a wedged publisher.
  EXPECT_EQ(
      writer.publish(2, info, counts.data(), kGenerousTimeoutNs), ncclSuccess);
}

/**
 * Test: a publisher stops once the segment is poisoned.
 *
 * consume() checks the abort flag on every iteration, but publish() only used
 * to observe it indirectly through waitForSlotDrain()'s wait loop -- which is
 * never entered when the slot is already drained. A publisher on a poisoned
 * segment therefore kept filling slots that no consumer would ever read.
 */
TEST(RelayControlBlockTest, PublishStopsOnAPoisonedSegment) {
  const uint64_t hash = uniqueHash(22);
  const uint32_t maxCalls = 4;
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 4, maxCalls)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, 4, maxCalls)));

  std::vector<size_t> counts(maxCalls, 5);
  RelayPlanInfo info{};
  info.nCalls = 1;
  info.opCode = kRelayOpAllReduce;
  ASSERT_EQ(
      writer.publish(0, info, counts.data(), kGenerousTimeoutNs), ncclSuccess);

  // The consumer gives up and poisons the segment for everyone.
  reader.setAbort(kRelayAbortCaller);
  EXPECT_EQ(reader.abortRank(), 1u);

  // Epoch 1's slot is untouched and drained, so without an explicit check this
  // would publish happily.
  EXPECT_EQ(
      writer.publish(1, info, counts.data(), kGenerousTimeoutNs),
      ncclInternalError);
}

TEST(RelayControlBlockTest, RejectsMalformedPlans) {
  const uint64_t hash = uniqueHash(12);
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 4, 8)));
  std::vector<size_t> counts(8, 1);

  RelayPlanInfo badFlags{};
  badFlags.nCalls = 1;
  badFlags.opCode = kRelayOpAllReduce;
  badFlags.flags = 1; // reserved, must be 0 so it can mean something later
  EXPECT_EQ(
      writer.publish(0, badFlags, counts.data(), kGenerousTimeoutNs),
      ncclInvalidArgument);

  RelayPlanInfo badOp{};
  badOp.nCalls = 1;
  badOp.opCode = kRelayOpCount; // one past the last valid opcode
  EXPECT_EQ(
      writer.publish(0, badOp, counts.data(), kGenerousTimeoutNs),
      ncclInvalidArgument);

  RelayPlanInfo nullCounts{};
  nullCounts.nCalls = 1;
  nullCounts.opCode = kRelayOpAllReduce;
  EXPECT_EQ(
      writer.publish(0, nullCounts, nullptr, kGenerousTimeoutNs),
      ncclInvalidArgument);
}

TEST(RelayControlBlockTest, ShutdownPlanRoundTrips) {
  const uint64_t hash = uniqueHash(13);
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 4, 8)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, 4, 8)));

  ASSERT_EQ(writer.publishShutdown(0, kGenerousTimeoutNs), ncclSuccess);
  RelayPlanInfo got{};
  std::vector<size_t> counts(8, 0);
  ASSERT_EQ(
      reader.consume(0, &got, counts.data(), 8, kGenerousTimeoutNs),
      ncclSuccess);
  // Shutdown is an opcode, not a separate entry point, so a graceful stop needs
  // no extra API and cannot be confused with a zero-call forward.
  EXPECT_EQ(got.opCode, static_cast<uint32_t>(kRelayOpShutdown));
  EXPECT_EQ(got.nCalls, 0u);
}

TEST(RelayControlBlockTest, AttachRejectsAGeometryMismatch) {
  const uint64_t hash = uniqueHash(14);
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 8, 0, 4, 128)));

  // Each of these is what a rank launched with a different environment would
  // present. All must fail at attach, because the alternative is reading a
  // differently shaped slot at runtime.
  RelayControlBlock wrongCapacity;
  EXPECT_FALSE(wrongCapacity.attach(makeConfig(hash, 8, 1, 4, 256)));
  RelayControlBlock wrongDepth;
  EXPECT_FALSE(wrongDepth.attach(makeConfig(hash, 8, 1, 8, 128)));
  RelayControlBlock wrongRanks;
  EXPECT_FALSE(wrongRanks.attach(makeConfig(hash, 4, 1, 4, 128)));

  // The matching configuration still works, so the rejections above are
  // specific rather than a segment that was broken all along.
  RelayControlBlock right;
  EXPECT_TRUE(right.attach(makeConfig(hash, 8, 1, 4, 128)));
}

TEST(RelayControlBlockTest, AttachFailsWhenThereIsNoSegment) {
  RelayControlBlock reader;
  EXPECT_FALSE(reader.attach(makeConfig(uniqueHash(15), 8, 1, 4, 128)));
}

TEST(RelayControlBlockTest, RefusesToStealASegmentFromALiveCreator) {
  const uint64_t hash = uniqueHash(16);
  RelayControlBlock first;
  ASSERT_TRUE(first.create(makeConfig(hash, 8, 0, 4, 128)));
  // Same name, and the recorded creator is this very much alive process.
  // Reclaiming here would corrupt a running job.
  RelayControlBlock second;
  EXPECT_FALSE(second.create(makeConfig(hash, 8, 0, 4, 128)));
}

TEST(RelayControlBlockTest, ReclaimsASegmentWhoseCreatorIsGone) {
  const uint64_t hash = uniqueHash(17);
  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    RelayControlBlock leaked;
    const bool ok = leaked.create(makeConfig(hash, 8, 0, 4, 128));
    // _exit skips the destructor, so the segment outlives this process exactly
    // as it would after a crash. That is the state being set up.
    _exit(ok ? kChildOk : kChildCreateFailed);
  }
  // Reap before probing: a zombie still answers kill(pid, 0), so an unreaped
  // child would look alive and the reclaim would (correctly) be refused.
  expectChildOk(pid);

  RelayControlBlock reclaimed;
  EXPECT_TRUE(reclaimed.create(makeConfig(hash, 8, 0, 4, 128)));
}

TEST(RelayControlBlockTest, DetachUnlinksOnlyForTheCreator) {
  const uint64_t hash = uniqueHash(18);
  {
    RelayControlBlock writer;
    ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 4, 8)));
    RelayControlBlock reader;
    ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, 4, 8)));
    reader.detach();
    // A consumer detaching must not remove the segment from under the others.
    RelayControlBlock again;
    EXPECT_TRUE(again.attach(makeConfig(hash, 2, 1, 4, 8)));
  }
  // The creator's destructor unlinked it, so nothing is left behind for the
  // next run to trip over.
  RelayControlBlock afterCreatorGone;
  EXPECT_FALSE(afterCreatorGone.attach(makeConfig(hash, 2, 1, 4, 8)));
}

TEST(RelayControlBlockTest, HandlesALargeNonDefaultCapacity) {
  // The capacity is a runtime parameter precisely so it can be raised for
  // workloads with many calls per forward; 1024 is above the 124-call figure
  // that motivated making it configurable.
  const uint64_t hash = uniqueHash(19);
  const uint32_t maxCalls = 1024;
  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, 2, maxCalls)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, 2, maxCalls)));

  std::vector<size_t> counts(maxCalls);
  RelayPlanInfo info{};
  info.nCalls = maxCalls;
  info.opCode = kRelayOpAllToAll;
  for (uint32_t i = 0; i < maxCalls; i++) {
    counts[i] = planCount(5, i);
  }
  ASSERT_EQ(
      writer.publish(5, info, counts.data(), kGenerousTimeoutNs), ncclSuccess);

  RelayPlanInfo got{};
  std::vector<size_t> out(maxCalls, 0);
  ASSERT_EQ(
      reader.consume(5, &got, out.data(), maxCalls, kGenerousTimeoutNs),
      ncclSuccess);
  ASSERT_EQ(got.nCalls, maxCalls);
  for (uint32_t i = 0; i < maxCalls; i++) {
    ASSERT_EQ(out[i], planCount(5, i)) << "count " << i;
  }
}

TEST(RelayControlBlockTest, ConfiguredGeometryIsClampedToSaneValues) {
  // Defaults, absent any environment override in this binary.
  EXPECT_GE(relayControlConfiguredMaxCalls(), 1u);
  EXPECT_LE(relayControlConfiguredMaxCalls(), 65536u);
  EXPECT_GE(relayControlConfiguredRingDepth(), 2u);
  EXPECT_LE(relayControlConfiguredRingDepth(), 1024u);
}

// Reports the per-operation cost being traded against the ~0.9 ms per call that
// the TCP store transport spent. Not a threshold test -- the harness is shared
// and timing there is not trustworthy enough to gate on -- but the number is
// the whole reason this exists, so it gets printed.
TEST(RelayControlBlockTest, Z_ReportsPublishAndConsumeCost) {
  const uint64_t hash = uniqueHash(20);
  const uint32_t maxCalls = 128;
  const uint32_t ringDepth = 4;
  const int kIters = 2000;

  RelayControlBlock writer;
  ASSERT_TRUE(writer.create(makeConfig(hash, 2, 0, ringDepth, maxCalls)));
  RelayControlBlock reader;
  ASSERT_TRUE(reader.attach(makeConfig(hash, 2, 1, ringDepth, maxCalls)));

  std::vector<size_t> counts(maxCalls);
  std::vector<size_t> out(maxCalls);
  RelayPlanInfo info{};
  info.nCalls = 8; // a realistic chunked-prefill forward
  info.opCode = kRelayOpAllReduce;
  fillCounts(0, info.nCalls, counts);

  double publishNs = 0.0;
  double consumeNs = 0.0;
  for (int i = 0; i < kIters; i++) {
    const uint64_t epoch = static_cast<uint64_t>(i);
    auto t0 = std::chrono::steady_clock::now();
    ASSERT_EQ(
        writer.publish(epoch, info, counts.data(), kGenerousTimeoutNs),
        ncclSuccess);
    auto t1 = std::chrono::steady_clock::now();
    RelayPlanInfo got{};
    ASSERT_EQ(
        reader.consume(epoch, &got, out.data(), maxCalls, kGenerousTimeoutNs),
        ncclSuccess);
    auto t2 = std::chrono::steady_clock::now();
    publishNs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    consumeNs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }
  printf(
      "[relay control] publish %.2f us/plan, consume %.2f us/plan (%d iters, %u calls/plan)\n",
      publishNs / kIters / 1000.0,
      consumeNs / kIters / 1000.0,
      kIters,
      info.nCalls);
  fflush(stdout);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
