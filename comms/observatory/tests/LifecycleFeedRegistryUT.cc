// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/observatory/colltrace/LifecycleFeedRegistry.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace meta::comms::colltrace {
namespace {

// A feed standing in for whatever a backend registers. The registry only ever
// calls back through LifecycleFeedOps, so a test needs nothing from colltrace.
//
// Always held by shared_ptr, and `alive` aliases that control block, so the
// token *is* the object the callables run on -- the shape the real registrant
// uses (`std::weak_ptr<void>(tracer)`). A standalone token with callables
// capturing `this` raw would let the header's promise, that a snapshot pins the
// feed "so a drain can request a flush and wait for it without the owner
// retiring in between", be false with no test noticing.
struct FakeFeed {
  std::vector<LifecycleEventRecord> unread;
  int drainCount{0};
  uint64_t commId{0};
  // The ids this feed captured. Anything else it answers empty for, the way a
  // tracer does for a graph it never recorded.
  std::vector<uint64_t> captured;
  int describeCount{0};
  bool throwOnDescribe{false};
  // A feed whose flush never completes -- a poll thread stuck inside a driver
  // call, which is exactly the state a hang investigation drains during.
  bool stallsForever{false};
  // What the drain offered this feed, in the order it was asked.
  std::vector<std::chrono::nanoseconds> budgets;

  LifecycleFeedOps ops(const std::shared_ptr<FakeFeed>& owner) {
    return LifecycleFeedOps{
        .commId = commId,
        // Aliasing: shares the owner's control block, points at this. Dropping
        // the owner expires it, exactly as dropping a tracer does.
        .alive = std::weak_ptr<void>(std::shared_ptr<void>(owner, this)),
        .requestFlush = [] { return uint64_t{0}; },
        .waitFlush =
            [this](uint64_t, std::chrono::nanoseconds remaining) {
              budgets.push_back(remaining);
              return !stallsForever;
            },
        .drainUnread =
            [this] {
              ++drainCount;
              return std::exchange(unread, {});
            },
        .describeCaptured =
            [this](uint64_t collId) -> std::optional<CapturedCollDescription> {
          ++describeCount;
          if (throwOnDescribe) {
            throw std::runtime_error("registrant blew up");
          }
          if (std::find(captured.begin(), captured.end(), collId) ==
              captured.end()) {
            return std::nullopt;
          }
          return CapturedCollDescription{.opName = "AllReduce"};
        },
    };
  }
};

std::shared_ptr<FakeFeed> makeFeed(uint64_t commId = 0) {
  auto feed = std::make_shared<FakeFeed>();
  feed->commId = commId;
  return feed;
}

LifecycleEventRecord eventWithCollId(uint64_t collId) {
  LifecycleEventRecord record;
  record.collId = collId;
  return record;
}

class LifecycleFeedRegistryTest : public ::testing::Test {
 protected:
  // The registry is process-wide by design, so a test that left entries behind
  // would change the next one.
  void SetUp() override {
    resetLifecycleFeedRegistryForTest();
  }
  void TearDown() override {
    resetLifecycleFeedRegistryForTest();
  }
};

TEST_F(LifecycleFeedRegistryTest, SnapshotSeesEveryRegisteredFeed) {
  auto first = makeFeed();
  auto second = makeFeed();
  ASSERT_TRUE(registerLifecycleFeed(first.get(), first->ops(first)));
  ASSERT_TRUE(registerLifecycleFeed(second.get(), second->ops(second)));

  EXPECT_EQ(snapshotLifecycleFeeds().size(), 2);
}

TEST_F(LifecycleFeedRegistryTest, SnapshotDrainsThroughTheRegistrantsCallable) {
  auto feed = makeFeed();
  feed->unread.push_back(eventWithCollId(7));
  ASSERT_TRUE(registerLifecycleFeed(feed.get(), feed->ops(feed)));

  std::vector<LifecycleEventRecord> drained;
  for (const auto& source : snapshotLifecycleFeeds()) {
    const auto generation = source.ops->requestFlush();
    source.ops->waitFlush(generation, std::chrono::seconds{1});
    drained = source.ops->drainUnread();
  }

  const std::vector<LifecycleEventRecord> expected{eventWithCollId(7)};
  EXPECT_EQ(drained, expected);
}

TEST_F(LifecycleFeedRegistryTest, DeregisteredFeedIsGone) {
  auto feed = makeFeed();
  ASSERT_TRUE(registerLifecycleFeed(feed.get(), feed->ops(feed)));

  deregisterLifecycleFeed(feed.get());

  EXPECT_TRUE(snapshotLifecycleFeeds().empty());
  EXPECT_EQ(feed->drainCount, 0);
}

TEST_F(LifecycleFeedRegistryTest, RegisteringAFeedWithNoOwnerIsRefused) {
  // What a registrant that moves its owning shared_ptr before registering
  // rather than after produces. Accepting it would register a feed the first
  // snapshot reaps, after which every drain returns a subset and says nothing.
  auto feed = makeFeed();
  auto ops = feed->ops(feed);
  feed.reset();

  // Sequenced: taking the key and moving the ops in one call leaves the two
  // argument evaluations unordered.
  const void* const owner = &ops;
  EXPECT_FALSE(registerLifecycleFeed(owner, std::move(ops)));
  EXPECT_TRUE(snapshotLifecycleFeeds().empty());
}

TEST_F(LifecycleFeedRegistryTest, DeadOwnerIsReapedWithoutBeingDrained) {
  auto feed = makeFeed();
  auto* key = feed.get();
  ASSERT_TRUE(registerLifecycleFeed(key, feed->ops(feed)));

  // What a caller that dies without deregistering looks like: the ops outlive
  // the owner, so the registry must decide liveness from the token alone.
  feed.reset();

  EXPECT_TRUE(snapshotLifecycleFeeds().empty());
}

TEST_F(LifecycleFeedRegistryTest, OwnerDroppedWhileSnapshotHeldStaysUsable) {
  // The header promises a snapshot pins the feed so a drain can flush and wait
  // without the owner retiring underneath it. That only holds if `alive` shares
  // the owner's control block, which is why the fake aliases it.
  auto feed = makeFeed();
  feed->unread.push_back(eventWithCollId(9));
  ASSERT_TRUE(registerLifecycleFeed(feed.get(), feed->ops(feed)));

  auto sources = snapshotLifecycleFeeds();
  ASSERT_EQ(sources.size(), 1);
  feed.reset();

  // The pin is the only thing keeping the callable's target alive here.
  const std::vector<LifecycleEventRecord> expected{eventWithCollId(9)};
  EXPECT_EQ(sources.front().ops->drainUnread(), expected);
}

TEST_F(LifecycleFeedRegistryTest, ReregisteringAnOwnerReplacesItsFeed) {
  auto original = makeFeed();
  auto replacement = makeFeed();
  replacement->unread.push_back(eventWithCollId(3));
  // One owner, two successive feeds -- a reconfigure replacing a communicator.
  const void* const owner = original.get();
  ASSERT_TRUE(registerLifecycleFeed(owner, original->ops(original)));

  ASSERT_TRUE(registerLifecycleFeed(owner, replacement->ops(replacement)));

  const auto sources = snapshotLifecycleFeeds();
  ASSERT_EQ(sources.size(), 1);
  const std::vector<LifecycleEventRecord> expected{eventWithCollId(3)};
  EXPECT_EQ(sources.front().ops->drainUnread(), expected);
}

TEST_F(LifecycleFeedRegistryTest, FeedsFromDifferentOwnersAllDrain) {
  auto first = makeFeed();
  auto second = makeFeed();
  first->unread.push_back(eventWithCollId(1));
  second->unread.push_back(eventWithCollId(2));
  ASSERT_TRUE(registerLifecycleFeed(first.get(), first->ops(first)));
  ASSERT_TRUE(registerLifecycleFeed(second.get(), second->ops(second)));

  std::vector<LifecycleEventRecord> drained;
  for (const auto& source : snapshotLifecycleFeeds()) {
    auto events = source.ops->drainUnread();
    drained.insert(drained.end(), events.begin(), events.end());
  }

  const std::vector<LifecycleEventRecord> expected{
      eventWithCollId(1), eventWithCollId(2)};
  EXPECT_EQ(drained, expected);
}

TEST_F(LifecycleFeedRegistryTest, DescribesThroughTheFeedStampingTheCommId) {
  auto other = makeFeed(1);
  auto feed = makeFeed(2);
  feed->captured.push_back(11);
  ASSERT_TRUE(registerLifecycleFeed(other.get(), other->ops(other)));
  ASSERT_TRUE(registerLifecycleFeed(feed.get(), feed->ops(feed)));

  const auto described = describeCapturedCollective(2, 11);

  ASSERT_TRUE(described.has_value());
  EXPECT_EQ(described->opName, "AllReduce");
  EXPECT_EQ(other->describeCount, 0) << "a feed stamping another id was asked";
}

TEST_F(LifecycleFeedRegistryTest, DescribeIsEmptyWhenNoFeedStampsTheCommId) {
  auto feed = makeFeed(2);
  feed->captured.push_back(11);
  ASSERT_TRUE(registerLifecycleFeed(feed.get(), feed->ops(feed)));

  EXPECT_FALSE(describeCapturedCollective(3, 11).has_value());
  EXPECT_FALSE(describeCapturedCollective(2, 12).has_value());
}

TEST_F(LifecycleFeedRegistryTest, DescribeAsksEveryFeedStampingTheCommId) {
  // Two feeds can carry one id: a registrant whose plugin was missing stamps
  // the zero default.
  auto silent = makeFeed(5);
  auto holder = makeFeed(5);
  holder->captured.push_back(11);
  ASSERT_TRUE(registerLifecycleFeed(silent.get(), silent->ops(silent)));
  ASSERT_TRUE(registerLifecycleFeed(holder.get(), holder->ops(holder)));

  EXPECT_TRUE(describeCapturedCollective(5, 11).has_value());
  EXPECT_EQ(silent->describeCount, 1);
}

TEST_F(LifecycleFeedRegistryTest, DescribeSkipsAFeedThatCannotAnswer) {
  // What a backend that forgot the callable registers.
  auto incomplete = makeFeed();
  auto holder = makeFeed(5);
  holder->captured.push_back(11);
  auto ops = incomplete->ops(incomplete);
  ops.commId = 5;
  ops.describeCaptured = nullptr;
  ASSERT_TRUE(registerLifecycleFeed(incomplete.get(), std::move(ops)));
  ASSERT_TRUE(registerLifecycleFeed(holder.get(), holder->ops(holder)));

  EXPECT_TRUE(describeCapturedCollective(5, 11).has_value());
}

TEST_F(LifecycleFeedRegistryTest, DescribeSurvivesAFeedThatThrows) {
  // describeCaptured is registrant code and this loop walks every backend in
  // the process. A throw that escaped would skip the feeds behind it, so one
  // backend's bug would take describe away from the other's communicators.
  auto thrower = makeFeed(5);
  auto holder = makeFeed(5);
  thrower->throwOnDescribe = true;
  holder->captured.push_back(11);
  ASSERT_TRUE(registerLifecycleFeed(thrower.get(), thrower->ops(thrower)));
  ASSERT_TRUE(registerLifecycleFeed(holder.get(), holder->ops(holder)));

  EXPECT_TRUE(describeCapturedCollective(5, 11).has_value());
  EXPECT_EQ(thrower->describeCount, 1) << "the throwing feed was not asked";
  // Skipping the feed is the point; skipping it silently is not.
  EXPECT_EQ(lifecycleFeedFailures(), 1u);
}

TEST_F(LifecycleFeedRegistryTest, DuplicateCommIdIsReportedAtRegistration) {
  // Consumers key their own state on the id a feed stamps, so two live feeds
  // sharing one merge into a single identity: their collectives collide and
  // each retires the other's. Answered under the registration lock, so two
  // registrants on different threads cannot both be told there is no collision.
  auto first = makeFeed(4);
  auto second = makeFeed(4);
  auto third = makeFeed(5);

  bool collision = true;
  ASSERT_TRUE(
      registerLifecycleFeed(first.get(), first->ops(first), &collision));
  EXPECT_FALSE(collision) << "the first feed to claim an id cannot collide";

  ASSERT_TRUE(
      registerLifecycleFeed(second.get(), second->ops(second), &collision));
  EXPECT_TRUE(collision);

  ASSERT_TRUE(
      registerLifecycleFeed(third.get(), third->ops(third), &collision));
  EXPECT_FALSE(collision) << "a different id is not a collision";
}

TEST_F(LifecycleFeedRegistryTest, DeadOwnerDoesNotHoldItsCommId) {
  // The entry outlives its owner until a snapshot reaps it. An id it still
  // carries is free, and a registrant reusing it is not colliding with
  // anything.
  auto feed = makeFeed(4);
  ASSERT_TRUE(registerLifecycleFeed(feed.get(), feed->ops(feed)));

  bool collision = false;
  auto other = makeFeed(4);
  ASSERT_TRUE(
      registerLifecycleFeed(other.get(), other->ops(other), &collision));
  EXPECT_TRUE(collision);
  deregisterLifecycleFeed(other.get());

  feed.reset();

  auto later = makeFeed(4);
  ASSERT_TRUE(
      registerLifecycleFeed(later.get(), later->ops(later), &collision));
  EXPECT_FALSE(collision) << "a dead owner does not hold its id";
}

TEST_F(LifecycleFeedRegistryTest, DrainAllMergesEveryBackendInTimestampOrder) {
  // The reason the drain lives here and not in a backend: it walks the whole
  // process. Two feeds registered independently, interleaved in time, come back
  // as one ordered stream.
  using namespace std::chrono_literals;
  const auto base = std::chrono::system_clock::now();
  auto first = makeFeed(1);
  auto second = makeFeed(2);

  auto at = [&](uint64_t collId, std::chrono::milliseconds offset) {
    auto record = eventWithCollId(collId);
    record.timestamp = base + offset;
    return record;
  };
  first->unread = {at(10, 0ms), at(30, 20ms)};
  second->unread = {at(20, 10ms), at(40, 30ms)};
  ASSERT_TRUE(registerLifecycleFeed(first.get(), first->ops(first)));
  ASSERT_TRUE(registerLifecycleFeed(second.get(), second->ops(second)));

  const auto events = drainAllLifecycleEvents();

  std::vector<uint64_t> order;
  order.reserve(events.size());
  for (const auto& event : events) {
    order.push_back(event.collId);
  }
  const std::vector<uint64_t> expected{10, 20, 30, 40};
  EXPECT_EQ(order, expected);
  EXPECT_EQ(first->drainCount, 1);
  EXPECT_EQ(second->drainCount, 1);
}

TEST_F(LifecycleFeedRegistryTest, DrainAllReapsAFeedWhoseOwnerIsGone) {
  auto live = makeFeed(1);
  auto dead = makeFeed(2);
  live->unread = {eventWithCollId(7)};
  ASSERT_TRUE(registerLifecycleFeed(live.get(), live->ops(live)));
  ASSERT_TRUE(registerLifecycleFeed(dead.get(), dead->ops(dead)));
  dead.reset();

  const auto events = drainAllLifecycleEvents();

  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events.front().collId, 7u);
}

TEST_F(LifecycleFeedRegistryTest, DrainAllSurvivesAFeedThatCannotAnswer) {
  // The ops fields are optional -- a backend can register without any of them,
  // the way one without describeCaptured does -- and calling an empty
  // std::function throws. A feed that cannot answer must cost the others
  // nothing.
  auto silent = makeFeed(1);
  auto thrower = makeFeed(2);
  auto holder = makeFeed(3);
  holder->unread = {eventWithCollId(7)};

  auto silentOps = silent->ops(silent);
  silentOps.requestFlush = nullptr;
  silentOps.waitFlush = nullptr;
  silentOps.drainUnread = nullptr;
  ASSERT_TRUE(registerLifecycleFeed(silent.get(), std::move(silentOps)));

  auto throwerOps = thrower->ops(thrower);
  throwerOps.drainUnread = []() -> std::vector<LifecycleEventRecord> {
    throw std::runtime_error("registrant blew up");
  };
  ASSERT_TRUE(registerLifecycleFeed(thrower.get(), std::move(throwerOps)));

  ASSERT_TRUE(registerLifecycleFeed(holder.get(), holder->ops(holder)));

  const auto events = drainAllLifecycleEvents();

  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events.front().collId, 7u);
  // The feed that threw is counted; the one that simply left its callables
  // unset is not, because that is allowed rather than broken. Without the
  // split, a backend registering two fields would look permanently faulty.
  EXPECT_EQ(lifecycleFeedFailures(), 1u);
}

TEST_F(LifecycleFeedRegistryTest, AStalledFeedDoesNotHoldTheDrain) {
  // One feed whose flush never completes must not cost the others their
  // events, and must not cost the caller an unbounded wait. The drain runs at
  // a step boundary and during hang investigations, so blocking here means
  // telemetry stalls the trainer exactly when someone is trying to find out
  // why it is stuck.
  auto stalled = makeFeed(1);
  auto healthy = makeFeed(2);
  stalled->stallsForever = true;
  stalled->unread = {eventWithCollId(1)};
  healthy->unread = {eventWithCollId(2)};
  ASSERT_TRUE(registerLifecycleFeed(stalled.get(), stalled->ops(stalled)));
  ASSERT_TRUE(registerLifecycleFeed(healthy.get(), healthy->ops(healthy)));

  const auto started = std::chrono::steady_clock::now();
  const auto events = drainAllLifecycleEvents(std::chrono::milliseconds{50});
  const auto elapsed = std::chrono::steady_clock::now() - started;

  // Generous: the point is that it returns at all, not that it is prompt.
  EXPECT_LT(elapsed, std::chrono::seconds{5});
  // The stalled feed's queue is still drained. It failed to confirm the flush,
  // which says its events may be incomplete, not that it has none.
  EXPECT_EQ(events.size(), 2u);
  // Counted, so a feed that is late on every drain shows up as a number rather
  // than as everyone else being slow.
  EXPECT_EQ(lifecycleFeedFailures(), 1u);
}

TEST_F(LifecycleFeedRegistryTest, TheBudgetIsSharedAcrossFeedsNotPerFeed) {
  // The waits are sequential, so a per-feed budget would let n feeds block for
  // n times what any one of them was promised. Each feed is offered what is
  // left of one allowance, so the offers never increase.
  auto first = makeFeed(1);
  auto second = makeFeed(2);
  ASSERT_TRUE(registerLifecycleFeed(first.get(), first->ops(first)));
  ASSERT_TRUE(registerLifecycleFeed(second.get(), second->ops(second)));

  constexpr auto kBudget = std::chrono::milliseconds{200};
  drainAllLifecycleEvents(kBudget);

  ASSERT_EQ(first->budgets.size(), 1u);
  ASSERT_EQ(second->budgets.size(), 1u);
  EXPECT_LE(first->budgets.front(), kBudget);
  EXPECT_LE(second->budgets.front(), first->budgets.front())
      << "the second feed was offered a fresh budget rather than the remainder";
}

TEST_F(LifecycleFeedRegistryTest, AHealthyDrainCountsNoFailures) {
  // The counter is only worth reading if a working process leaves it at zero.
  auto feed = makeFeed(1);
  feed->unread = {eventWithCollId(7)};
  ASSERT_TRUE(registerLifecycleFeed(feed.get(), feed->ops(feed)));

  EXPECT_EQ(drainAllLifecycleEvents().size(), 1u);
  EXPECT_EQ(lifecycleFeedFailures(), 0u);
}

TEST_F(LifecycleFeedRegistryTest, RetiredOpsAreDestroyedOutsideTheLock) {
  // A callable's destructor can run arbitrary code. Holding the registry lock
  // across it would deadlock the moment one of them reaches back in here.
  struct ReenteringOnDestroy {
    bool armed{false};
    ~ReenteringOnDestroy() {
      if (armed) {
        snapshotLifecycleFeeds();
      }
    }
  };
  auto feed = makeFeed();
  auto ops = feed->ops(feed);
  auto guard = std::make_shared<ReenteringOnDestroy>();
  guard->armed = true;
  ops.requestFlush = [guard] { return uint64_t{0}; };
  ASSERT_TRUE(registerLifecycleFeed(feed.get(), std::move(ops)));
  guard.reset();

  ASSERT_TRUE(registerLifecycleFeed(feed.get(), feed->ops(feed)));
  deregisterLifecycleFeed(feed.get());

  auto second = feed->ops(feed);
  auto resetGuard = std::make_shared<ReenteringOnDestroy>();
  resetGuard->armed = true;
  second.requestFlush = [resetGuard] { return uint64_t{0}; };
  ASSERT_TRUE(registerLifecycleFeed(feed.get(), std::move(second)));
  resetGuard.reset();
  resetLifecycleFeedRegistryForTest();

  EXPECT_TRUE(snapshotLifecycleFeeds().empty());
}

TEST_F(LifecycleFeedRegistryTest, ConcurrentRegisterSnapshotDeregister) {
  // The reason this library exists is that two backends share one registry in
  // one process, and every test above is single-threaded. The interleaving
  // worth pinning is reap-versus-register: a snapshot erases from `entries`
  // under the lock while another thread registers or deregisters. Under TSAN
  // this is the case that catches a lock that stopped covering the container.
  constexpr int kThreads = 8;
  constexpr int kRounds = 200;
  std::barrier start(kThreads);
  std::atomic<int> refused{0};

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      start.arrive_and_wait();
      for (int i = 0; i < kRounds; ++i) {
        auto feed = makeFeed(static_cast<uint64_t>(t));
        bool collision = false;
        if (!registerLifecycleFeed(feed.get(), feed->ops(feed), &collision)) {
          refused.fetch_add(1, std::memory_order_relaxed);
        }
        for (const auto& source : snapshotLifecycleFeeds()) {
          source.ops->requestFlush();
        }
        // Half the rounds deregister and half just drop the owner, so the
        // reaping branch and the explicit-removal branch both race the others.
        if (i % 2 == 0) {
          deregisterLifecycleFeed(feed.get());
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(refused.load(), 0) << "a live feed was refused";
  // Every owner is gone, so one reaping pass must empty the registry.
  EXPECT_TRUE(snapshotLifecycleFeeds().empty());
}

} // namespace
} // namespace meta::comms::colltrace
