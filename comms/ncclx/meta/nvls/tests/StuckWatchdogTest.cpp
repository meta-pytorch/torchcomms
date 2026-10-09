// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/nvls/StuckWatchdog.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>

using namespace std::chrono_literals;
using ncclx::nvls::StuckWatchdog;

namespace {

// Long enough that a loaded test host still gets several ticks in, short enough
// that the test stays quick. Assertions are lower bounds, never exact counts.
constexpr auto kTick = 25ms;

// Polls instead of sleeping a fixed time, so a slow or sanitized host only
// makes a passing test slower, never flaky.
bool waitForAtLeast(const std::atomic<int>& count, int target) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (count.load() < target) {
    if (std::chrono::steady_clock::now() > deadline) {
      return false;
    }
    std::this_thread::sleep_for(kTick);
  }
  return true;
}

} // namespace

TEST(StuckWatchdogTest, NonPositiveIntervalStartsNoThread) {
  std::atomic<int> stuck{0};
  StuckWatchdog watchdog{0ns, [&] { ++stuck; }};

  EXPECT_FALSE(watchdog.running());
  std::this_thread::sleep_for(4 * kTick);

  EXPECT_TRUE(watchdog.finish());
  EXPECT_EQ(stuck.load(), 0);
}

TEST(StuckWatchdogTest, KeepsReportingAfterTheCallbackThrows) {
  std::atomic<int> stuck{0};
  StuckWatchdog watchdog{kTick, [&] {
                           ++stuck;
                           throw std::runtime_error("callback failed");
                         }};
  ASSERT_TRUE(watchdog.running());

  EXPECT_TRUE(waitForAtLeast(stuck, 3));
  EXPECT_TRUE(watchdog.finish());
}

TEST(StuckWatchdogTest, ReportsRepeatedlyWhileTheCallIsBlocked) {
  std::atomic<int> stuck{0};
  StuckWatchdog watchdog{kTick, [&] { ++stuck; }};
  ASSERT_TRUE(watchdog.running());

  // Stand in for the blocked call.
  EXPECT_TRUE(waitForAtLeast(stuck, 2));

  ASSERT_TRUE(watchdog.finish());
  const int atFinish = stuck.load();
  std::this_thread::sleep_for(4 * kTick);
  EXPECT_EQ(stuck.load(), atFinish);
  EXPECT_FALSE(watchdog.running());
}

TEST(StuckWatchdogTest, FastCallFinishesWithoutWaitingOutTheInterval) {
  std::atomic<int> stuck{0};
  StuckWatchdog watchdog{1h, [&] { ++stuck; }};
  ASSERT_TRUE(watchdog.running());

  const auto start = std::chrono::steady_clock::now();
  const bool joined = watchdog.finish();
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_TRUE(joined);
  EXPECT_LT(elapsed, 10s);
  EXPECT_EQ(stuck.load(), 0);
}

TEST(StuckWatchdogTest, ThreadStartHookRunsOnceOnTheWatchdogThread) {
  std::atomic<int> starts{0};
  std::atomic<std::thread::id> startedOn{};
  StuckWatchdog watchdog{
      kTick,
      [] {},
      [&] {
        ++starts;
        startedOn.store(std::this_thread::get_id());
      }};

  ASSERT_TRUE(watchdog.finish());
  EXPECT_EQ(starts.load(), 1);
  EXPECT_NE(startedOn.load(), std::this_thread::get_id());
}

TEST(StuckWatchdogTest, FinishIsIdempotentAndDestructorNeedsNoCaller) {
  std::atomic<int> stuck{0};
  {
    StuckWatchdog watchdog{kTick, [&] { ++stuck; }};
    EXPECT_TRUE(watchdog.finish());
    EXPECT_TRUE(watchdog.finish());
  }
  const int afterScope = stuck.load();
  std::this_thread::sleep_for(4 * kTick);
  EXPECT_EQ(stuck.load(), afterScope);
}

TEST(StuckWatchdogTest, DestructorStopsAWatchdogThatIsStillWaiting) {
  std::atomic<int> stuck{0};
  {
    StuckWatchdog watchdog{kTick, [&] { ++stuck; }};
    ASSERT_TRUE(waitForAtLeast(stuck, 1));
    // Leaves scope without finish(): the destructor must stop the thread.
  }
  const int afterScope = stuck.load();
  std::this_thread::sleep_for(4 * kTick);
  EXPECT_EQ(stuck.load(), afterScope);
}
