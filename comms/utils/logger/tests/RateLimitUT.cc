// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/RateLimit.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using meta::comms::logger::firstNExact;
using meta::comms::logger::IntervalRateLimiter;
using namespace std::chrono_literals;

namespace {

// Distinct tag types give each call site its own counter.
struct TagA {};
struct TagB {};
struct TagConcurrent {};

} // namespace

TEST(FirstNExactTest, ReturnsTrueExactlyNTimes) {
  int trueCount = 0;
  for (int i = 0; i < 10; ++i) {
    if (firstNExact<TagA>(3)) {
      ++trueCount;
    }
  }
  EXPECT_EQ(trueCount, 3);
}

TEST(FirstNExactTest, DistinctTagsHaveIndependentCounters) {
  // Self-contained: exhaust TagB, then show TagC is untouched by it. (Tests run
  // in separate processes, so this must not rely on any other test's state.)
  struct TagC {};
  EXPECT_TRUE(firstNExact<TagB>(1));
  EXPECT_FALSE(firstNExact<TagB>(1));

  EXPECT_TRUE(firstNExact<TagC>(1));
  EXPECT_FALSE(firstNExact<TagC>(1));
}

TEST(FirstNExactTest, ZeroNeverFires) {
  struct TagZero {};
  EXPECT_FALSE(firstNExact<TagZero>(0));
}

TEST(FirstNExactTest, IsExactUnderConcurrency) {
  constexpr uint64_t kLimit = 100;
  constexpr int kThreads = 8;
  constexpr int kPerThread = 500;

  std::atomic<int> trueCount{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < kPerThread; ++i) {
        if (firstNExact<TagConcurrent>(kLimit)) {
          trueCount.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  // "Exact" is the contract: never more, never fewer, than the limit.
  EXPECT_EQ(trueCount.load(), static_cast<int>(kLimit));
}

TEST(IntervalRateLimiterTest, AllowsMaxPerIntervalThenBlocks) {
  IntervalRateLimiter limiter(3, 10s);
  EXPECT_TRUE(limiter.check());
  EXPECT_TRUE(limiter.check());
  EXPECT_TRUE(limiter.check());
  EXPECT_FALSE(limiter.check());
  EXPECT_FALSE(limiter.check());
}

TEST(IntervalRateLimiterTest, ZeroBudgetNeverAdmits) {
  // A zero-duration interval rolls over on every check, exercising both the
  // initial and regular reset paths without relying on wall-clock sleeps.
  IntervalRateLimiter limiter(0, 0ns);
  EXPECT_FALSE(limiter.check());
  EXPECT_FALSE(limiter.check());
  EXPECT_FALSE(limiter.check());
}

TEST(IntervalRateLimiterTest, ResetsAfterIntervalElapses) {
  IntervalRateLimiter limiter(1, 0ns);
  EXPECT_TRUE(limiter.check());
  EXPECT_TRUE(limiter.check());
}

TEST(IntervalRateLimiterTest, NeverExceedsBudgetUnderConcurrency) {
  // A long interval means no reset can occur mid-test, so the total number of
  // successes across all threads must not exceed the budget.
  constexpr uint64_t kBudget = 10;
  constexpr int kThreads = 8;

  IntervalRateLimiter limiter(kBudget, 10s);
  std::atomic<int> allowed{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 100; ++i) {
        if (limiter.check()) {
          allowed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(allowed.load(), static_cast<int>(kBudget));
}
