// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/IPersistPlan.h"

namespace {

using ctran::algos::IPersistPlan;
using ctran::algos::PersistPlanKey;

struct TestPlan : public IPersistPlan {
  int value;
  explicit TestPlan(int v) : value(v) {}
};

TEST(PersistPlanConcurrencyTest, SingleThreadCreateAndGet) {
  CtranAlgo algo(CtranAlgo::TestOnly{});
  const auto* p1 = algo.getOrCreatePersistPlan(
      PersistPlanKey::kAllgatherCtsrd,
      []() { return std::make_unique<TestPlan>(42); });
  ASSERT_NE(p1, nullptr);
  EXPECT_EQ(static_cast<const TestPlan*>(p1)->value, 42);

  const auto* p2 =
      algo.getOrCreatePersistPlan(PersistPlanKey::kAllgatherCtsrd, []() {
        ADD_FAILURE() << "createFn should not be called on second get";
        return std::make_unique<TestPlan>(99);
      });
  EXPECT_EQ(p1, p2);
}

TEST(PersistPlanConcurrencyTest, ConcurrentGetOrCreateSameKey) {
  CtranAlgo algo(CtranAlgo::TestOnly{});
  std::atomic<int> createCount{0};
  constexpr int kNumThreads = 32;

  std::vector<const IPersistPlan*> results(kNumThreads, nullptr);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back([&, i]() {
      results[i] =
          algo.getOrCreatePersistPlan(PersistPlanKey::kAllgatherCtsrd, [&]() {
            createCount.fetch_add(1, std::memory_order_relaxed);
            return std::make_unique<TestPlan>(7);
          });
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(createCount.load(), 1) << "createFn must be called exactly once";

  for (int i = 0; i < kNumThreads; i++) {
    ASSERT_NE(results[i], nullptr);
    EXPECT_EQ(results[i], results[0]) << "all threads must get same pointer";
    EXPECT_EQ(static_cast<const TestPlan*>(results[i])->value, 7);
  }
}

TEST(PersistPlanConcurrencyTest, ConcurrentReadAfterCreate) {
  CtranAlgo algo(CtranAlgo::TestOnly{});
  algo.getOrCreatePersistPlan(PersistPlanKey::kAllgatherCtsrd, []() {
    return std::make_unique<TestPlan>(100);
  });

  constexpr int kNumThreads = 64;
  constexpr int kReadsPerThread = 1000;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back([&]() {
      for (int j = 0; j < kReadsPerThread; j++) {
        const auto* p =
            algo.getOrCreatePersistPlan(PersistPlanKey::kAllgatherCtsrd, []() {
              ADD_FAILURE() << "createFn should not be called on read path";
              return std::make_unique<TestPlan>(0);
            });
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(static_cast<const TestPlan*>(p)->value, 100);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

} // namespace
