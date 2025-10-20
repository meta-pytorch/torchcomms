// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/utils/LogInit.h"
#include "comms/ctran/utils/PinnedHostPool.h"

struct TestItem {
  static const char* name() {
    return "TestItem";
  }

  void reset() {
    inUse_ = false;
  }

  bool inUse() {
    return inUse_;
  }

  void onPop() {
    inUse_ = true;
  }

  bool inUse_{false};
};

using TestItemPool = PinnedHostPool<TestItem>;

class PinnedHostPoolTest : public ::testing::Test {
 public:
  int cudaDev;
  PinnedHostPoolTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    EXPECT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

    ctran::logging::initCtranLogging();
  }
};

TEST_F(PinnedHostPoolTest, Initialize) {
  constexpr int poolSize = 1000;
  auto pool = std::make_unique<TestItemPool>(poolSize);

  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->size(), poolSize);
  EXPECT_EQ(pool->capacity(), poolSize);
}

TEST_F(PinnedHostPoolTest, PopTest) {
  constexpr int poolSize = 10;
  auto pool = std::make_unique<TestItemPool>(poolSize);

  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->size(), poolSize);
  EXPECT_EQ(pool->capacity(), poolSize);

  for (int i = 0; i < poolSize; ++i) {
    pool->pop();
    EXPECT_EQ(pool->size(), poolSize - (i + 1));
    // Capacity is unchanged
    EXPECT_EQ(pool->capacity(), poolSize);
  }

  auto another_item = pool->pop();
  EXPECT_EQ(another_item, nullptr);
}

TEST_F(PinnedHostPoolTest, ReclaimTest) {
  constexpr int poolSize = 10;
  constexpr int popSize = 6;
  constexpr int reclaimSize = 2;
  auto pool = std::make_unique<TestItemPool>(poolSize);

  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->size(), poolSize);

  std::vector<TestItem*> allocated_items;
  for (int i = 0; i < popSize; ++i) {
    auto item = pool->pop();
    ASSERT_NE(item, nullptr);
    allocated_items.push_back(item);
  }
  EXPECT_EQ(pool->size(), poolSize - popSize);
  // Capacity is unchanged
  EXPECT_EQ(pool->capacity(), poolSize);

  for (int i = 0; i < reclaimSize; ++i) {
    allocated_items[i]->inUse_ = false;
  }
  EXPECT_EQ(pool->size(), poolSize - popSize);
  // Capacity is unchanged
  EXPECT_EQ(pool->capacity(), poolSize);

  pool->reclaim();
  EXPECT_EQ(pool->size(), poolSize - popSize + reclaimSize);
  // Capacity is unchanged
  EXPECT_EQ(pool->capacity(), poolSize);
}
