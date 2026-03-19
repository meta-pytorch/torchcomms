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
}

// Verify pool auto-grows when exhausted: popping beyond initial capacity
// allocates a new chunk, increasing capacity.
TEST_F(PinnedHostPoolTest, AutoGrowOnExhaustion) {
  constexpr int chunkSize = 4;
  auto pool = std::make_unique<TestItemPool>(chunkSize);

  EXPECT_EQ(pool->size(), chunkSize);
  EXPECT_EQ(pool->capacity(), chunkSize);

  std::vector<TestItem*> items;
  for (int i = 0; i < chunkSize; ++i) {
    auto* item = pool->pop();
    ASSERT_NE(item, nullptr);
    items.push_back(item);
  }
  EXPECT_EQ(pool->size(), 0);
  EXPECT_EQ(pool->capacity(), chunkSize);

  auto* overflow = pool->pop();
  ASSERT_NE(overflow, nullptr);
  EXPECT_EQ(pool->capacity(), chunkSize * 2);
  EXPECT_EQ(pool->size(), chunkSize - 1);

  for (int i = 0; i < chunkSize - 1; ++i) {
    auto* item = pool->pop();
    ASSERT_NE(item, nullptr);
    items.push_back(item);
  }
  EXPECT_EQ(pool->size(), 0);

  auto* overflow2 = pool->pop();
  ASSERT_NE(overflow2, nullptr);
  EXPECT_EQ(pool->capacity(), chunkSize * 3);

  for (auto* item : items) {
    item->inUse_ = false;
  }
  overflow->inUse_ = false;
  overflow2->inUse_ = false;
  pool->reclaim();
  EXPECT_EQ(pool->size(), pool->capacity());
}

// Verify that items marked as not in use are reclaimed automatically on
// subsequent pops, ensuring no pool element leaks without explicit reclaim.
TEST_F(PinnedHostPoolTest, NoLeakAcrossPopReclaimCycles) {
  constexpr int chunkSize = 8;
  auto pool = std::make_unique<TestItemPool>(chunkSize);

  for (int cycle = 0; cycle < 3; ++cycle) {
    std::vector<TestItem*> items;
    for (int i = 0; i < chunkSize; ++i) {
      auto* item = pool->pop();
      ASSERT_NE(item, nullptr);
      items.push_back(item);
    }
    EXPECT_EQ(pool->size(), 0);

    // Mark all as done — do NOT call reclaim explicitly
    for (auto* item : items) {
      item->inUse_ = false;
    }

    // The next pop should trigger an automatic reclaim inside pop(),
    // recovering all items rather than growing the pool.
    auto* next = pool->pop();
    ASSERT_NE(next, nullptr);
    // Capacity should not have grown — reclaim recovered items
    EXPECT_EQ(pool->capacity(), chunkSize);
    // All items minus the one we just popped should be free
    EXPECT_EQ(pool->size(), chunkSize - 1);

    next->inUse_ = false;
  }
}

// Verify that reclaim across grown chunks returns all items correctly and
// that repeated pop/reclaim cycles don't leak elements from any chunk.
TEST_F(PinnedHostPoolTest, NoLeakAcrossChunks) {
  constexpr int chunkSize = 4;
  auto pool = std::make_unique<TestItemPool>(chunkSize);

  // Pop all items to exhaust the first chunk, then trigger growth
  std::vector<TestItem*> items;
  items.reserve(chunkSize + 1);
  for (int i = 0; i < chunkSize + 1; ++i) {
    items.push_back(pool->pop());
  }
  EXPECT_EQ(pool->capacity(), chunkSize * 2);

  // Mark all as done and reclaim
  for (auto* item : items) {
    item->inUse_ = false;
  }
  pool->reclaim();
  EXPECT_EQ(pool->size(), pool->capacity());

  // Pop everything again (spans both chunks) and reclaim
  items.clear();
  const size_t total = pool->size();
  for (size_t i = 0; i < total; ++i) {
    items.push_back(pool->pop());
  }
  EXPECT_EQ(pool->size(), 0);

  for (auto* item : items) {
    item->inUse_ = false;
  }
  pool->reclaim();
  EXPECT_EQ(pool->size(), pool->capacity());
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
