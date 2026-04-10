// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "comms/uniflow/core/MpscQueue.h"

using namespace uniflow;

class MpscQueueTest : public ::testing::Test {};

// ---------------------------------------------------------------------------
// Fixture: single-threaded functionality
// ---------------------------------------------------------------------------
TEST_F(MpscQueueTest, PushAndPop) {
  MpscQueue<int> q;
  q.push(42);
  auto val = q.pop();
  ASSERT_TRUE(val.has_value());
  EXPECT_EQ(*val, 42);
  EXPECT_FALSE(q.pop().has_value());
}

TEST_F(MpscQueueTest, FIFOOrder) {
  MpscQueue<int> q;
  q.push(1);
  q.push(2);
  q.push(3);

  const std::vector<int> expected{1, 2, 3};
  std::vector<int> actual;
  while (auto val = q.pop()) {
    actual.push_back(*val);
  }
  EXPECT_EQ(actual, expected);
}

TEST_F(MpscQueueTest, EmptyQueue) {
  MpscQueue<int> q;
  EXPECT_FALSE(q.pop().has_value());
}

TEST_F(MpscQueueTest, MoveOnlyType) {
  MpscQueue<std::unique_ptr<int>> q;
  q.push(std::make_unique<int>(99));
  auto val = q.pop();
  ASSERT_TRUE(val.has_value());
  EXPECT_EQ(**val, 99);
  EXPECT_FALSE(q.pop().has_value());
}

TEST_F(MpscQueueTest, DestructorCleanup) {
  struct Counted {
    static std::atomic<int>& alive() {
      static std::atomic<int> count{0};
      return count;
    }

    int value;

    Counted() : value(0) {
      alive().fetch_add(1, std::memory_order_relaxed);
    }
    explicit Counted(int v) : value(v) {
      alive().fetch_add(1, std::memory_order_relaxed);
    }
    Counted(const Counted& o) : value(o.value) {
      alive().fetch_add(1, std::memory_order_relaxed);
    }
    Counted(Counted&& o) noexcept : value(o.value) {
      alive().fetch_add(1, std::memory_order_relaxed);
    }
    Counted& operator=(const Counted&) = default;
    Counted& operator=(Counted&&) noexcept = default;
    ~Counted() {
      alive().fetch_sub(1, std::memory_order_relaxed);
    }
  };

  Counted::alive().store(0, std::memory_order_relaxed);

  {
    MpscQueue<Counted> q;
    // Stub has a default-constructed Counted
    EXPECT_EQ(Counted::alive().load(std::memory_order_relaxed), 1);

    q.push(Counted{1});
    q.push(Counted{2});
    q.push(Counted{3});
  }
  EXPECT_EQ(Counted::alive().load(std::memory_order_relaxed), 0);

  Counted::alive().store(0, std::memory_order_relaxed);

  {
    MpscQueue<Counted> q;
    q.push(Counted{4});
    q.push(Counted{5});
    // Pop one, let destructor handle the rest
    q.pop();
  }
  EXPECT_EQ(Counted::alive().load(std::memory_order_relaxed), 0);
}

// ---------------------------------------------------------------------------
// Fixture: concurrent functionality
// ---------------------------------------------------------------------------
class MpscQueueConcurrentTest : public MpscQueueTest {
 protected:
  void SetUp() override {
    nThreads_ = std::max(8, available_concurrency());
    nItemsPerThread_ = 10000;
    threads_.reserve(nThreads_);
  }
  void TearDown() override {
    for (auto& thr : threads_) {
      if (thr.joinable()) {
        thr.join();
      }
    }
  }
  std::vector<std::thread> threads_;
  int nThreads_;
  int nItemsPerThread_;

  int available_concurrency() {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof(cs), &cs) == 0) {
      return CPU_COUNT(&cs);
    }
    // NOLINTNEXTLINE(facebook-avoid-std-thread-hardware-concurrency)
    return std::thread::hardware_concurrency();
  }

  void runPushOrderTest() {
    struct Item {
      int threadId;
      int seq;
    };

    MpscQueue<Item> q;
    std::atomic<int> done{0};

    for (int t = 0; t < nThreads_; ++t) {
      threads_.emplace_back([this, &q, &done, t]() {
        for (int i = 0; i < nItemsPerThread_; ++i) {
          q.push(Item{t, i});
        }
        done.fetch_add(1, std::memory_order_release);
      });
    }

    std::vector<int> lastSeen(nThreads_, -1);
    int count = 0;
    auto pollAndCheck = [&q, &count, &lastSeen]() {
      while (auto item = q.pop()) {
        EXPECT_GT(item->seq, lastSeen[item->threadId])
            << "Out-of-order for thread " << item->threadId;
        lastSeen[item->threadId] = item->seq;
        ++count;
      }
    };
    while (done.load(std::memory_order_acquire) < nThreads_) {
      pollAndCheck();
    }

    // Drain the queue.
    pollAndCheck();
    EXPECT_EQ(count, nThreads_ * nItemsPerThread_);
  }
};

TEST_F(MpscQueueConcurrentTest, ConcurrentPush) {
  MpscQueue<int> q;
  std::atomic<int> done{0};

  for (int t = 0; t < nThreads_; ++t) {
    threads_.emplace_back([this, &q, &done, t]() {
      for (int i = 0; i < nItemsPerThread_; ++i) {
        q.push(t * nItemsPerThread_ + i);
      }
      done.fetch_add(1, std::memory_order_release);
    });
  }

  int count = 0;
  auto poll = [&q, &count]() {
    while (q.pop().has_value()) {
      ++count;
    }
  };
  while (done.load(std::memory_order_acquire) < nThreads_) {
    poll();
  }

  // Drain the queue.
  poll();

  EXPECT_EQ(count, nThreads_ * nItemsPerThread_);
}

TEST_F(MpscQueueConcurrentTest, ConcurrentPushOrder) {
  runPushOrderTest();
}

TEST_F(MpscQueueConcurrentTest, HighContention) {
  nThreads_ = std::max(32, available_concurrency());
  nItemsPerThread_ = 100000;
  threads_.reserve(nThreads_);
  runPushOrderTest();
}
