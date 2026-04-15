// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/futures/Future.h>

#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/ctran/tests/bootstrap/CtranTestBootstrap.h"

// ============================================================================
// InProcessSendRecvBootstrap — minimal thread-safe fake IBootstrap that only
// provides send/recv via shared memory. Used as the backing transport for
// CtranTestBootstrap, which builds collective operations on top of it.
// ============================================================================

namespace {

struct SharedStore {
  std::mutex mu;
  std::condition_variable cv;
  std::map<std::tuple<int, int, int>, std::vector<char>> store;
};

class InProcessSendRecvBootstrap : public meta::comms::IBootstrap {
 public:
  InProcessSendRecvBootstrap(int rank, std::shared_ptr<SharedStore> store)
      : rank_(rank), store_(std::move(store)) {}

  folly::SemiFuture<int>
  allGather(void* /*buf*/, int /*len*/, int /*rank*/, int /*nranks*/) override {
    return folly::makeSemiFuture(-1);
  }

  folly::SemiFuture<int> barrier(int /*rank*/, int /*nranks*/) override {
    return folly::makeSemiFuture(-1);
  }

  folly::SemiFuture<int> send(void* buf, int len, int peer, int tag) override {
    auto key = std::make_tuple(rank_, peer, tag);
    std::vector<char> data(
        static_cast<char*>(buf), static_cast<char*>(buf) + len);
    {
      std::lock_guard<std::mutex> lk(store_->mu);
      store_->store[key] = std::move(data);
    }
    store_->cv.notify_all();
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> recv(void* buf, int len, int peer, int tag) override {
    auto key = std::make_tuple(peer, rank_, tag);
    std::unique_lock<std::mutex> lk(store_->mu);
    store_->cv.wait(
        lk, [&] { return store_->store.find(key) != store_->store.end(); });
    auto& data = store_->store[key];
    std::memcpy(
        buf, data.data(), std::min(static_cast<size_t>(len), data.size()));
    store_->store.erase(key);
    return folly::makeSemiFuture(0);
  }

  std::unique_ptr<meta::comms::IBootstrap> duplicate() override {
    return std::make_unique<InProcessSendRecvBootstrap>(rank_, store_);
  }

 private:
  int rank_;
  std::shared_ptr<SharedStore> store_;
};

} // namespace

// ============================================================================
// Delegation tests — verify methods delegate to underlying IBootstrap
// ============================================================================

using meta::comms::testing::MockBootstrap;
using ::testing::_;
using ::testing::ByMove;
using ::testing::Return;

TEST(CtranTestBootstrapDelegation, AllGatherDelegates) {
  auto mock = std::make_unique<MockBootstrap>();
  auto* mockPtr = mock.get();

  EXPECT_CALL(*mockPtr, duplicate())
      .WillOnce(Return(ByMove(std::make_unique<MockBootstrap>())));
  EXPECT_CALL(*mockPtr, allGather(_, 10, 0, 2))
      .WillOnce(Return(folly::makeSemiFuture(0)));

  ctran::testing::CtranTestBootstrap uut(std::move(mock));
  std::vector<char> buf(20, 0);
  auto rc = uut.allGather(buf.data(), 10, 0, 2).get();
  EXPECT_EQ(rc, 0);
}

TEST(CtranTestBootstrapDelegation, BarrierDelegates) {
  auto mock = std::make_unique<MockBootstrap>();
  auto* mockPtr = mock.get();

  EXPECT_CALL(*mockPtr, duplicate())
      .WillOnce(Return(ByMove(std::make_unique<MockBootstrap>())));
  EXPECT_CALL(*mockPtr, barrier(1, 3))
      .WillOnce(Return(folly::makeSemiFuture(0)));

  ctran::testing::CtranTestBootstrap uut(std::move(mock));
  auto rc = uut.barrier(1, 3).get();
  EXPECT_EQ(rc, 0);
}

TEST(CtranTestBootstrapDelegation, SendDelegates) {
  auto mock = std::make_unique<MockBootstrap>();
  auto* mockPtr = mock.get();

  EXPECT_CALL(*mockPtr, duplicate())
      .WillOnce(Return(ByMove(std::make_unique<MockBootstrap>())));

  char data[] = "hello";
  EXPECT_CALL(*mockPtr, send(data, 5, 2, 99))
      .WillOnce(Return(folly::makeSemiFuture(0)));

  ctran::testing::CtranTestBootstrap uut(std::move(mock));
  auto rc = uut.send(data, 5, 2, 99).get();
  EXPECT_EQ(rc, 0);
}

TEST(CtranTestBootstrapDelegation, RecvDelegates) {
  auto mock = std::make_unique<MockBootstrap>();
  auto* mockPtr = mock.get();

  EXPECT_CALL(*mockPtr, duplicate())
      .WillOnce(Return(ByMove(std::make_unique<MockBootstrap>())));

  char buf[8] = {};
  EXPECT_CALL(*mockPtr, recv(buf, 8, 3, 42))
      .WillOnce(Return(folly::makeSemiFuture(0)));

  ctran::testing::CtranTestBootstrap uut(std::move(mock));
  auto rc = uut.recv(buf, 8, 3, 42).get();
  EXPECT_EQ(rc, 0);
}

TEST(CtranTestBootstrapDelegation, BroadcastDelegates) {
  auto mock = std::make_unique<MockBootstrap>();
  auto* mockPtr = mock.get();

  EXPECT_CALL(*mockPtr, duplicate())
      .WillOnce(Return(ByMove(std::make_unique<MockBootstrap>())));

  char buf[4] = {};
  EXPECT_CALL(*mockPtr, broadcast(buf, 4, 0, 1, 3))
      .WillOnce(Return(folly::makeSemiFuture(0)));

  ctran::testing::CtranTestBootstrap uut(std::move(mock));
  auto rc = uut.broadcast(buf, 4, 0, 1, 3).get();
  EXPECT_EQ(rc, 0);
}

// ============================================================================
// Error propagation tests
// ============================================================================

TEST(CtranTestBootstrapErrors, SendErrorPropagatesInAllGatherNvlDomain) {
  auto nvlMock = std::make_unique<MockBootstrap>();
  auto* nvlMockPtr = nvlMock.get();

  auto mock = std::make_unique<MockBootstrap>();
  EXPECT_CALL(*mock, duplicate()).WillOnce(Return(ByMove(std::move(nvlMock))));

  EXPECT_CALL(*nvlMockPtr, send(_, _, _, _))
      .WillOnce(Return(folly::makeSemiFuture(42)));

  ctran::testing::CtranTestBootstrap uut(std::move(mock));
  std::vector<char> buf(20, 0);
  // rank 0 (global 0) sends first to rank 1 (global 1) since 0 < 1
  auto rc = uut.allGatherNvlDomain(buf.data(), 10, 0, 2, {0, 1}).get();
  EXPECT_EQ(rc, 42);
}

TEST(CtranTestBootstrapErrors, RecvErrorPropagatesInAllGatherNvlDomain) {
  auto nvlMock = std::make_unique<MockBootstrap>();
  auto* nvlMockPtr = nvlMock.get();

  auto mock = std::make_unique<MockBootstrap>();
  EXPECT_CALL(*mock, duplicate()).WillOnce(Return(ByMove(std::move(nvlMock))));

  // rank 1 (global 1) receives first from rank 0 (global 0) since 1 > 0
  EXPECT_CALL(*nvlMockPtr, recv(_, _, _, _))
      .WillOnce(Return(folly::makeSemiFuture(77)));

  ctran::testing::CtranTestBootstrap uut(std::move(mock));
  std::vector<char> buf(20, 0);
  auto rc = uut.allGatherNvlDomain(buf.data(), 10, 1, 2, {0, 1}).get();
  EXPECT_EQ(rc, 77);
}

// ============================================================================
// allGatherNvlDomain multi-threaded tests
// ============================================================================

namespace {

void runAllGatherNvlDomainTest(
    int nRanks,
    const std::vector<int>& globalRanks) {
  ASSERT_EQ(static_cast<int>(globalRanks.size()), nRanks);

  auto store = std::make_shared<SharedStore>();
  const int len = static_cast<int>(sizeof(int));

  std::vector<std::unique_ptr<ctran::testing::CtranTestBootstrap>> bootstraps(
      nRanks);
  for (int r = 0; r < nRanks; ++r) {
    bootstraps[r] = std::make_unique<ctran::testing::CtranTestBootstrap>(
        std::make_unique<InProcessSendRecvBootstrap>(globalRanks[r], store));
  }

  std::vector<std::vector<int>> results(nRanks, std::vector<int>(nRanks, -1));
  std::vector<std::thread> threads;
  threads.reserve(nRanks);

  for (int r = 0; r < nRanks; ++r) {
    threads.emplace_back([&, r] {
      results[r][r] = globalRanks[r] + 1000;
      auto rc = bootstraps[r]
                    ->allGatherNvlDomain(
                        results[r].data(), len, r, nRanks, globalRanks)
                    .get();
      EXPECT_EQ(rc, 0) << "rank " << r << " failed";
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int r = 0; r < nRanks; ++r) {
    for (int s = 0; s < nRanks; ++s) {
      EXPECT_EQ(results[r][s], globalRanks[s] + 1000)
          << "rank " << r << " expected globalRank[" << s << "]+1000";
    }
  }
}

// ============================================================================
// Send/recv and NVL domain isolation — both use tag 0, must not collide
// ============================================================================

TEST(
    CtranTestBootstrapIsolation,
    DirectSendRecvAndNvlDomainWithSameTagAreIndependent) {
  // This test reproduces the original bug: NVL domain operations and direct
  // send/recv both use tag=0. Without isolation, the second operation would
  // see stale data from the first. With duplicate(), each goes through
  // a different bootstrap so their keys/messages never collide.
  constexpr int kNRanks = 2;
  const std::vector<int> globalRanks = {0, 1};
  const int len = static_cast<int>(sizeof(int));

  auto store = std::make_shared<SharedStore>();

  std::vector<std::unique_ptr<ctran::testing::CtranTestBootstrap>> bootstraps(
      kNRanks);
  for (int r = 0; r < kNRanks; ++r) {
    bootstraps[r] = std::make_unique<ctran::testing::CtranTestBootstrap>(
        std::make_unique<InProcessSendRecvBootstrap>(globalRanks[r], store));
  }

  // Round 1: NVL domain allGather (internally uses tag=0 on the isolated
  // bootstrap).
  std::vector<std::vector<int>> nvlResults(
      kNRanks, std::vector<int>(kNRanks, -1));
  {
    std::vector<std::thread> threads;
    for (int r = 0; r < kNRanks; ++r) {
      threads.emplace_back([&, r] {
        nvlResults[r][r] = r + 100;
        auto rc = bootstraps[r]
                      ->allGatherNvlDomain(
                          nvlResults[r].data(), len, r, kNRanks, globalRanks)
                      .get();
        EXPECT_EQ(rc, 0) << "NVL allGather rank " << r;
      });
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  // Round 2: Direct send/recv with tag=0 — the same tag that NVL domain
  // used internally. Without isolation this would collide.
  constexpr int kDirectTag = 0;
  int sendVal = 42;
  int recvVal = -1;
  {
    std::vector<std::thread> threads;
    // rank 0 sends to rank 1
    threads.emplace_back([&] {
      auto rc = bootstraps[0]->send(&sendVal, len, 1, kDirectTag).get();
      EXPECT_EQ(rc, 0) << "direct send failed";
    });
    // rank 1 receives from rank 0
    threads.emplace_back([&] {
      auto rc = bootstraps[1]->recv(&recvVal, len, 0, kDirectTag).get();
      EXPECT_EQ(rc, 0) << "direct recv failed";
    });
    for (auto& t : threads) {
      t.join();
    }
  }

  // Verify NVL domain results are correct
  for (int r = 0; r < kNRanks; ++r) {
    for (int s = 0; s < kNRanks; ++s) {
      EXPECT_EQ(nvlResults[r][s], s + 100)
          << "NVL result rank " << r << " slot " << s;
    }
  }

  // Verify direct send/recv got the right value (not stale NVL data)
  EXPECT_EQ(recvVal, 42);
}

TEST(
    CtranTestBootstrapIsolation,
    InterleavedDirectSendRecvAndNvlDomainAreIndependent) {
  // Interleave direct send/recv and NVL domain operations multiple times
  // to confirm they never interfere, even when both use tag=0 repeatedly.
  constexpr int kNRanks = 2;
  const std::vector<int> globalRanks = {0, 1};
  const int len = static_cast<int>(sizeof(int));

  auto store = std::make_shared<SharedStore>();

  std::vector<std::unique_ptr<ctran::testing::CtranTestBootstrap>> bootstraps(
      kNRanks);
  for (int r = 0; r < kNRanks; ++r) {
    bootstraps[r] = std::make_unique<ctran::testing::CtranTestBootstrap>(
        std::make_unique<InProcessSendRecvBootstrap>(globalRanks[r], store));
  }

  constexpr int kRounds = 3;
  for (int round = 0; round < kRounds; ++round) {
    // Direct send/recv with tag=0
    int sendVal = round * 10;
    int recvVal = -1;
    {
      std::vector<std::thread> threads;
      threads.emplace_back([&] {
        auto rc = bootstraps[0]->send(&sendVal, len, 1, 0).get();
        EXPECT_EQ(rc, 0);
      });
      threads.emplace_back([&] {
        auto rc = bootstraps[1]->recv(&recvVal, len, 0, 0).get();
        EXPECT_EQ(rc, 0);
      });
      for (auto& t : threads) {
        t.join();
      }
    }
    EXPECT_EQ(recvVal, round * 10) << "direct send/recv round " << round;

    // NVL domain allGather (internally uses tag=0 on isolated bootstrap)
    std::vector<std::vector<int>> nvlResults(
        kNRanks, std::vector<int>(kNRanks, -1));
    {
      std::vector<std::thread> threads;
      for (int r = 0; r < kNRanks; ++r) {
        threads.emplace_back([&, r] {
          nvlResults[r][r] = round * 100 + r;
          auto rc = bootstraps[r]
                        ->allGatherNvlDomain(
                            nvlResults[r].data(), len, r, kNRanks, globalRanks)
                        .get();
          EXPECT_EQ(rc, 0);
        });
      }
      for (auto& t : threads) {
        t.join();
      }
    }
    for (int r = 0; r < kNRanks; ++r) {
      for (int s = 0; s < kNRanks; ++s) {
        EXPECT_EQ(nvlResults[r][s], round * 100 + s)
            << "NVL round " << round << " rank " << r << " slot " << s;
      }
    }
  }
}

} // namespace

TEST(CtranTestBootstrapAllGatherNvlDomain, TwoRanks) {
  runAllGatherNvlDomainTest(2, {0, 1});
}

TEST(CtranTestBootstrapAllGatherNvlDomain, NonContiguousMapping) {
  runAllGatherNvlDomainTest(3, {2, 5, 8});
}

// ============================================================================
// barrierNvlDomain multi-threaded tests
// ============================================================================

namespace {

void runBarrierNvlDomainTest(int nRanks, const std::vector<int>& globalRanks) {
  ASSERT_EQ(static_cast<int>(globalRanks.size()), nRanks);

  auto store = std::make_shared<SharedStore>();

  std::vector<std::unique_ptr<ctran::testing::CtranTestBootstrap>> bootstraps(
      nRanks);
  for (int r = 0; r < nRanks; ++r) {
    bootstraps[r] = std::make_unique<ctran::testing::CtranTestBootstrap>(
        std::make_unique<InProcessSendRecvBootstrap>(globalRanks[r], store));
  }

  std::atomic<int> completed{0};
  std::vector<std::thread> threads;
  threads.reserve(nRanks);

  for (int r = 0; r < nRanks; ++r) {
    threads.emplace_back([&, r] {
      auto rc = bootstraps[r]->barrierNvlDomain(r, nRanks, globalRanks).get();
      EXPECT_EQ(rc, 0) << "rank " << r << " failed";
      completed.fetch_add(1);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(completed.load(), nRanks);
}

} // namespace

TEST(CtranTestBootstrapBarrierNvlDomain, TwoRanks) {
  runBarrierNvlDomainTest(2, {0, 1});
}

TEST(CtranTestBootstrapBarrierNvlDomain, FourRanks) {
  runBarrierNvlDomainTest(4, {0, 1, 2, 3});
}

// ============================================================================
// Sequential tag correctness — two calls don't collide
// ============================================================================

TEST(
    CtranTestBootstrapTagCorrectness,
    SequentialAllGatherNvlDomainTagsIndependent) {
  constexpr int kNRanks = 2;
  const std::vector<int> globalRanks = {0, 1};
  const int len = static_cast<int>(sizeof(int));

  auto store = std::make_shared<SharedStore>();

  std::vector<std::unique_ptr<ctran::testing::CtranTestBootstrap>> bootstraps(
      kNRanks);
  for (int r = 0; r < kNRanks; ++r) {
    bootstraps[r] = std::make_unique<ctran::testing::CtranTestBootstrap>(
        std::make_unique<InProcessSendRecvBootstrap>(globalRanks[r], store));
  }

  // First round
  std::vector<std::vector<int>> results1(
      kNRanks, std::vector<int>(kNRanks, -1));
  {
    std::vector<std::thread> threads;
    for (int r = 0; r < kNRanks; ++r) {
      threads.emplace_back([&, r] {
        results1[r][r] = r + 10;
        auto rc = bootstraps[r]
                      ->allGatherNvlDomain(
                          results1[r].data(), len, r, kNRanks, globalRanks)
                      .get();
        EXPECT_EQ(rc, 0);
      });
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  // Second round with different data
  std::vector<std::vector<int>> results2(
      kNRanks, std::vector<int>(kNRanks, -1));
  {
    std::vector<std::thread> threads;
    for (int r = 0; r < kNRanks; ++r) {
      threads.emplace_back([&, r] {
        results2[r][r] = r + 20;
        auto rc = bootstraps[r]
                      ->allGatherNvlDomain(
                          results2[r].data(), len, r, kNRanks, globalRanks)
                      .get();
        EXPECT_EQ(rc, 0);
      });
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  // Verify both rounds independently correct
  for (int r = 0; r < kNRanks; ++r) {
    for (int s = 0; s < kNRanks; ++s) {
      EXPECT_EQ(results1[r][s], s + 10)
          << "round 1 rank " << r << " slot " << s;
      EXPECT_EQ(results2[r][s], s + 20)
          << "round 2 rank " << r << " slot " << s;
    }
  }
}
