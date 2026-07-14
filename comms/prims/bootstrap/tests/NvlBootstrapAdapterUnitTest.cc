// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/bootstrap/NvlBootstrapAdapter.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace comms::prims::tests {

namespace {

class RecordingBootstrap final : public meta::comms::IBootstrap {
 public:
  folly::SemiFuture<int> allGather(void* /*buf*/, int, int, int) override {
    return folly::makeSemiFuture(EINVAL);
  }

  folly::SemiFuture<int> barrier(int, int) override {
    return folly::makeSemiFuture(EINVAL);
  }

  folly::SemiFuture<int> allGatherNvlDomain(
      void* buf,
      int len,
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) override {
    ++allGatherNvlDomainCalls;
    lastAllGatherBuf = buf;
    lastAllGatherLen = len;
    lastAllGatherNvlLocalRank = nvlLocalRank;
    lastAllGatherNvlNranks = nvlNranks;
    lastAllGatherNvlRankToCommRank = std::move(nvlRankToCommRank);
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> barrierNvlDomain(
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) override {
    ++barrierNvlDomainCalls;
    lastBarrierNvlLocalRank = nvlLocalRank;
    lastBarrierNvlNranks = nvlNranks;
    lastBarrierNvlRankToCommRank = std::move(nvlRankToCommRank);
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> send(void* /*buf*/, int, int peer, int tag) override {
    ++sendCalls;
    lastSendPeer = peer;
    lastSendTag = tag;
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> recv(void* /*buf*/, int, int peer, int tag) override {
    ++recvCalls;
    lastRecvPeer = peer;
    lastRecvTag = tag;
    return folly::makeSemiFuture(0);
  }

  int allGatherNvlDomainCalls{0};
  void* lastAllGatherBuf{nullptr};
  int lastAllGatherLen{-1};
  int lastAllGatherNvlLocalRank{-1};
  int lastAllGatherNvlNranks{-1};
  std::vector<int> lastAllGatherNvlRankToCommRank;

  int barrierNvlDomainCalls{0};
  int lastBarrierNvlLocalRank{-1};
  int lastBarrierNvlNranks{-1};
  std::vector<int> lastBarrierNvlRankToCommRank;

  int sendCalls{0};
  int lastSendPeer{-1};
  int lastSendTag{-1};

  int recvCalls{0};
  int lastRecvPeer{-1};
  int lastRecvTag{-1};
};

} // namespace

TEST(NvlBootstrapAdapterUnitTest, AllGatherUsesAdapterRankMap) {
  auto underlying = std::make_shared<RecordingBootstrap>();
  NvlBootstrapAdapter adapter(underlying, {12, 7, 19, 3});

  int buffer[4] = {};
  EXPECT_EQ(adapter.allGather(buffer, sizeof(int), 2, 4).get(), 0);

  EXPECT_EQ(underlying->allGatherNvlDomainCalls, 1);
  EXPECT_EQ(underlying->lastAllGatherBuf, buffer);
  EXPECT_EQ(underlying->lastAllGatherLen, sizeof(int));
  EXPECT_EQ(underlying->lastAllGatherNvlLocalRank, 2);
  EXPECT_EQ(underlying->lastAllGatherNvlNranks, 4);
  EXPECT_EQ(
      underlying->lastAllGatherNvlRankToCommRank,
      (std::vector<int>{12, 7, 19, 3}));
}

TEST(NvlBootstrapAdapterUnitTest, NestedAllGatherTranslatesLocalRankMap) {
  auto underlying = std::make_shared<RecordingBootstrap>();
  NvlBootstrapAdapter adapter(underlying, {12, 7, 19, 3});

  int buffer[3] = {};
  EXPECT_EQ(
      adapter
          .allGatherNvlDomain(
              buffer,
              sizeof(int),
              /*nvlLocalRank=*/1,
              /*nvlNranks=*/3,
              /*nvlRankToCommRank=*/{3, 0, 2})
          .get(),
      0);

  EXPECT_EQ(underlying->allGatherNvlDomainCalls, 1);
  EXPECT_EQ(underlying->lastAllGatherBuf, buffer);
  EXPECT_EQ(underlying->lastAllGatherLen, sizeof(int));
  EXPECT_EQ(underlying->lastAllGatherNvlLocalRank, 1);
  EXPECT_EQ(underlying->lastAllGatherNvlNranks, 3);
  EXPECT_EQ(
      underlying->lastAllGatherNvlRankToCommRank,
      (std::vector<int>{3, 12, 19}));
}

TEST(NvlBootstrapAdapterUnitTest, NestedBarrierTranslatesLocalRankMap) {
  auto underlying = std::make_shared<RecordingBootstrap>();
  NvlBootstrapAdapter adapter(underlying, {12, 7, 19, 3});

  EXPECT_EQ(
      adapter
          .barrierNvlDomain(
              /*nvlLocalRank=*/2,
              /*nvlNranks=*/3,
              /*nvlRankToCommRank=*/{1, 3, 0})
          .get(),
      0);

  EXPECT_EQ(underlying->barrierNvlDomainCalls, 1);
  EXPECT_EQ(underlying->lastBarrierNvlLocalRank, 2);
  EXPECT_EQ(underlying->lastBarrierNvlNranks, 3);
  EXPECT_EQ(
      underlying->lastBarrierNvlRankToCommRank, (std::vector<int>{7, 3, 12}));
}

TEST(NvlBootstrapAdapterUnitTest, NestedRankMapRejectsOutOfRangeLocalRank) {
  auto underlying = std::make_shared<RecordingBootstrap>();
  NvlBootstrapAdapter adapter(underlying, {12, 7});

  int buffer[1] = {};
  EXPECT_THROW(
      adapter.allGatherNvlDomain(
          buffer,
          sizeof(int),
          /*nvlLocalRank=*/0,
          /*nvlNranks=*/1,
          /*nvlRankToCommRank=*/{2}),
      std::out_of_range);
  EXPECT_EQ(underlying->allGatherNvlDomainCalls, 0);

  EXPECT_THROW(
      adapter.barrierNvlDomain(
          /*nvlLocalRank=*/0,
          /*nvlNranks=*/1,
          /*nvlRankToCommRank=*/{2}),
      std::out_of_range);
  EXPECT_EQ(underlying->barrierNvlDomainCalls, 0);
}

TEST(NvlBootstrapAdapterUnitTest, SendRecvTranslatePeerRank) {
  auto underlying = std::make_shared<RecordingBootstrap>();
  NvlBootstrapAdapter adapter(underlying, {12, 7, 19, 3});

  int value = 0;
  EXPECT_EQ(
      adapter.send(&value, sizeof(value), /*peer=*/2, /*tag=*/33).get(), 0);
  EXPECT_EQ(underlying->sendCalls, 1);
  EXPECT_EQ(underlying->lastSendPeer, 19);
  EXPECT_EQ(underlying->lastSendTag, 33);

  EXPECT_EQ(
      adapter.recv(&value, sizeof(value), /*peer=*/3, /*tag=*/44).get(), 0);
  EXPECT_EQ(underlying->recvCalls, 1);
  EXPECT_EQ(underlying->lastRecvPeer, 3);
  EXPECT_EQ(underlying->lastRecvTag, 44);
}

} // namespace comms::prims::tests
