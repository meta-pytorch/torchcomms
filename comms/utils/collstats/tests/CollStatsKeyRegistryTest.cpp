// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsKeyRegistry.h"

#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

namespace meta::comms::collstats {
namespace {

// `id` varies the dtype field so each call yields a distinct key. These tests
// need more distinct keys than the op enum has enumerators, so op and algorithm
// stay fixed and a raw code field carries the variation.
CollStatKey key(uint8_t id, uint8_t sizeClass = 0) {
  return CollStatKey{
      CollStatOp::AllReduce,
      CollStatAlgo::Direct,
      CollStatProto::Unknown,
      id,
      sizeClass};
}

TEST(CollStatsKeyRegistryTest, AssignsDenseIdsInFirstTouchOrder) {
  CollStatsKeyRegistry reg(16);
  EXPECT_EQ(reg.size(), 0u);

  EXPECT_EQ(reg.resolve(key(1)), 0u);
  EXPECT_EQ(reg.resolve(key(2)), 1u);
  EXPECT_EQ(reg.resolve(key(3)), 2u);
  EXPECT_EQ(reg.size(), 3u);
}

TEST(CollStatsKeyRegistryTest, SameKeyKeepsItsId) {
  CollStatsKeyRegistry reg(16);
  const uint32_t id = reg.resolve(key(1, 4));
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(reg.resolve(key(1, 4)), id);
  }
  EXPECT_EQ(reg.size(), 1u);
}

TEST(CollStatsKeyRegistryTest, EveryKeyFieldParticipates) {
  CollStatsKeyRegistry reg(16);
  const CollStatKey baseKey{
      CollStatOp::AllReduce, CollStatAlgo::Direct, CollStatProto::Simple, 1, 1};
  const uint32_t base = reg.resolve(baseKey);

  const auto differs = [&](const CollStatKey& k) {
    EXPECT_NE(reg.resolve(k), base);
  };
  differs(
      {CollStatOp::AllGather,
       CollStatAlgo::Direct,
       CollStatProto::Simple,
       1,
       1});
  differs(
      {CollStatOp::AllReduce, CollStatAlgo::Ring, CollStatProto::Simple, 1, 1});
  differs(
      {CollStatOp::AllReduce, CollStatAlgo::Direct, CollStatProto::LL, 1, 1});
  differs(
      {CollStatOp::AllReduce,
       CollStatAlgo::Direct,
       CollStatProto::Simple,
       2,
       1});
  differs(
      {CollStatOp::AllReduce,
       CollStatAlgo::Direct,
       CollStatProto::Simple,
       1,
       2});
  EXPECT_EQ(reg.size(), 6u);
}

TEST(CollStatsKeyRegistryTest, KeysAreIndexedById) {
  CollStatsKeyRegistry reg(16);
  const uint32_t a = reg.resolve(key(1));
  const uint32_t b = reg.resolve(key(2));

  const auto keys = reg.keys();
  ASSERT_EQ(keys.size(), 2u);
  EXPECT_EQ(keys[a].dtype, 1u);
  EXPECT_EQ(keys[b].dtype, 2u);
}

// Saturation must be visible: overflow keys share the catch-all slot and are
// counted, rather than silently displacing an existing key.
TEST(CollStatsKeyRegistryTest, OverflowGoesToCountedCatchAll) {
  CollStatsKeyRegistry reg(3);
  EXPECT_EQ(reg.resolve(key(1)), 0u);
  EXPECT_EQ(reg.resolve(key(2)), 1u);
  EXPECT_EQ(reg.resolve(key(3)), 2u);
  EXPECT_EQ(reg.catchAllCount(), 0u);

  EXPECT_EQ(reg.resolve(key(4)), reg.catchAllId());
  EXPECT_EQ(reg.resolve(key(5)), reg.catchAllId());
  EXPECT_EQ(reg.catchAllCount(), 2u);

  // Already-assigned keys keep working after saturation.
  EXPECT_EQ(reg.resolve(key(2)), 1u);
  EXPECT_EQ(reg.size(), 3u);
}

TEST(CollStatsKeyRegistryTest, CatchAllIsTheTrailingValueSlot) {
  CollStatsKeyRegistry reg(8);
  EXPECT_EQ(reg.catchAllId(), 8u);
}

// The enqueue thread is the normal caller, but readout attribution reads the
// registry from whichever thread harvested the window.
TEST(CollStatsKeyRegistryTest, ConcurrentResolveAssignsEachKeyOneId) {
  CollStatsKeyRegistry reg(64);
  constexpr int kThreads = 8;
  constexpr int kKeys = 32;

  std::vector<std::vector<uint32_t>> seen(kThreads);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int k = 0; k < kKeys; ++k) {
        seen[t].push_back(reg.resolve(key(static_cast<uint8_t>(k))));
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  // Exactly kKeys ids handed out, and every thread saw the same id per key.
  EXPECT_EQ(reg.size(), static_cast<uint32_t>(kKeys));
  for (int t = 1; t < kThreads; ++t) {
    EXPECT_EQ(seen[t], seen[0]);
  }
  std::unordered_set<uint32_t> unique(seen[0].begin(), seen[0].end());
  EXPECT_EQ(unique.size(), static_cast<size_t>(kKeys));
}

} // namespace
} // namespace meta::comms::collstats
