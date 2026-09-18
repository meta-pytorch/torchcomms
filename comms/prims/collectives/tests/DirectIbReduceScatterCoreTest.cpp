// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "comms/prims/collectives/ReduceScatterDirectIbCore.cuh"

namespace comms::prims::test {
namespace {

int legacy_peer_for_step(
    int rank,
    int num_ranks,
    std::uint32_t channel,
    int step,
    DirectIbReduceScatterRole role,
    bool stagger_channels) {
  const int peer_offset = stagger_channels
      ? static_cast<int>(
            (static_cast<std::uint32_t>(step) + channel) %
            static_cast<std::uint32_t>(num_ranks - 1))
      : step;
  return role == DirectIbReduceScatterRole::RECEIVE
      ? (rank + 1 + peer_offset) % num_ranks
      : (rank + num_ranks - 1 - peer_offset) % num_ranks;
}

void expect_matched_peer_walk(int num_ranks, bool stagger_channels) {
  const std::vector<std::uint32_t> channels{
      0,
      1,
      static_cast<std::uint32_t>(num_ranks - 1),
      static_cast<std::uint32_t>(num_ranks),
      static_cast<std::uint32_t>(2 * num_ranks + 1),
  };
  for (const std::uint32_t channel : channels) {
    for (int rank = 0; rank < num_ranks; ++rank) {
      std::vector<bool> visited(num_ranks, false);
      for (int step = 0; step < num_ranks - 1; ++step) {
        const int peer = direct_ib_reduce_scatter_peer_for_step(
            rank,
            num_ranks,
            channel,
            step,
            DirectIbReduceScatterRole::RECEIVE,
            stagger_channels);
        ASSERT_GE(peer, 0);
        ASSERT_LT(peer, num_ranks);
        EXPECT_NE(peer, rank);
        EXPECT_FALSE(visited[peer]);
        visited[peer] = true;

        EXPECT_EQ(
            direct_ib_reduce_scatter_peer_for_step(
                peer,
                num_ranks,
                channel,
                step,
                DirectIbReduceScatterRole::SEND,
                stagger_channels),
            rank);
      }

      for (int peer = 0; peer < num_ranks; ++peer) {
        EXPECT_EQ(visited[peer], peer != rank);
      }
    }
  }
}

TEST(DirectIbReduceScatterCoreTest, PeerWalkMatchesLegacyOrdering) {
  for (const int num_ranks : {2, 3, 4, 7, 8, 31, 32, 127, 128, 255, 256}) {
    for (std::uint32_t channel = 0; channel < 256; ++channel) {
      for (int rank = 0; rank < num_ranks; ++rank) {
        for (int step = 0; step < num_ranks - 1; ++step) {
          for (const auto role :
               {DirectIbReduceScatterRole::RECEIVE,
                DirectIbReduceScatterRole::SEND}) {
            EXPECT_EQ(
                direct_ib_reduce_scatter_peer_for_step(
                    rank, num_ranks, channel, step, role, true),
                legacy_peer_for_step(
                    rank, num_ranks, channel, step, role, true));
            EXPECT_EQ(
                direct_ib_reduce_scatter_peer_for_step(
                    rank, num_ranks, channel, step, role, false),
                legacy_peer_for_step(
                    rank, num_ranks, channel, step, role, false));
          }
        }
      }
    }
  }
}

TEST(DirectIbReduceScatterCoreTest, StaggeredWalkIsMatchedPermutation) {
  for (const int num_ranks : {2, 3, 4, 7, 8, 31, 32, 127, 128, 255, 256}) {
    expect_matched_peer_walk(num_ranks, true);
  }
}

TEST(DirectIbReduceScatterCoreTest, UnstaggeredWalkIsMatchedPermutation) {
  for (const int num_ranks : {2, 3, 4, 7, 8, 31, 32, 127, 128, 255, 256}) {
    expect_matched_peer_walk(num_ranks, false);
  }
}

TEST(DirectIbReduceScatterCoreTest, SingleRankIdentityReturnsSelf) {
  EXPECT_EQ(
      direct_ib_reduce_scatter_peer_for_step(
          0, 1, 0, 0, DirectIbReduceScatterRole::RECEIVE),
      0);
  EXPECT_EQ(
      direct_ib_reduce_scatter_peer_for_step(
          0, 1, 0, 0, DirectIbReduceScatterRole::SEND),
      0);
}

} // namespace
} // namespace comms::prims::test
