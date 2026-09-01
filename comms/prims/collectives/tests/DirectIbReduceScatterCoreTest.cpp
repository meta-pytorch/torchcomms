// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "comms/prims/collectives/ReduceScatterDirectIbCore.cuh"

namespace comms::prims::test {
namespace {

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

TEST(DirectIbReduceScatterCoreTest, RoleBoundaryMatchesThreadSplit) {
  EXPECT_EQ(
      direct_ib_reduce_scatter_role_for_thread(0, 384),
      DirectIbReduceScatterRole::RECEIVE);
  EXPECT_EQ(
      direct_ib_reduce_scatter_role_for_thread(383, 384),
      DirectIbReduceScatterRole::RECEIVE);
  EXPECT_EQ(
      direct_ib_reduce_scatter_role_for_thread(384, 384),
      DirectIbReduceScatterRole::SEND);
  EXPECT_EQ(
      direct_ib_reduce_scatter_role_for_thread(511, 384),
      DirectIbReduceScatterRole::SEND);
}

TEST(DirectIbReduceScatterCoreTest, RoleGroupsPreserveLogicalIdentity) {
  const auto make_parent = [](std::uint32_t thread_id) {
    return ThreadGroup{
        .thread_id_in_group = thread_id,
        .group_size = 512,
        .group_id = 7,
        .block_id = 11,
        .total_groups = 13,
        .scope = SyncScope::BLOCK};
  };

  const auto receive =
      make_direct_ib_reduce_scatter_role_group<128, 384, 512>(make_parent(383));
  EXPECT_EQ(receive.role, DirectIbReduceScatterRole::RECEIVE);
  EXPECT_EQ(receive.group.thread_id_in_group, 383);
  EXPECT_EQ(receive.group.group_size, 384);
  EXPECT_EQ(receive.group.group_id, 7);
  EXPECT_EQ(receive.group.block_id, 11);
  EXPECT_EQ(receive.group.total_groups, 13);
  EXPECT_EQ(receive.group.scope, SyncScope::MULTIWARP);
  EXPECT_EQ(receive.group.barrier_id, ThreadGroup::kAutoBarrierId);

  const auto send =
      make_direct_ib_reduce_scatter_role_group<128, 384, 512>(make_parent(384));
  EXPECT_EQ(send.role, DirectIbReduceScatterRole::SEND);
  EXPECT_EQ(send.group.thread_id_in_group, 0);
  EXPECT_EQ(send.group.group_size, 128);
  EXPECT_EQ(send.group.group_id, 7);
  EXPECT_EQ(send.group.block_id, 11);
  EXPECT_EQ(send.group.total_groups, 13);
  EXPECT_EQ(send.group.scope, SyncScope::MULTIWARP);
  EXPECT_EQ(send.group.barrier_id, ThreadGroup::kAutoBarrierId);
}

TEST(DirectIbReduceScatterCoreTest, ExplicitBarriersSupportReverseSplit) {
  const ThreadGroup parent{
      .thread_id_in_group = 128,
      .group_size = 512,
      .group_id = 3,
      .block_id = 5,
      .total_groups = 8,
      .scope = SyncScope::BLOCK};
  const auto send =
      make_direct_ib_reduce_scatter_role_group<384, 128, 512, 1, 2>(parent);
  EXPECT_EQ(send.role, DirectIbReduceScatterRole::SEND);
  EXPECT_EQ(send.group.thread_id_in_group, 0);
  EXPECT_EQ(send.group.group_size, 384);
  EXPECT_EQ(send.group.barrier_id, 2);
}

} // namespace
} // namespace comms::prims::test
