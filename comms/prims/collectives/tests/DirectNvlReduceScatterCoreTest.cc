// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "comms/prims/collectives/ReduceScatterDirectNvlCore.cuh"

namespace comms::prims::test {
namespace {

constexpr int expected_rotated_peer(
    int local_rank,
    int local_size,
    std::uint32_t channel,
    std::uint32_t step,
    DirectNvlPeerRole role) {
  const int offset = static_cast<int>(
      (static_cast<std::uint64_t>(channel) + step) %
      static_cast<std::uint32_t>(local_size - 1));
  return role == DirectNvlPeerRole::SEND
      ? (local_rank + local_size - 1 - offset) % local_size
      : (local_rank + 1 + offset) % local_size;
}

TEST(DirectNvlReduceScatterCoreTest, RotatedWalksPairAndCoverEveryPeer) {
  for (int local_size = 2; local_size <= 72; ++local_size) {
    for (std::uint32_t channel_index = 0;
         channel_index < static_cast<std::uint32_t>(local_size);
         ++channel_index) {
      const std::uint32_t channel =
          channel_index == static_cast<std::uint32_t>(local_size - 1)
          ? std::numeric_limits<std::uint32_t>::max()
          : channel_index;
      for (int local_rank = 0; local_rank < local_size; ++local_rank) {
        SCOPED_TRACE(
            ::testing::Message() << "local_size=" << local_size << " channel="
                                 << channel << " local_rank=" << local_rank);
        DirectNvlPeerIterator sends(
            local_rank,
            local_size,
            channel,
            DirectNvlPeerRole::SEND,
            /*rotate_peers=*/true);
        DirectNvlPeerIterator receives(
            local_rank,
            local_size,
            channel,
            DirectNvlPeerRole::RECEIVE,
            /*rotate_peers=*/true);
        std::vector<int> sent(local_size);
        std::vector<int> received(local_size);
        for (int step = 0; step < 2 * (local_size - 1); ++step) {
          const int send_peer = sends.next();
          const int receive_peer = receives.next();
          EXPECT_EQ(
              send_peer,
              expected_rotated_peer(
                  local_rank,
                  local_size,
                  channel,
                  step,
                  DirectNvlPeerRole::SEND));
          EXPECT_EQ(
              receive_peer,
              expected_rotated_peer(
                  local_rank,
                  local_size,
                  channel,
                  step,
                  DirectNvlPeerRole::RECEIVE));
          sent.at(send_peer)++;
          received.at(receive_peer)++;

          EXPECT_EQ(
              expected_rotated_peer(
                  send_peer,
                  local_size,
                  channel,
                  step,
                  DirectNvlPeerRole::RECEIVE),
              local_rank);
        }
        for (int peer = 0; peer < local_size; ++peer) {
          const int expected_count = peer == local_rank ? 0 : 2;
          EXPECT_EQ(sent.at(peer), expected_count);
          EXPECT_EQ(received.at(peer), expected_count);
        }
      }
    }
  }
}

TEST(DirectNvlReduceScatterCoreTest, RankOrderWalkPreservesLegacyOrder) {
  for (int local_size = 2; local_size <= 72; ++local_size) {
    for (const std::uint32_t channel :
         {0U,
          static_cast<std::uint32_t>(local_size - 2),
          std::numeric_limits<std::uint32_t>::max()}) {
      for (int local_rank = 0; local_rank < local_size; ++local_rank) {
        DirectNvlPeerIterator sends(
            local_rank,
            local_size,
            channel,
            DirectNvlPeerRole::SEND,
            /*rotate_peers=*/false);
        DirectNvlPeerIterator receives(
            local_rank,
            local_size,
            channel,
            DirectNvlPeerRole::RECEIVE,
            /*rotate_peers=*/false);
        for (int cycle = 0; cycle < 2; ++cycle) {
          for (int peer = 0; peer < local_size; ++peer) {
            if (peer == local_rank) {
              continue;
            }
            EXPECT_EQ(sends.next(), peer);
            EXPECT_EQ(receives.next(), peer);
          }
        }
      }
    }
  }
}

TEST(DirectNvlReduceScatterCoreTest, OwnerFilteringKeepsEdgesPaired) {
  for (const int local_size : {2, 3, 4, 7, 8, 16, 72}) {
    for (int owner_count = 1; owner_count <= local_size; ++owner_count) {
      for (const bool rotate_peers : {false, true}) {
        for (std::uint32_t channel = 0;
             channel < static_cast<std::uint32_t>(local_size - 1);
             ++channel) {
          std::vector<std::vector<int>> sends(
              local_size, std::vector<int>(local_size));
          std::vector<std::vector<int>> receives(
              local_size, std::vector<int>(local_size));
          for (int local_rank = 0; local_rank < local_size; ++local_rank) {
            DirectNvlPeerIterator send_peers(
                local_rank,
                local_size,
                channel,
                DirectNvlPeerRole::SEND,
                rotate_peers);
            DirectNvlPeerIterator receive_peers(
                local_rank,
                local_size,
                channel,
                DirectNvlPeerRole::RECEIVE,
                rotate_peers);
            const int owner_steps = rotate_peers
                ? local_size - 1
                : owner_count - static_cast<int>(local_rank < owner_count);
            for (int step = 0; step < owner_steps; ++step) {
              const int owner = send_peers.next();
              if (owner < owner_count) {
                sends.at(local_rank).at(owner)++;
              }
            }
            if (local_rank < owner_count) {
              for (int step = 0; step < local_size - 1; ++step) {
                receives.at(receive_peers.next()).at(local_rank)++;
              }
            }
          }
          EXPECT_EQ(sends, receives)
              << "local_size=" << local_size << " owner_count=" << owner_count
              << " channel=" << channel << " rotate_peers=" << rotate_peers;
        }
      }
    }
  }
}

template <std::size_t kDomains, std::size_t kLocalSize>
void expect_node_major_owner_packing() {
  std::vector<bool> source_chunks(kDomains * kLocalSize);
  for (std::size_t domain = 0; domain < kDomains; ++domain) {
    for (std::size_t owner = 0; owner < kLocalSize; ++owner) {
      const std::size_t source =
          DirectNvlNodeMajorOwnerStridedLayout::source_chunk(
              domain, owner, kLocalSize, kDomains);
      EXPECT_EQ(source, domain * kLocalSize + owner);
      ASSERT_LT(source, source_chunks.size());
      EXPECT_FALSE(source_chunks.at(source));
      source_chunks.at(source) = true;
      EXPECT_EQ(
          DirectNvlNodeMajorOwnerStridedLayout::packed_chunk(domain), domain);
    }
  }
  for (const bool visited : source_chunks) {
    EXPECT_TRUE(visited);
  }
}

TEST(DirectNvlReduceScatterCoreTest, NodeMajorTwoByFourPacksByDomain) {
  expect_node_major_owner_packing<2, 4>();
}

TEST(DirectNvlReduceScatterCoreTest, NodeMajorFourByTwoPacksByDomain) {
  expect_node_major_owner_packing<4, 2>();
}

TEST(DirectNvlReduceScatterCoreTest, ContiguousOwnerLayoutGroupsOwnerBatches) {
  constexpr std::size_t kDomains = 4;
  constexpr std::size_t kLocalSize = 2;
  for (std::size_t owner = 0; owner < kLocalSize; ++owner) {
    for (std::size_t domain = 0; domain < kDomains; ++domain) {
      EXPECT_EQ(
          DirectNvlContiguousOwnerLayout::source_chunk(
              domain, owner, kLocalSize, kDomains),
          owner * kDomains + domain);
      EXPECT_EQ(DirectNvlContiguousOwnerLayout::packed_chunk(domain), domain);
    }
  }
}

TEST(DirectNvlReduceScatterCoreTest, LayoutUsesSizeTForLargeChunkIndices) {
  constexpr std::size_t kLocalSize = std::size_t{1} << 20;
  constexpr std::size_t kDomain = (std::size_t{1} << 20) + 3;
  constexpr std::size_t kOwner = kLocalSize - 1;
  constexpr std::size_t kExpected = kDomain * kLocalSize + kOwner;
  static_assert(kExpected > std::numeric_limits<std::uint32_t>::max());
  EXPECT_EQ(
      DirectNvlNodeMajorOwnerStridedLayout::source_chunk(
          kDomain, kOwner, kLocalSize, kDomain + 1),
      kExpected);
}

} // namespace
} // namespace comms::prims::test
