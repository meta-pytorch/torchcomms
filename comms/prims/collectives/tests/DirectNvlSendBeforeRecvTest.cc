// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

#include "comms/prims/collectives/DirectCollectiveUtils.cuh"

namespace comms::prims::test {
namespace {

P2pNvlTransportDevice makePeer(std::size_t windowBytes, std::size_t slotBytes) {
  return P2pNvlTransportDevice(
      /*myRank=*/0,
      /*peerRank=*/1,
      P2pNvlTransportOptions{
          .pipelineDepth = slotBytes == 0 ? 0 : windowBytes / slotBytes,
          .per_channel_buffer = windowBytes,
          .per_channel_slot = slotBytes,
          .max_num_channels = 1,
      },
      LocalState{},
      RemoteState{});
}

TEST(DirectNvlSendBeforeRecvTest, ReservesWorstCaseSignalTail) {
  constexpr std::size_t kWindowBytes = 256;
  constexpr std::size_t kSlotBytes = 128;
  const auto peer = makePeer(kWindowBytes, kSlotBytes);

  // Zero, sub-protocol, and at-least-slot signal sizes use slot granularity.
  EXPECT_EQ(peer.max_payload_without_peer_progress(0), 144);
  EXPECT_EQ(peer.max_payload_without_peer_progress(15), 144);
  EXPECT_EQ(peer.max_payload_without_peer_progress(128), 144);
  EXPECT_EQ(peer.max_payload_without_peer_progress(129), 144);

  // Partial-slot signal sizes are rounded down to the protocol alignment.
  EXPECT_EQ(peer.max_payload_without_peer_progress(16), kWindowBytes);
  EXPECT_EQ(peer.max_payload_without_peer_progress(17), kWindowBytes);
  EXPECT_EQ(peer.max_payload_without_peer_progress(95), 192);
  EXPECT_EQ(peer.max_payload_without_peer_progress(96), 176);
}

TEST(DirectNvlSendBeforeRecvTest, RejectsInvalidChannelGeometry) {
  EXPECT_EQ(makePeer(0, 0).max_payload_without_peer_progress(0), 0);
  EXPECT_EQ(makePeer(255, 128).max_payload_without_peer_progress(0), 0);
  EXPECT_EQ(makePeer(256, 96).max_payload_without_peer_progress(0), 0);
  EXPECT_EQ(makePeer(128, 256).max_payload_without_peer_progress(0), 0);
}

TEST(DirectNvlSendBeforeRecvTest, SelectsMinimumAcrossNonSelfPeers) {
  const std::array peers = {
      makePeer(/*windowBytes=*/0, /*slotBytes=*/0),
      makePeer(/*windowBytes=*/512, /*slotBytes=*/128),
      makePeer(/*windowBytes=*/256, /*slotBytes=*/128),
  };

  EXPECT_EQ(
      direct_nvl_send_before_recv_payload_bytes(
          peers,
          /*my_rank=*/0,
          /*num_ranks=*/3,
          /*max_signal_bytes=*/96),
      176);
  EXPECT_EQ(direct_nvl_send_before_recv_payload_bytes(peers, 0, 1, 96), 0);
  EXPECT_EQ(direct_nvl_send_before_recv_payload_bytes(peers, -1, 3, 96), 0);
  EXPECT_EQ(direct_nvl_send_before_recv_payload_bytes(peers, 3, 3, 96), 0);
}

TEST(DirectNvlSendBeforeRecvTest, RejectsAnyInvalidPeer) {
  const std::array peers = {
      makePeer(/*windowBytes=*/256, /*slotBytes=*/128),
      makePeer(/*windowBytes=*/255, /*slotBytes=*/128),
      makePeer(/*windowBytes=*/512, /*slotBytes=*/128),
  };

  EXPECT_EQ(direct_nvl_send_before_recv_payload_bytes(peers, 0, 3, 96), 0);
}

} // namespace
} // namespace comms::prims::test
