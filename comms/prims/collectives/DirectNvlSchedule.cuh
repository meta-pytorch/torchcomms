// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define PRIMS_DIRECT_NVL_SCHEDULE_HOST_DEVICE __host__ __device__
#else
#define PRIMS_DIRECT_NVL_SCHEDULE_HOST_DEVICE
#endif

namespace comms::prims {

/** Communication direction for one Direct NVLink peer walk. */
enum class DirectNvlPeerRole : std::uint8_t {
  SEND,
  RECEIVE,
};

/**
 * Iterate over every non-self local peer.
 *
 * With rotation disabled, both roles retain the legacy ascending-rank order.
 * With rotation enabled, send and receive walks are complementary for a given
 * channel and step, spreading simultaneous channels over different peers.
 * Callers own the rotation policy because its threshold is algorithm tuning,
 * not a property of the transport. `local_size` must be greater than one;
 * collective entry points handle the single-rank identity before constructing
 * an iterator.
 */
class DirectNvlPeerIterator {
 public:
  PRIMS_DIRECT_NVL_SCHEDULE_HOST_DEVICE constexpr DirectNvlPeerIterator(
      int local_rank,
      int local_size,
      std::uint32_t channel,
      DirectNvlPeerRole role,
      bool rotate_peers)
      : local_rank_(local_rank),
        peer_count_(local_size - 1),
        peer_index_(0),
        peer_step_(1) {
    if (rotate_peers) {
      const std::uint32_t peer_count = static_cast<std::uint32_t>(peer_count_);
      const int channel_offset = static_cast<int>(
          channel < peer_count ? channel : channel % peer_count);
      if (role == DirectNvlPeerRole::SEND) {
        peer_index_ = local_rank - 1 - channel_offset;
        if (peer_index_ < 0) {
          peer_index_ += peer_count_;
        }
        peer_step_ = -1;
      } else {
        peer_index_ = local_rank + channel_offset;
        if (peer_index_ >= peer_count_) {
          peer_index_ -= peer_count_;
        }
      }
    }
  }

  PRIMS_DIRECT_NVL_SCHEDULE_HOST_DEVICE constexpr int next() {
    // Iterate in the dense local_size-1 index space, then insert self again.
    const int peer = peer_index_ + (peer_index_ >= local_rank_);
    peer_index_ += peer_step_;
    if (peer_index_ < 0) {
      peer_index_ += peer_count_;
    } else if (peer_index_ >= peer_count_) {
      peer_index_ -= peer_count_;
    }
    return peer;
  }

 private:
  int local_rank_;
  int peer_count_;
  int peer_index_;
  int peer_step_;
};

/** Owner-major source chunks: all destination-domain chunks for one owner. */
struct DirectNvlContiguousOwnerLayout {
  PRIMS_DIRECT_NVL_SCHEDULE_HOST_DEVICE static constexpr std::size_t
  source_chunk(
      std::size_t destination_domain,
      std::size_t local_owner,
      std::size_t /* local_size */,
      std::size_t num_destination_domains) {
    return local_owner * num_destination_domains + destination_domain;
  }

  PRIMS_DIRECT_NVL_SCHEDULE_HOST_DEVICE static constexpr std::size_t
  packed_chunk(std::size_t destination_domain) {
    return destination_domain;
  }
};

/**
 * Node-major source chunks with owner-strided packing.
 *
 * A local owner `l` collects global chunks `(d * local_size + l)` for every
 * destination domain `d`, and packs them as consecutive chunks `[d]` for the
 * following inter-domain ReduceScatter phase.
 */
struct DirectNvlNodeMajorOwnerStridedLayout {
  PRIMS_DIRECT_NVL_SCHEDULE_HOST_DEVICE static constexpr std::size_t
  source_chunk(
      std::size_t destination_domain,
      std::size_t local_owner,
      std::size_t local_size,
      std::size_t /* num_destination_domains */) {
    return destination_domain * local_size + local_owner;
  }

  PRIMS_DIRECT_NVL_SCHEDULE_HOST_DEVICE static constexpr std::size_t
  packed_chunk(std::size_t destination_domain) {
    return destination_domain;
  }
};

} // namespace comms::prims

#undef PRIMS_DIRECT_NVL_SCHEDULE_HOST_DEVICE
