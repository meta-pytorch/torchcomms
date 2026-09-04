// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define PRIMS_DIRECT_IB_RS_HOST_DEVICE_INLINE \
  __host__ __device__ __forceinline__
#else
#define PRIMS_DIRECT_IB_RS_HOST_DEVICE_INLINE inline
#endif

namespace comms::prims {

/** Communication role assigned to a Direct IB ReduceScatter thread group. */
enum class DirectIbReduceScatterRole : std::uint8_t {
  RECEIVE,
  SEND,
};

/**
 * Return the peer for one step of the matched Direct IB ReduceScatter walk.
 *
 * A receiver visits peers in the positive direction while the matching sender
 * visits them in the negative direction. Channel staggering rotates both walks
 * by the same amount, preserving the one-to-one send/receive pairing while
 * spreading channels across peers. The rank and step preconditions are
 * `0 <= my_rank < num_ranks` and `0 <= step < num_ranks - 1`. For the
 * single-rank identity, the function returns `my_rank` without performing
 * modulo arithmetic.
 */
PRIMS_DIRECT_IB_RS_HOST_DEVICE_INLINE constexpr int
direct_ib_reduce_scatter_peer_for_step(
    int my_rank,
    int num_ranks,
    std::uint32_t channel,
    int step,
    DirectIbReduceScatterRole role,
    bool stagger_channels = true) {
  if (num_ranks <= 1) {
    return my_rank;
  }
  const auto peer_count = static_cast<std::uint32_t>(num_ranks - 1);
  const auto peer_offset = stagger_channels
      ? (static_cast<std::uint32_t>(step) + channel) % peer_count
      : static_cast<std::uint32_t>(step);
  return role == DirectIbReduceScatterRole::RECEIVE
      ? (my_rank + 1 + static_cast<int>(peer_offset)) % num_ranks
      : (my_rank + num_ranks - 1 - static_cast<int>(peer_offset)) % num_ranks;
}

} // namespace comms::prims

#undef PRIMS_DIRECT_IB_RS_HOST_DEVICE_INLINE
