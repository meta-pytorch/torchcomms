// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

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

/** A Direct IB role together with its cooperative thread group. */
struct DirectIbReduceScatterRoleGroup {
  ThreadGroup group;
  DirectIbReduceScatterRole role;
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
  const int peer_count = num_ranks - 1;
  const int peer_offset = stagger_channels
      ? (step + static_cast<int>(channel % peer_count)) % peer_count
      : step;
  return role == DirectIbReduceScatterRole::RECEIVE
      ? (my_rank + 1 + peer_offset) % num_ranks
      : (my_rank + num_ranks - 1 - peer_offset) % num_ranks;
}

/** Return the communication role for a physical thread within the CTA. */
PRIMS_DIRECT_IB_RS_HOST_DEVICE_INLINE constexpr DirectIbReduceScatterRole
direct_ib_reduce_scatter_role_for_thread(
    std::uint32_t thread_id,
    std::uint32_t receive_threads) {
  return thread_id < receive_threads ? DirectIbReduceScatterRole::RECEIVE
                                     : DirectIbReduceScatterRole::SEND;
}

/**
 * Split a block-wide logical channel into concurrent receive and send roles.
 *
 * The returned groups retain the parent's logical channel identity and
 * physical block identity. This is required by IB transports, whose persistent
 * cursors and QPs are indexed by those identities rather than by the role.
 */
template <
    int kSendThreads,
    int kRecvThreads,
    int kBlockThreads,
    std::uint32_t kRecvBarrierId = ThreadGroup::kAutoBarrierId,
    std::uint32_t kSendBarrierId = ThreadGroup::kAutoBarrierId>
PRIMS_DIRECT_IB_RS_HOST_DEVICE_INLINE DirectIbReduceScatterRoleGroup
make_direct_ib_reduce_scatter_role_group(const ThreadGroup& block_group) {
  static_assert(kSendThreads > 0);
  static_assert(kRecvThreads > 0);
  static_assert(kSendThreads % comms::device::kWarpSize == 0);
  static_assert(kRecvThreads % comms::device::kWarpSize == 0);
  static_assert(kSendThreads + kRecvThreads == kBlockThreads);
  static_assert(
      kSendBarrierId != ThreadGroup::kAutoBarrierId ||
          kRecvThreads % kSendThreads == 0,
      "automatic send barrier requires an aligned send-group start");

  constexpr std::uint32_t kEffectiveRecvBarrierId =
      kRecvBarrierId == ThreadGroup::kAutoBarrierId ? 0 : kRecvBarrierId;
  constexpr std::uint32_t kEffectiveSendBarrierId =
      kSendBarrierId == ThreadGroup::kAutoBarrierId
      ? kRecvThreads / kSendThreads
      : kSendBarrierId;
  static_assert(
      kEffectiveRecvBarrierId != kEffectiveSendBarrierId,
      "Direct IB receive and send roles require distinct named barriers");
  static_assert(
      kEffectiveRecvBarrierId < kMaxMultiwarpsPerBlock &&
          kEffectiveSendBarrierId < kMaxMultiwarpsPerBlock,
      "Direct IB role barrier IDs exceed the hardware range");

  PIPES_DEVICE_CHECK_MSG(
      block_group.scope == SyncScope::BLOCK,
      "Direct IB roles require a block-scoped parent group");
  PIPES_DEVICE_CHECK_MSG(
      block_group.group_size == kBlockThreads,
      "Direct IB parent group size does not match its template geometry");
  PIPES_DEVICE_CHECK_MSG(
      block_group.thread_id_in_group < kBlockThreads,
      "Direct IB parent thread ID is out of bounds");

  const bool is_receive = direct_ib_reduce_scatter_role_for_thread(
                              block_group.thread_id_in_group, kRecvThreads) ==
      DirectIbReduceScatterRole::RECEIVE;
  if (is_receive) {
    return DirectIbReduceScatterRoleGroup{
        .group =
            ThreadGroup{
                .thread_id_in_group = block_group.thread_id_in_group,
                .group_size = kRecvThreads,
                .group_id = block_group.group_id,
                .block_id = block_group.block_id,
                .total_groups = block_group.total_groups,
                .scope = SyncScope::MULTIWARP,
                .barrier_id = kRecvBarrierId},
        .role = DirectIbReduceScatterRole::RECEIVE};
  }

  return DirectIbReduceScatterRoleGroup{
      .group =
          ThreadGroup{
              .thread_id_in_group =
                  block_group.thread_id_in_group - kRecvThreads,
              .group_size = kSendThreads,
              .group_id = block_group.group_id,
              .block_id = block_group.block_id,
              .total_groups = block_group.total_groups,
              .scope = SyncScope::MULTIWARP,
              .barrier_id = kSendBarrierId},
      .role = DirectIbReduceScatterRole::SEND};
}

} // namespace comms::prims

#undef PRIMS_DIRECT_IB_RS_HOST_DEVICE_INLINE
