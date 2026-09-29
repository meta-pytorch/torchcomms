// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Transport -> CopyOp feedback: measurements a CopyOp can read from the
// transport driving it.
//
// Knowledge flows one way. The transport exposes measurements and never learns
// what a CopyOp does with them; a CopyOp holds an opaque handle to the
// transport and asks through the queries below. Queries are duck-typed: a
// transport without the accessor (NVL, loopback, the backend-dispatch wrapper)
// reports "no measurement" and the CopyOp keeps its static behaviour.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims {

/**
 * NicSendBacklog - bytes the local NIC still owes on one transport.
 *
 * `pending_bytes` is RDMA payload bytes posted to the NIC minus bytes the NIC
 * reports sent, summed over every channel of the transport, so it describes
 * the link rather than the calling block's queue. It counts actual RDMA write
 * lengths, not staging-slot reservations or flow-control padding.
 *
 * `valid == false` means no measurement is available (no transport, or one
 * that does not track per-put NIC completion).
 */
struct NicSendBacklog {
  std::size_t pending_bytes{0};
  bool valid{false};
};

/** Feedback source for a CopyOp driven outside any transport. */
struct NoTransportFeedback {};

namespace detail {

template <typename T, typename = void>
struct has_nic_send_backlog : std::false_type {};
template <typename T>
struct has_nic_send_backlog<
    T,
    std::void_t<decltype(std::declval<T&>().nic_send_backlog(
        std::declval<ThreadGroup&>()))>> : std::true_type {};

} // namespace detail

/**
 * query_nic_send_backlog - Read a transport's NIC send backlog.
 *
 * Returns an invalid backlog for a null `source` or a type without
 * `nic_send_backlog()`. Otherwise group-collective: every thread of `group`
 * must call it from convergent control flow, and all receive the same value.
 * `source` must be group-uniform so the null check cannot split the group.
 */
template <typename Source>
__device__ __forceinline__ NicSendBacklog
query_nic_send_backlog(Source* source, ThreadGroup& group) {
  if constexpr (detail::has_nic_send_backlog<Source>::value) {
    if (source == nullptr) {
      return NicSendBacklog{};
    }
    return source->nic_send_backlog(group);
  } else {
    (void)source;
    (void)group;
    return NicSendBacklog{};
  }
}

} // namespace comms::prims
