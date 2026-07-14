// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/collectives/moe_ep/cpp/shared/Config.h"

// `LowLatencyLayout`. Pure host-side buffer-offset calculator; no kernels,
// no transport. Used by both:
//   - `Buffer.get_low_latency_rdma_size_hint` (Python `Buffer.__init__` to
//     pre-size the RDMA buffer)
//   - `LowLatencyRuntime` ctor (to slice the allocated buffer into the
//     send-x / recv-x / signal regions the LL kernels expect)

namespace comms::prims::moe_ep {

/**
 * LowLatencyLayout — computes byte offsets and sizes for the symmetric
 * low-latency buffer used by the LL dispatch / combine kernels.
 *
 * Buffer regions (in offset order):
 *   - `dispatch_send_x`     : per-rank send buffer for tokens being dispatched
 *   - `dispatch_send_count` : per-rank atomic counter for dispatch send slots
 *   - `dispatch_recv_x`     : per-(src-rank × local-expert) receive buffer
 *   - `dispatch_recv_count` : per-(src-rank × local-expert) recv slot count
 *   - `combine_send_x`      : per-(src-rank × local-expert) combine send buffer
 *   - `combine_recv_x`      : per-rank combine recv buffer
 *
 * All offsets are 128-byte aligned (per `NUM_BUFFER_ALIGNMENT_BYTES`).
 */
struct LowLatencyLayout {
  static constexpr std::size_t kBytesPerToken =
      2; // bf16; FP8 is per_byte=1+scale
  static constexpr std::size_t kSlotHeaderBytes = 16; // sizeof(int4) header
  static constexpr std::size_t kSignalSlotBytes = 64;

  std::size_t dispatchSendXOffset{0};
  std::size_t dispatchSendXBytes{0};

  std::size_t dispatchSendCountOffset{0};
  std::size_t dispatchSendCountBytes{0};

  std::size_t dispatchRecvXOffset{0};
  std::size_t dispatchRecvXBytes{0};

  std::size_t dispatchRecvCountOffset{0};
  std::size_t dispatchRecvCountBytes{0};

  std::size_t combineSendXOffset{0};
  std::size_t combineSendXBytes{0};

  std::size_t combineRecvXOffset{0};
  std::size_t combineRecvXBytes{0};

  std::size_t combineRecvFlagOffset{0};
  std::size_t combineRecvFlagBytes{0};

  std::size_t totalBytes{0};

  static std::size_t alignUp(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
  }

  static LowLatencyLayout compute(
      int numMaxDispatchTokensPerRank,
      int hidden,
      int numRanks,
      int numExperts) {
    LowLatencyLayout layout;
    const std::size_t perTokenBytes =
        static_cast<std::size_t>(hidden) * kBytesPerToken;
    // Each slot = int4 header (16B) + hidden bf16 data
    const std::size_t slotBytes = kSlotHeaderBytes + perTokenBytes;
    const int numLocalExperts = numExperts / numRanks;
    const std::size_t numTokens =
        static_cast<std::size_t>(numMaxDispatchTokensPerRank);

    std::size_t cursor = 0;

    // dispatch_send_x: local staging area, one slot per token
    layout.dispatchSendXOffset = cursor;
    layout.dispatchSendXBytes = numTokens * slotBytes;
    cursor =
        alignUp(cursor + layout.dispatchSendXBytes, NUM_BUFFER_ALIGNMENT_BYTES);

    // dispatch_send_count: 1 atomic counter per dest rank
    layout.dispatchSendCountOffset = cursor;
    layout.dispatchSendCountBytes =
        static_cast<std::size_t>(numRanks) * sizeof(std::int64_t);
    cursor = alignUp(
        cursor + layout.dispatchSendCountBytes, NUM_BUFFER_ALIGNMENT_BYTES);

    // dispatch_recv_x: per-local-expert × per-src-rank × tokens
    layout.dispatchRecvXOffset = cursor;
    layout.dispatchRecvXBytes = numTokens * slotBytes *
        static_cast<std::size_t>(numRanks) *
        static_cast<std::size_t>(numLocalExperts);
    cursor =
        alignUp(cursor + layout.dispatchRecvXBytes, NUM_BUFFER_ALIGNMENT_BYTES);

    // dispatch_recv_count: per-local-expert × per-src-rank atomic counter
    layout.dispatchRecvCountOffset = cursor;
    layout.dispatchRecvCountBytes = static_cast<std::size_t>(numRanks) *
        static_cast<std::size_t>(numLocalExperts) * sizeof(std::int64_t);
    cursor = alignUp(
        cursor + layout.dispatchRecvCountBytes, NUM_BUFFER_ALIGNMENT_BYTES);

    // combine_send_x: per-local-expert × per-src-rank × tokens
    layout.combineSendXOffset = cursor;
    layout.combineSendXBytes = numTokens * slotBytes *
        static_cast<std::size_t>(numRanks) *
        static_cast<std::size_t>(numLocalExperts);
    cursor =
        alignUp(cursor + layout.combineSendXBytes, NUM_BUFFER_ALIGNMENT_BYTES);

    // combine_recv_x: per-expert × tokens (peers write by global expert idx)
    layout.combineRecvXOffset = cursor;
    layout.combineRecvXBytes =
        numTokens * slotBytes * static_cast<std::size_t>(numExperts);
    cursor =
        alignUp(cursor + layout.combineRecvXBytes, NUM_BUFFER_ALIGNMENT_BYTES);

    // combine_recv_flag: per-expert completion flag (int64_t each)
    layout.combineRecvFlagOffset = cursor;
    layout.combineRecvFlagBytes =
        static_cast<std::size_t>(numExperts) * sizeof(std::int64_t);
    cursor = alignUp(
        cursor + layout.combineRecvFlagBytes, NUM_BUFFER_ALIGNMENT_BYTES);

    layout.totalBytes = cursor;
    return layout;
  }
};

} // namespace comms::prims::moe_ep
