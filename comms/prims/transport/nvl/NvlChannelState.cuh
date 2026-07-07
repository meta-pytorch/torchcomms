// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <cstdint>

#include "comms/prims/core/SignalState.cuh"

namespace comms::prims {

/**
 * NvlChannelState — protocol bookkeeping for one NVLink channel on this rank.
 *
 * One ChannelState per channel per peer. The transport owns an array of these
 * (length = options_.max_num_channels). The data staging buffer is separate;
 * this struct only carries the per-channel cursors and signals.
 *
 *   send_cursor: bytes sent from this local endpoint (monotonic, persistent)
 *   recv_cursor: bytes received by this local endpoint (monotonic, persistent)
 *   data_ready:  bumped by remote sender; local recv waits on this
 *   slot_free:   bumped by remote receiver; local send waits on this
 *
 * Cursors get one local-only cache line (this rank reads/writes; remote never
 * touches them). `data_ready` and `slot_free` each get their own 128-byte
 * cache line. Do not pack both signals into the cursor line: the remote
 * sender and receiver can write them independently via NVLink, and sharing
 * a line would introduce cross-GPU false sharing on the hot wait/signal path.
 *
 * The cursor / signal layout deliberately uses individual fields rather than
 * the prior "step_state[2*N]" / "local_signals[2*N]" half-array convention.
 * Each channel's state is a single struct with named fields; the transport
 * indexes the channels array by group_id directly.
 *
 * Slot-major staging layout: channel c's slice at slot s lives at
 *   staging_base + s * (max_num_channels * per_channel_slot) + c *
 * per_channel_slot (Do not switch to channel-major without re-running the
 * slot-vs-channel-major NVL bandwidth benchmark — slot-major was measured ~5%
 * faster on H100.)
 */
struct alignas(128) NvlChannelState {
  int64_t send_cursor{0};
  int64_t recv_cursor{0};
  SignalState data_ready;
  SignalState slot_free;
};

// 128 (cursors + pad) + 128 (data_ready) + 128 (slot_free) = 384 B.
// Guardrail: if SignalState changes size or someone adds a field below, this
// assert catches the bloat before it lands.
static_assert(
    sizeof(NvlChannelState) == 384,
    "NvlChannelState must be 384 B (one cache line of cursors + one per signal). "
    "If SignalState size changed or a field was added, update the static_assert "
    "and verify no false-sharing across signals.");

static_assert(
    alignof(NvlChannelState) == 128,
    "NvlChannelState must be 128-byte aligned for cache-line isolation.");

} // namespace comms::prims
