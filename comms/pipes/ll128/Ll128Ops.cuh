// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// @lint-ignore-every CLANGTIDY facebook-modularize-issue-check

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/common/AtomicUtils.cuh"
#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/ll128/Ll128Packet.cuh"

namespace comms::pipes {

// =============================================================================
// LL128 Send/Recv/Forward Operations
// =============================================================================
//
// Low-level LL128 operations for broadcast-style communication.
//
// THREAD ORGANIZATION:
//   - Warp-based (32 threads per warp)
//   - Each warp is split into 4 sub-warp groups of 8 threads
//   - Each 8-thread group handles one 128-byte LL128 packet
//   - One warp handles 4 packets (480 bytes of payload) per iteration
//
// SUB-WARP INDEXING (within each group's ThreadGroup):
//   group_idx      = thread_id_in_group / 8   (which of the 4 groups: 0-3)
//   lane_in_group  = thread_id_in_group % 8   (position within group: 0-7)
//
// THREAD-TO-DATA MAPPING (within 8-thread group):
//   Threads 0-6: write data[0] through data[6] — 16B pure payload each
//   Thread 7:    writes data[7] — 8B payload + 8B flag
//
// MEMORY ORDERING:
//   All operations use volatile load/store (bypass L1, NVLink-visible).
//   Cache-line atomicity guarantees the receiver sees either all-old or
//   all-new data when 8 threads write to the same 128B-aligned address.
//
// COOPERATIVE POLLING:
//   Flag polls use __shfl_sync to broadcast lane 7's poll result to all
//   lanes in the subgroup, keeping all 8 threads converged at the same PC.
//   This avoids divergent-thread reconvergence issues when __syncwarp
//   must bridge a long-running poll loop (e.g., during buffer wrapping).
//
// WARP CLAMPING (when buffer_num_packets > 0):
//   When the LL128 buffer is smaller than the message, at most
//   `min(buf_packets / kLl128PacketsPerWarp, total_groups)` warps are
//   active per iteration.
//   Active warps loop over the message using modular buffer indexing
//   (pkt_idx % aligned_buf_packets) and per-packet flag values to
//   prevent ABA issues. Flag management is internal — callers do not
//   supply a flag.
//   Inactive warps skip all work — they must NOT touch the buffer.
//
// CHUNKING / WINDOWED BUFFER PROTOCOL:
//   When `buffer_num_packets > 0` and `buffer_num_packets < total_packets`,
//   the buffer is smaller than the message. The sender and receiver loop
//   through the message in multiple rounds, reusing buffer slots via modular
//   indexing (`pkt_idx % aligned_buf_packets`). `aligned_buf_packets`
//   is `buf_packets` rounded down to a multiple of the warp stride
//   (see WARP SLOT OWNERSHIP below).
//
//   Flow control: The sender polls for `kLl128ReadyToWrite` (-1) before
//   overwriting each slot. The receiver reads the data, then ACKs by writing
//   -1 back to the flag. This creates back-pressure; the sender cannot
//   outrun the receiver.
//
//   ABA prevention within a step: Each buffer slot is reused across rounds.
//   To prevent the receiver from mistaking a previous round's data (still in
//   the same buffer slot) for the current round's data, the sender stamps
//   each packet with a monotonically increasing flag value:
//   `flag_value + (pkt_idx / aligned_buf_packets)`. The receiver polls
//   for exactly this value. For example, with `flag_value = 1`: round 0 uses
//   flag 1, round 1 uses flag 2, round 2 uses flag 3, etc. The receiver waiting
//   for flag 2 cannot be satisfied by round 0's leftover flag 1.
//
//   Sender/receiver flag asymmetry: The sender always polls for the fixed
//   value -1 (slot is free). The receiver polls for the round-specific
//   `pkt_flag_value` (data is ready for this round). This asymmetry is what
//   makes the protocol work; -1 can never collide with any valid
//   `pkt_flag_value` (which is always >= 1).
//
//   Two-state invariant: Flags only ever hold two categories of values:
//   -1 (free / ready-to-write) or a positive integer (data ready for a
//   specific round). This clean separation prevents all ABA scenarios.
//
// WARP SLOT OWNERSHIP:
//   Exclusive slot ownership in chunked mode requires two mechanisms:
//
//   1. Warp clamping: `active_warps = min(buf_packets / kLl128PacketsPerWarp,
//      total_groups)` limits active warps so the buffer has enough capacity
//      for one iteration's worth of work. Inactive warps
//      (`group_id >= active_warps`) skip all work entirely.
//
//   2. Stride alignment: `aligned_buf_packets` is `buf_packets` rounded
//      down to a multiple of `stride = active_warps * kLl128PacketsPerWarp`.
//      This ensures `pkt_idx % aligned_buf_packets` maps each buf_idx to
//      exactly one (warp, sub-group) pair across all loop iterations.
//      Without alignment, when `buf_packets % stride != 0`, different
//      warps in different iterations can map to the same buf_idx and
//      race on the slot.
//
// MULTI-STEP USAGE:
//   `flag_value` is hardcoded to 1 in all three operations. Multi-step usage
//   (calling the ops repeatedly on the same buffer) works without
//   incrementing `flag_value` because the receiver ACKs every slot with -1
//   after reading. This resets the buffer state machine between steps:
//     Step N:   sender writes pkt_flag_value >= 1, receiver reads, ACKs with -1
//     Step N+1: sender waits for -1 (sees ACK), writes pkt_flag_value >= 1
//     again
//   Within a step, increasing per-round flag values (1, 2, 3, ...) prevent
//   ABA between chunking rounds. Between steps, the -1 ACK reset prevents
//   ABA because the sender blocks until -1 is observed before writing the
//   next step.

/**
 * ll128_send — Send data to a remote LL128 buffer.
 *
 * The sender reads user data from a local source buffer, packs it into
 * LL128 packets, and volatile-stores each 128B packet to the remote
 * (receiver's) LL128 buffer with the step ID embedded as the flag.
 *
 * @param group     ThreadGroup (auto-converted to warp scope via
 * to_warp_group())
 * @param src       Local source buffer (user data, contiguous, 16-byte aligned)
 * @param nbytes    Total message size in bytes (must be a multiple of 16)
 * @param remote_ll128_buf  Pointer to receiver's LL128 packet buffer
 * @param timeout   Timeout for flag polling
 * @param buffer_num_packets  Number of packets in the LL128 buffer.
 *                            0 = buffer is pre-sized to fit the entire message
 *                            (no chunking). >0 and < total packets =
 *                            windowed/chunked mode. Must be >=
 *                            kLl128PacketsPerWarp (4) when chunking.
 */
__device__ __forceinline__ void ll128_send(
    const ThreadGroup& group,
    const char* __restrict__ src,
    size_t nbytes,
    Ll128Packet* __restrict__ remote_ll128_buf,
    const Timeout& timeout,
    size_t buffer_num_packets = 0) {
#ifdef __CUDA_ARCH__
  // Constant base flag. Multi-step works via receiver ACK (-1) reset between
  // steps.
  const int64_t flag_value = 1;
  auto warp = group.to_warp_group();

  if (nbytes == 0) {
    return;
  }

  PIPES_DEVICE_CHECK(can_use_ll128(src, nbytes));

  const int group_idx = warp.thread_id_in_group / kLl128ThreadsPerPacket;
  const int lane_in_group = warp.thread_id_in_group % kLl128ThreadsPerPacket;

  // Subgroup mask: only the 8 threads handling the same packet need to
  // coordinate. All 8 threads share the same active/pkt_idx state.
  const unsigned int subgroup_mask = 0xFFu
      << (group_idx * kLl128ThreadsPerPacket);

  // Warp-level lane of the flag-owning thread in this subgroup, used as
  // the source lane for __shfl_sync broadcasts during cooperative polling.
  const int flag_src_lane = group_idx * kLl128ThreadsPerPacket + kLl128FlagLane;

  const size_t total_packets = ll128_num_packets(nbytes);

  // Compute effective buffer size in packets
  const size_t buf_packets =
      (buffer_num_packets > 0 && buffer_num_packets < total_packets)
      ? buffer_num_packets
      : total_packets;

  // Runtime guard: buffer must hold at least one warp's worth of packets
  PIPES_DEVICE_CHECK(
      buf_packets >= total_packets || buf_packets >= kLl128PacketsPerWarp);

  // Warp clamping: use the lesser of buffer capacity and available warps
  const size_t buf_warps = buf_packets / kLl128PacketsPerWarp;
  const size_t active_warps =
      (buf_packets < total_packets && buf_warps < warp.total_groups)
      ? buf_warps
      : warp.total_groups;

  // Align buf_packets down to a multiple of the warp stride to prevent
  // cross-warp races on the same buf_idx. Without this, when
  // buf_packets % stride != 0, different warps in different loop iterations
  // can map to the same buf_idx and race on the ACK poll + write.
  const size_t stride = active_warps * kLl128PacketsPerWarp;
  const size_t aligned_buf_packets =
      (buf_packets < total_packets && buf_packets > stride)
      ? (buf_packets / stride) * stride
      : buf_packets;

  for (size_t base = warp.group_id * kLl128PacketsPerWarp; base < total_packets;
       base += active_warps * kLl128PacketsPerWarp) {
    const size_t pkt_idx = base + group_idx;
    const bool active = warp.group_id < active_warps && pkt_idx < total_packets;
    // Modular indexing reuses buffer slots; flag increments per round to
    // prevent ABA.
    const size_t buf_idx = active ? (pkt_idx % aligned_buf_packets) : 0;
    const int64_t pkt_flag_value = flag_value +
        static_cast<int64_t>(active ? (pkt_idx / aligned_buf_packets) : 0);

    // --- Cooperative poll: all subgroup threads participate via __shfl_sync,
    // keeping them at the same PC for NVLink 128B store coalescing.
    // Only lane 7 reads the flag; the result is broadcast to all lanes.
    // Sender polls for -1 (slot free); receiver polls for pkt_flag_value
    // (data ready). ---
    if (active) {
      int ready = 0;
      do {
        if (lane_in_group == kLl128FlagLane) {
          ready = (remote_ll128_buf[buf_idx].load_flag() == kLl128ReadyToWrite)
              ? 1
              : 0;
          if (!ready) {
            TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
                timeout,
                "ll128_send: waiting for READY_TO_WRITE on packet %llu (buf_idx=%llu, current=%lld)",
                (unsigned long long)pkt_idx,
                (unsigned long long)buf_idx,
                (long long)remote_ll128_buf[buf_idx].load_flag());
          }
        }
        ready = __shfl_sync(subgroup_mask, ready, flag_src_lane);
      } while (!ready);
    }

    // --- Write: volatile-store 16B per thread ---
    // Declare outside `if (active)` so they survive across the __syncwarp.
    // Inactive lanes hold safe defaults but never execute the store.
    volatile uint64_t* slot = nullptr;
    uint64_t v0 = 0, v1 = 0;

    if (active) {
      Ll128Packet& remote_pkt = remote_ll128_buf[buf_idx];
      const size_t valid_payload = ll128_packet_payload_size(pkt_idx, nbytes);
      const size_t byte_offset_in_payload = lane_in_group * 16;
      slot = ll128_slot_ptr(remote_pkt, lane_in_group);

      // Lane 7 has 8B payload + 8B flag; lanes 0-6 have 16B payload
      const size_t max_payload = (lane_in_group == kLl128FlagLane) ? 8 : 16;

      if (byte_offset_in_payload < valid_payload) {
        const char* payload_src =
            src + pkt_idx * kLl128PayloadSize + byte_offset_in_payload;
        size_t avail = valid_payload - byte_offset_in_payload;
        if (avail > max_payload) {
          avail = max_payload;
        }

        if (avail >= 16) {
          v0 = *reinterpret_cast<const uint64_t*>(payload_src);
          v1 = *reinterpret_cast<const uint64_t*>(payload_src + 8);
        } else {
          // avail is exactly 8 (guaranteed by 16B alignment of nbytes)
          v0 = *reinterpret_cast<const uint64_t*>(payload_src);
        }
      }

      // Lane 7: override v1 with per-packet flag
      if (lane_in_group == kLl128FlagLane) {
        v1 = static_cast<uint64_t>(pkt_flag_value);
      }
    }

    // Force reconvergence after data-prep branches so ALL subgroup lanes
    // reach store128 at the same PC — required for NVLink 128B coalescing.
    __syncwarp(subgroup_mask);

    if (active) {
      comms::device::store128_volatile_global(slot, v0, v1);
    }

    // Full warp barrier: keeps all sub-groups at the same loop iteration.
    // Stride alignment of aligned_buf_packets (always a multiple of
    // kLl128PacketsPerWarp) prevents different sub-groups from colliding
    // on the same buf_idx across iterations.
    __syncwarp();
  }
#else
  (void)group;
  (void)src;
  (void)nbytes;
  (void)remote_ll128_buf;
  (void)timeout;
  (void)buffer_num_packets;
#endif
}

/**
 * ll128_recv — Receive data from a local LL128 buffer.
 *
 * The receiver polls the local LL128 buffer (which the sender wrote to
 * remotely), reads the data into registers, stores payload to the output
 * buffer, and then ACKs by writing kLl128ReadyToWrite back to the flag.
 *
 * @param group     ThreadGroup (auto-converted to warp scope via
 * to_warp_group())
 * @param dst       Local output buffer (contiguous user data, 16-byte aligned)
 * @param nbytes    Total message size in bytes (must be a multiple of 16)
 * @param local_ll128_buf  Pointer to local LL128 packet buffer
 * @param timeout   Timeout for flag polling
 * @param buffer_num_packets  Number of packets in the LL128 buffer.
 *                            0 = buffer is pre-sized to fit the entire message
 *                            (no chunking). >0 and < total packets =
 *                            windowed/chunked mode. Must be >=
 *                            kLl128PacketsPerWarp (4) when chunking.
 */
__device__ __forceinline__ void ll128_recv(
    const ThreadGroup& group,
    char* __restrict__ dst,
    size_t nbytes,
    Ll128Packet* __restrict__ local_ll128_buf,
    const Timeout& timeout,
    size_t buffer_num_packets = 0) {
#ifdef __CUDA_ARCH__
  // Constant base flag. Multi-step works via receiver ACK (-1) reset between
  // steps.
  const int64_t flag_value = 1;
  auto warp = group.to_warp_group();

  if (nbytes == 0) {
    return;
  }

  PIPES_DEVICE_CHECK(can_use_ll128(dst, nbytes));

  const int group_idx = warp.thread_id_in_group / kLl128ThreadsPerPacket;
  const int lane_in_group = warp.thread_id_in_group % kLl128ThreadsPerPacket;

  // Subgroup mask: only the 8 threads handling the same packet need to
  // coordinate. All 8 threads share the same active/pkt_idx state.
  const unsigned int subgroup_mask = 0xFFu
      << (group_idx * kLl128ThreadsPerPacket);

  // Warp-level lane of the flag-owning thread in this subgroup.
  const int flag_src_lane = group_idx * kLl128ThreadsPerPacket + kLl128FlagLane;

  const size_t total_packets = ll128_num_packets(nbytes);

  // Compute effective buffer size in packets
  const size_t buf_packets =
      (buffer_num_packets > 0 && buffer_num_packets < total_packets)
      ? buffer_num_packets
      : total_packets;

  // Runtime guard: buffer must hold at least one warp's worth of packets
  PIPES_DEVICE_CHECK(
      buf_packets >= total_packets || buf_packets >= kLl128PacketsPerWarp);

  // Warp clamping: use the lesser of buffer capacity and available warps
  const size_t buf_warps = buf_packets / kLl128PacketsPerWarp;
  const size_t active_warps =
      (buf_packets < total_packets && buf_warps < warp.total_groups)
      ? buf_warps
      : warp.total_groups;

  // Align buf_packets down to a multiple of the warp stride to prevent
  // cross-warp races on the same buf_idx. Without this, when
  // buf_packets % stride != 0, different warps in different loop iterations
  // can map to the same buf_idx and race on the ACK poll + write.
  const size_t stride = active_warps * kLl128PacketsPerWarp;
  const size_t aligned_buf_packets =
      (buf_packets < total_packets && buf_packets > stride)
      ? (buf_packets / stride) * stride
      : buf_packets;

  for (size_t base = warp.group_id * kLl128PacketsPerWarp; base < total_packets;
       base += active_warps * kLl128PacketsPerWarp) {
    const size_t pkt_idx = base + group_idx;
    const bool active = warp.group_id < active_warps && pkt_idx < total_packets;
    const size_t buf_idx = active ? (pkt_idx % aligned_buf_packets) : 0;
    const int64_t pkt_flag_value = flag_value +
        static_cast<int64_t>(active ? (pkt_idx / aligned_buf_packets) : 0);

    // --- Cooperative poll: all subgroup threads participate via __shfl_sync,
    // keeping them converged for the subsequent 128B load. ---
    if (active) {
      int ready = 0;
      do {
        if (lane_in_group == kLl128FlagLane) {
          ready =
              (local_ll128_buf[buf_idx].load_flag() == pkt_flag_value) ? 1 : 0;
          if (!ready) {
            TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
                timeout,
                "ll128_recv: waiting for flag_value=%lld on packet %llu (buf_idx=%llu, current=%lld)",
                (long long)pkt_flag_value,
                (unsigned long long)pkt_idx,
                (unsigned long long)buf_idx,
                (long long)local_ll128_buf[buf_idx].load_flag());
          }
        }
        ready = __shfl_sync(subgroup_mask, ready, flag_src_lane);
      } while (!ready);
    }

    // Ensure all threads converge before the 128B load.
    __syncwarp(subgroup_mask);

    // --- Read: volatile-load 16B per thread, then write to output ---
    if (active) {
      Ll128Packet& local_pkt = local_ll128_buf[buf_idx];
      const size_t valid_payload = ll128_packet_payload_size(pkt_idx, nbytes);
      const size_t byte_offset_in_payload = lane_in_group * 16;
      volatile uint64_t* slot = ll128_slot_ptr(local_pkt, lane_in_group);

      uint64_t v0, v1;
      comms::device::load128_volatile_global(slot, v0, v1);

      if (lane_in_group < kLl128FlagLane) {
        // Threads 0-6: pure payload (16B each)
        if (byte_offset_in_payload < valid_payload) {
          char* payload_dst =
              dst + pkt_idx * kLl128PayloadSize + byte_offset_in_payload;
          size_t avail = valid_payload - byte_offset_in_payload;
          if (avail >= 16) {
            auto* dst_u64 = reinterpret_cast<uint64_t*>(payload_dst);
            dst_u64[0] = v0;
            dst_u64[1] = v1;
          } else {
            // avail is exactly 8 (guaranteed by 16B alignment of nbytes)
            *reinterpret_cast<uint64_t*>(payload_dst) = v0;
          }
        }
      } else {
        // Thread 7: first 8B is payload, second 8B is flag (discard)
        const size_t t7_payload_offset = kLl128FlagLane * 16; // = 112
        if (t7_payload_offset < valid_payload) {
          char* payload_dst =
              dst + pkt_idx * kLl128PayloadSize + t7_payload_offset;
          size_t avail = valid_payload - t7_payload_offset;
          if (avail >= 8) {
            *reinterpret_cast<uint64_t*>(payload_dst) = v0;
          }
        }
      }
    }

    // Ensure all threads in the group have consumed data before ACKing.
    __syncwarp(subgroup_mask);
    if (active && lane_in_group == kLl128FlagLane) {
      local_ll128_buf[buf_idx].ack();
    }

    // Full warp barrier: keeps all sub-groups at the same loop iteration.
    // Stride alignment of aligned_buf_packets (always a multiple of
    // kLl128PacketsPerWarp) prevents different sub-groups from colliding
    // on the same buf_idx across iterations.
    __syncwarp();
  }
#else
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)local_ll128_buf;
  (void)timeout;
  (void)buffer_num_packets;
#endif
}

/**
 * ll128_forward — Receive from predecessor and forward to successor.
 *
 * For intermediate ranks in a broadcast ring. Reads data from the local
 * LL128 buffer (written by predecessor), forwards it to the successor's
 * remote LL128 buffer, copies payload to the local output buffer, and
 * ACKs the predecessor.
 *
 * @param group     ThreadGroup (auto-converted to warp scope via
 * to_warp_group())
 * @param dst       Local output buffer (contiguous user data, 16-byte aligned)
 * @param nbytes    Total message size in bytes (must be a multiple of 16)
 * @param local_ll128_buf   Pointer to local LL128 buffer (predecessor wrote)
 * @param remote_ll128_buf  Pointer to successor's LL128 buffer
 * @param timeout   Timeout for flag polling
 * @param buffer_num_packets  Number of packets in the LL128 buffer.
 *                            0 = buffer is pre-sized to fit the entire message
 *                            (no chunking). >0 and < total packets =
 *                            windowed/chunked mode. Must be >=
 *                            kLl128PacketsPerWarp (4) when chunking.
 */
__device__ __forceinline__ void ll128_forward(
    const ThreadGroup& group,
    char* __restrict__ dst,
    size_t nbytes,
    Ll128Packet* __restrict__ local_ll128_buf,
    Ll128Packet* __restrict__ remote_ll128_buf,
    const Timeout& timeout,
    size_t buffer_num_packets = 0) {
#ifdef __CUDA_ARCH__
  // Constant base flag. Multi-step works via receiver ACK (-1) reset between
  // steps.
  const int64_t flag_value = 1;
  auto warp = group.to_warp_group();

  if (nbytes == 0) {
    return;
  }

  PIPES_DEVICE_CHECK(can_use_ll128(dst, nbytes));

  const int group_idx = warp.thread_id_in_group / kLl128ThreadsPerPacket;
  const int lane_in_group = warp.thread_id_in_group % kLl128ThreadsPerPacket;

  // Subgroup mask: only the 8 threads handling the same packet need to
  // coordinate. All 8 threads share the same active/pkt_idx state.
  const unsigned int subgroup_mask = 0xFFu
      << (group_idx * kLl128ThreadsPerPacket);

  // Warp-level lane of the flag-owning thread in this subgroup.
  const int flag_src_lane = group_idx * kLl128ThreadsPerPacket + kLl128FlagLane;

  const size_t total_packets = ll128_num_packets(nbytes);

  // Compute effective buffer size in packets
  const size_t buf_packets =
      (buffer_num_packets > 0 && buffer_num_packets < total_packets)
      ? buffer_num_packets
      : total_packets;

  // Runtime guard: buffer must hold at least one warp's worth of packets
  PIPES_DEVICE_CHECK(
      buf_packets >= total_packets || buf_packets >= kLl128PacketsPerWarp);

  // Warp clamping: use the lesser of buffer capacity and available warps
  const size_t buf_warps = buf_packets / kLl128PacketsPerWarp;
  const size_t active_warps =
      (buf_packets < total_packets && buf_warps < warp.total_groups)
      ? buf_warps
      : warp.total_groups;

  // Align buf_packets down to a multiple of the warp stride to prevent
  // cross-warp races on the same buf_idx. Without this, when
  // buf_packets % stride != 0, different warps in different loop iterations
  // can map to the same buf_idx and race on the ACK poll + write.
  const size_t stride = active_warps * kLl128PacketsPerWarp;
  const size_t aligned_buf_packets =
      (buf_packets < total_packets && buf_packets > stride)
      ? (buf_packets / stride) * stride
      : buf_packets;

  for (size_t base = warp.group_id * kLl128PacketsPerWarp; base < total_packets;
       base += active_warps * kLl128PacketsPerWarp) {
    const size_t pkt_idx = base + group_idx;
    const bool active = warp.group_id < active_warps && pkt_idx < total_packets;
    const size_t buf_idx = active ? (pkt_idx % aligned_buf_packets) : 0;
    const int64_t pkt_flag_value = flag_value +
        static_cast<int64_t>(active ? (pkt_idx / aligned_buf_packets) : 0);

    // --- Phase 1: Cooperative poll local — all subgroup threads participate
    // via __shfl_sync, keeping them converged for the subsequent 128B load. ---
    if (active) {
      int ready = 0;
      do {
        if (lane_in_group == kLl128FlagLane) {
          ready =
              (local_ll128_buf[buf_idx].load_flag() == pkt_flag_value) ? 1 : 0;
          if (!ready) {
            TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
                timeout,
                "ll128_forward: waiting for flag_value=%lld on packet %llu (buf_idx=%llu, current=%lld)",
                (long long)pkt_flag_value,
                (unsigned long long)pkt_idx,
                (unsigned long long)buf_idx,
                (long long)local_ll128_buf[buf_idx].load_flag());
          }
        }
        ready = __shfl_sync(subgroup_mask, ready, flag_src_lane);
      } while (!ready);
    }

    // Ensure all threads converge before the 128B load.
    __syncwarp(subgroup_mask);

    // TODO: Experiment with bubble propagation in a follow-up.
    // --- Phase 2: Read local data + wait for remote ready ---
    uint64_t v0 = 0, v1 = 0;
    if (active) {
      volatile uint64_t* local_slot =
          ll128_slot_ptr(local_ll128_buf[buf_idx], lane_in_group);
      comms::device::load128_volatile_global(local_slot, v0, v1);
    }

    // --- Cooperative poll remote — all subgroup threads participate. ---
    if (active) {
      int ready = 0;
      do {
        if (lane_in_group == kLl128FlagLane) {
          ready = (remote_ll128_buf[buf_idx].load_flag() == kLl128ReadyToWrite)
              ? 1
              : 0;
          if (!ready) {
            TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
                timeout,
                "ll128_forward: waiting for READY_TO_WRITE on remote packet %llu (buf_idx=%llu, current=%lld)",
                (unsigned long long)pkt_idx,
                (unsigned long long)buf_idx,
                (long long)remote_ll128_buf[buf_idx].load_flag());
          }
        }
        ready = __shfl_sync(subgroup_mask, ready, flag_src_lane);
      } while (!ready);
    }

    // --- Phase 3: Forward to remote + copy to local + ACK ---
    volatile uint64_t* remote_slot = nullptr;

    if (active) {
      // Override v1 with per-packet flag_value before forwarding.
      // The flag read from local may differ from what needs to be forwarded
      // when chunking changes per-packet flags.
      if (lane_in_group == kLl128FlagLane) {
        v1 = static_cast<uint64_t>(pkt_flag_value);
      }

      remote_slot = ll128_slot_ptr(remote_ll128_buf[buf_idx], lane_in_group);
    }

    // Force reconvergence after flag-override branch so ALL subgroup lanes
    // reach store128 at the same PC — required for NVLink 128B coalescing.
    __syncwarp(subgroup_mask);

    if (active) {
      comms::device::store128_volatile_global(remote_slot, v0, v1);

      // Copy payload to local output buffer
      const size_t valid_payload = ll128_packet_payload_size(pkt_idx, nbytes);
      const size_t byte_offset_in_payload = lane_in_group * 16;

      if (lane_in_group < kLl128FlagLane) {
        if (byte_offset_in_payload < valid_payload) {
          char* payload_dst =
              dst + pkt_idx * kLl128PayloadSize + byte_offset_in_payload;
          size_t avail = valid_payload - byte_offset_in_payload;
          if (avail >= 16) {
            auto* dst_u64 = reinterpret_cast<uint64_t*>(payload_dst);
            dst_u64[0] = v0;
            dst_u64[1] = v1;
          } else {
            // avail is exactly 8 (guaranteed by 16B alignment of nbytes)
            *reinterpret_cast<uint64_t*>(payload_dst) = v0;
          }
        }
      } else {
        // Thread 7: v0 is payload (8B), v1 is flag (discard for local output)
        const size_t t7_payload_offset = kLl128FlagLane * 16; // = 112
        if (t7_payload_offset < valid_payload) {
          char* payload_dst =
              dst + pkt_idx * kLl128PayloadSize + t7_payload_offset;
          size_t avail = valid_payload - t7_payload_offset;
          if (avail >= 8) {
            *reinterpret_cast<uint64_t*>(payload_dst) = v0;
          }
        }
      }
    }

    // Ensure all threads in the group have forwarded and copied before ACKing.
    __syncwarp(subgroup_mask);
    if (active && lane_in_group == kLl128FlagLane) {
      local_ll128_buf[buf_idx].ack();
    }

    // Full warp barrier: keeps all sub-groups at the same loop iteration.
    // Stride alignment of aligned_buf_packets (always a multiple of
    // kLl128PacketsPerWarp) prevents different sub-groups from colliding
    // on the same buf_idx across iterations.
    __syncwarp();
  }
#else
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)local_ll128_buf;
  (void)remote_ll128_buf;
  (void)timeout;
  (void)buffer_num_packets;
#endif
}

} // namespace comms::pipes
