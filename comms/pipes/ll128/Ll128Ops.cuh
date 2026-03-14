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
// WARP CLAMPING (when buffer_num_packets > 0):
//   When the LL128 buffer is smaller than the message, at most
//   `min(buf_packets / kLl128PacketsPerWarp, total_groups)` warps are
//   active per iteration.
//   Active warps loop over the message using modular buffer indexing
//   (pkt_idx % buf_packets) and per-packet flag values to prevent ABA
//   issues. Flag management is internal — callers do not supply a flag.
//   Inactive warps skip all work — they must NOT touch the buffer.

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
 * @param buffer_num_packets  Number of packets in the LL128 buffer (0 = no
 *                            chunking, buffer is pre-sized to fit the message).
 */
__device__ __forceinline__ void ll128_send(
    const ThreadGroup& group,
    const char* __restrict__ src,
    size_t nbytes,
    Ll128Packet* __restrict__ remote_ll128_buf,
    const Timeout& timeout,
    size_t buffer_num_packets = 0) {
#ifdef __CUDA_ARCH__
  const int64_t flag_value = 1;
  auto warp = group.to_warp_group();

  if (nbytes == 0) {
    return;
  }

  PIPES_DEVICE_CHECK(can_use_ll128(src, nbytes));

  const int group_idx = warp.thread_id_in_group / kLl128ThreadsPerPacket;
  const int lane_in_group = warp.thread_id_in_group % kLl128ThreadsPerPacket;

  // Subgroup sync: only the 8 threads handling the same packet need to
  // coordinate. All 8 threads share the same active/pkt_idx state.
  const unsigned int subgroup_mask = 0xFFu
      << (group_idx * kLl128ThreadsPerPacket);

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

  for (size_t base = warp.group_id * kLl128PacketsPerWarp; base < total_packets;
       base += active_warps * kLl128PacketsPerWarp) {
    const size_t pkt_idx = base + group_idx;
    const bool active = warp.group_id < active_warps && pkt_idx < total_packets;
    const size_t buf_idx = active ? (pkt_idx % buf_packets) : 0;
    const int64_t pkt_flag_value =
        flag_value + static_cast<int64_t>(active ? (pkt_idx / buf_packets) : 0);

    // --- Poll: only flag-owning thread polls remote readiness ---
    if (active && lane_in_group == kLl128FlagLane) {
      Ll128Packet& remote_pkt = remote_ll128_buf[buf_idx];
      while (remote_pkt.load_flag() != kLl128ReadyToWrite) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "ll128_send: waiting for READY_TO_WRITE on packet %llu (buf_idx=%llu, current=%lld)",
            (unsigned long long)pkt_idx,
            (unsigned long long)buf_idx,
            (long long)remote_pkt.load_flag());
      }
    }

    __syncwarp(subgroup_mask);

    // --- Write: volatile-store 16B per thread ---
    if (active) {
      Ll128Packet& remote_pkt = remote_ll128_buf[buf_idx];
      const size_t valid_payload = ll128_packet_payload_size(pkt_idx, nbytes);
      const size_t byte_offset_in_payload = lane_in_group * 16;
      volatile uint64_t* slot = ll128_slot_ptr(remote_pkt, lane_in_group);

      uint64_t v0 = 0, v1 = 0;

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

      __syncwarp(subgroup_mask); // Force reconvergence for 128B coalescing
      // ALL lanes store at the same PC — ensures NVLink cache-line coalescing
      comms::device::store128_volatile_global(slot, v0, v1);
    }
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
 * @param buffer_num_packets  Number of packets in the LL128 buffer (0 = no
 *                            chunking, buffer is pre-sized to fit the message).
 */
__device__ __forceinline__ void ll128_recv(
    const ThreadGroup& group,
    char* __restrict__ dst,
    size_t nbytes,
    Ll128Packet* __restrict__ local_ll128_buf,
    const Timeout& timeout,
    size_t buffer_num_packets = 0) {
#ifdef __CUDA_ARCH__
  const int64_t flag_value = 1;
  auto warp = group.to_warp_group();

  if (nbytes == 0) {
    return;
  }

  PIPES_DEVICE_CHECK(can_use_ll128(dst, nbytes));

  const int group_idx = warp.thread_id_in_group / kLl128ThreadsPerPacket;
  const int lane_in_group = warp.thread_id_in_group % kLl128ThreadsPerPacket;

  // Subgroup sync: only the 8 threads handling the same packet need to
  // coordinate. All 8 threads share the same active/pkt_idx state.
  const unsigned int subgroup_mask = 0xFFu
      << (group_idx * kLl128ThreadsPerPacket);

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

  for (size_t base = warp.group_id * kLl128PacketsPerWarp; base < total_packets;
       base += active_warps * kLl128PacketsPerWarp) {
    const size_t pkt_idx = base + group_idx;
    const bool active = warp.group_id < active_warps && pkt_idx < total_packets;
    const size_t buf_idx = active ? (pkt_idx % buf_packets) : 0;
    const int64_t pkt_flag_value =
        flag_value + static_cast<int64_t>(active ? (pkt_idx / buf_packets) : 0);

    // --- Poll: only flag-owning thread polls local readiness ---
    if (active && lane_in_group == kLl128FlagLane) {
      Ll128Packet& local_pkt = local_ll128_buf[buf_idx];
      while (local_pkt.load_flag() != pkt_flag_value) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "ll128_recv: waiting for flag_value=%lld on packet %llu (buf_idx=%llu, current=%lld)",
            (long long)pkt_flag_value,
            (unsigned long long)pkt_idx,
            (unsigned long long)buf_idx,
            (long long)local_pkt.load_flag());
      }
    }

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
 * @param buffer_num_packets  Number of packets in the LL128 buffer (0 = no
 *                            chunking, buffer is pre-sized to fit the message).
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
  const int64_t flag_value = 1;
  auto warp = group.to_warp_group();

  if (nbytes == 0) {
    return;
  }

  PIPES_DEVICE_CHECK(can_use_ll128(dst, nbytes));

  const int group_idx = warp.thread_id_in_group / kLl128ThreadsPerPacket;
  const int lane_in_group = warp.thread_id_in_group % kLl128ThreadsPerPacket;

  // Subgroup sync: only the 8 threads handling the same packet need to
  // coordinate. All 8 threads share the same active/pkt_idx state.
  const unsigned int subgroup_mask = 0xFFu
      << (group_idx * kLl128ThreadsPerPacket);

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

  for (size_t base = warp.group_id * kLl128PacketsPerWarp; base < total_packets;
       base += active_warps * kLl128PacketsPerWarp) {
    const size_t pkt_idx = base + group_idx;
    const bool active = warp.group_id < active_warps && pkt_idx < total_packets;
    const size_t buf_idx = active ? (pkt_idx % buf_packets) : 0;
    const int64_t pkt_flag_value =
        flag_value + static_cast<int64_t>(active ? (pkt_idx / buf_packets) : 0);

    // --- Phase 1: Poll local — only flag-owning thread polls ---
    if (active && lane_in_group == kLl128FlagLane) {
      Ll128Packet& local_pkt = local_ll128_buf[buf_idx];
      while (local_pkt.load_flag() != pkt_flag_value) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "ll128_forward: waiting for flag_value=%lld on packet %llu (buf_idx=%llu, current=%lld)",
            (long long)pkt_flag_value,
            (unsigned long long)pkt_idx,
            (unsigned long long)buf_idx,
            (long long)local_pkt.load_flag());
      }
    }

    __syncwarp(subgroup_mask);

    // TODO: Experiment with bubble propagation in a follow-up.
    // --- Phase 2: Read local data + wait for remote ready ---
    uint64_t v0 = 0, v1 = 0;
    if (active) {
      volatile uint64_t* local_slot =
          ll128_slot_ptr(local_ll128_buf[buf_idx], lane_in_group);
      comms::device::load128_volatile_global(local_slot, v0, v1);
    }

    // --- Poll: only flag-owning thread polls remote readiness ---
    if (active && lane_in_group == kLl128FlagLane) {
      Ll128Packet& remote_pkt = remote_ll128_buf[buf_idx];
      while (remote_pkt.load_flag() != kLl128ReadyToWrite) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "ll128_forward: waiting for READY_TO_WRITE on remote packet %llu (buf_idx=%llu, current=%lld)",
            (unsigned long long)pkt_idx,
            (unsigned long long)buf_idx,
            (long long)remote_pkt.load_flag());
      }
    }

    __syncwarp(subgroup_mask);

    // --- Phase 3: Forward to remote + copy to local + ACK ---
    if (active) {
      // Override v1 with per-packet flag_value before forwarding.
      // The flag read from local may differ from what needs to be forwarded
      // when chunking changes per-packet flags.
      if (lane_in_group == kLl128FlagLane) {
        v1 = static_cast<uint64_t>(pkt_flag_value);
      }

      // Forward to successor's remote LL128 buffer
      volatile uint64_t* remote_slot =
          ll128_slot_ptr(remote_ll128_buf[buf_idx], lane_in_group);
      __syncwarp(subgroup_mask);
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
