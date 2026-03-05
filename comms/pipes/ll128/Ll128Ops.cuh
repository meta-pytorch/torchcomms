// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// @lint-ignore-every CLANGTIDY facebook-modularize-issue-check

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/common/AtomicUtils.cuh"
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
 * @param flag_value   Step identifier (positive) for flag signaling
 * @param timeout   Timeout for flag polling
 * @param poll_ready  When true (default), poll for READY_TO_WRITE before each
 *                    batch. Set to false for single-step callers (e.g.,
 *                    AllToAllv) where each packet slot is used exactly once per
 *                    call and the buffer is pre-initialized/ACK'd between
 * calls. Multi-step callers that reuse buffer slots MUST leave this true.
 */
__device__ __forceinline__ void ll128_send(
    const ThreadGroup& group,
    const char* src,
    size_t nbytes,
    Ll128Packet* remote_ll128_buf,
    int64_t flag_value,
    const Timeout& timeout,
    bool poll_ready = true) {
#ifdef __CUDA_ARCH__
  auto warp = group.to_warp_group();

  if (nbytes == 0) {
    return;
  }

  if (!can_use_ll128(src, nbytes)) {
    printf(
        "ll128_send: requires 16-byte aligned pointer (%p) and "
        "nbytes (%llu) multiple of 16\n",
        (const void*)src,
        (unsigned long long)nbytes);
    __trap();
  }

  const int group_idx = warp.thread_id_in_group / kLl128ThreadsPerPacket;
  const int lane_in_group = warp.thread_id_in_group % kLl128ThreadsPerPacket;

  const size_t total_packets = ll128_num_packets(nbytes);
  const size_t total_warps = warp.total_groups;

  // Iterate over packets assigned to this warp in contiguous rounds.
  // Warp w handles packets [w*4, w*4+4) in round 0,
  // [w*4 + total_warps*4, ...) in round 1, etc.
  for (size_t base = warp.group_id * kLl128PacketsPerWarp; base < total_packets;
       base += total_warps * kLl128PacketsPerWarp) {
    const size_t pkt_idx = base + group_idx;
    const bool packet_active = pkt_idx < total_packets;

    // --- Poll: only flag-owning thread (lane 7) polls remote readiness ---
    // Reduces NVLink polling traffic by 8x: 1 reader per packet instead of 8
    // redundant readers. Other threads skip and converge at warp.sync().
    //
    // Skipped when poll_ready=false: single-step callers (e.g., AllToAllv with
    // kStepId=1) use each packet slot exactly once per call. The buffer is
    // initialized to READY_TO_WRITE and ACK'd back before the next kernel
    // launch, so polling is redundant. This eliminates 1 NVLink volatile load
    // + 1 warp.sync() per batch on the send path.
    if (poll_ready) {
      if (packet_active && lane_in_group == 7) {
        Ll128Packet& remote_pkt = remote_ll128_buf[pkt_idx];
        while (remote_pkt.load_flag() != kLl128ReadyToWrite) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "ll128_send: waiting for READY_TO_WRITE on packet %llu (current=%lld)",
              (unsigned long long)pkt_idx,
              (long long)remote_pkt.load_flag());
        }
      }

      warp.sync();
    }

    // --- Write: volatile-store 16B per thread ---
    if (packet_active) {
      Ll128Packet& remote_pkt = remote_ll128_buf[pkt_idx];
      const size_t valid_payload = ll128_packet_payload_size(pkt_idx, nbytes);
      const size_t byte_offset_in_payload = lane_in_group * 16;
      volatile uint64_t* slot = ll128_slot_ptr(remote_pkt, lane_in_group);

      uint64_t v0 = 0, v1 = 0;

      // Lane 7 has 8B payload + 8B flag; lanes 0-6 have 16B payload
      const size_t max_payload = (lane_in_group == 7) ? 8 : 16;

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

      // Lane 7: override v1 with flag
      if (lane_in_group == 7) {
        v1 = static_cast<uint64_t>(flag_value);
      }

      // ALL lanes store at the same PC — ensures NVLink cache-line coalescing
      comms::device::store128_volatile_global(slot, v0, v1);
    }
  }
#else
  (void)group;
  (void)src;
  (void)nbytes;
  (void)remote_ll128_buf;
  (void)flag_value;
  (void)timeout;
  (void)poll_ready;
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
 * @param flag_value   Expected flag value to wait for
 * @param timeout   Timeout for flag polling
 */
__device__ __forceinline__ void ll128_recv(
    const ThreadGroup& group,
    char* dst,
    size_t nbytes,
    Ll128Packet* local_ll128_buf,
    int64_t flag_value,
    const Timeout& timeout) {
#ifdef __CUDA_ARCH__
  auto warp = group.to_warp_group();

  if (nbytes == 0) {
    return;
  }

  if (!can_use_ll128(dst, nbytes)) {
    printf(
        "ll128_recv: requires 16-byte aligned pointer (%p) and "
        "nbytes (%llu) multiple of 16\n",
        (const void*)dst,
        (unsigned long long)nbytes);
    __trap();
  }

  const int group_idx = warp.thread_id_in_group / kLl128ThreadsPerPacket;
  const int lane_in_group = warp.thread_id_in_group % kLl128ThreadsPerPacket;

  const size_t total_packets = ll128_num_packets(nbytes);
  const size_t total_warps = warp.total_groups;

  for (size_t base = warp.group_id * kLl128PacketsPerWarp; base < total_packets;
       base += total_warps * kLl128PacketsPerWarp) {
    const size_t pkt_idx = base + group_idx;
    const bool packet_active = pkt_idx < total_packets;

    // --- Poll: only flag-owning thread (lane 7) polls local readiness ---
    // Reduces L2 cache pressure by 8x: 1 reader per packet instead of 8.
    if (packet_active && lane_in_group == 7) {
      Ll128Packet& local_pkt = local_ll128_buf[pkt_idx];
      while (local_pkt.load_flag() != flag_value) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "ll128_recv: waiting for flag_value=%lld on packet %llu (current=%lld)",
            (long long)flag_value,
            (unsigned long long)pkt_idx,
            (long long)local_pkt.load_flag());
      }
    }

    warp.sync();

    // --- Read: volatile-load 16B per thread, then write to output ---
    if (packet_active) {
      Ll128Packet& local_pkt = local_ll128_buf[pkt_idx];
      const size_t valid_payload = ll128_packet_payload_size(pkt_idx, nbytes);
      const size_t byte_offset_in_payload = lane_in_group * 16;
      volatile uint64_t* slot = ll128_slot_ptr(local_pkt, lane_in_group);

      uint64_t v0, v1;
      comms::device::load128_volatile_global(slot, v0, v1);

      if (lane_in_group < 7) {
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
        const size_t t7_payload_offset = 7 * 16; // = 112
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

    // Batched ACK: ensure all 4 packets' data is consumed before ACKing.
    // The sender sees ACKs in batches of 4, reducing poll stalls when
    // reusing buffer slots. 4x reduction in ACK-to-poll feedback latency.
    warp.sync();
    if (packet_active && lane_in_group == 7) {
      local_ll128_buf[pkt_idx].ack();
    }
  }
#else
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)local_ll128_buf;
  (void)flag_value;
  (void)timeout;
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
 * @param flag_value   Expected/forwarded flag value
 * @param timeout   Timeout for flag polling
 */
__device__ __forceinline__ void ll128_forward(
    const ThreadGroup& group,
    char* dst,
    size_t nbytes,
    Ll128Packet* local_ll128_buf,
    Ll128Packet* remote_ll128_buf,
    int64_t flag_value,
    const Timeout& timeout) {
#ifdef __CUDA_ARCH__
  auto warp = group.to_warp_group();

  if (nbytes == 0) {
    return;
  }

  if (!can_use_ll128(dst, nbytes)) {
    printf(
        "ll128_forward: requires 16-byte aligned pointer (%p) and "
        "nbytes (%llu) multiple of 16\n",
        (const void*)dst,
        (unsigned long long)nbytes);
    __trap();
  }

  const int group_idx = warp.thread_id_in_group / kLl128ThreadsPerPacket;
  const int lane_in_group = warp.thread_id_in_group % kLl128ThreadsPerPacket;

  const size_t total_packets = ll128_num_packets(nbytes);
  const size_t total_warps = warp.total_groups;

  for (size_t base = warp.group_id * kLl128PacketsPerWarp; base < total_packets;
       base += total_warps * kLl128PacketsPerWarp) {
    const size_t pkt_idx = base + group_idx;
    const bool packet_active = pkt_idx < total_packets;

    // --- Phase 1: Poll local — only flag-owning thread (lane 7) polls ---
    if (packet_active && lane_in_group == 7) {
      Ll128Packet& local_pkt = local_ll128_buf[pkt_idx];
      while (local_pkt.load_flag() != flag_value) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "ll128_forward: waiting for flag_value=%lld on packet %llu (current=%lld)",
            (long long)flag_value,
            (unsigned long long)pkt_idx,
            (long long)local_pkt.load_flag());
      }
    }

    warp.sync();

    // TODO: Follow-up to experiment with bubble propagation.
    // --- Phase 2: Read local data + wait for remote ready ---
    uint64_t v0 = 0, v1 = 0;
    if (packet_active) {
      volatile uint64_t* local_slot =
          ll128_slot_ptr(local_ll128_buf[pkt_idx], lane_in_group);
      comms::device::load128_volatile_global(local_slot, v0, v1);
    }

    // Only flag-owning thread polls remote readiness
    if (packet_active && lane_in_group == 7) {
      Ll128Packet& remote_pkt = remote_ll128_buf[pkt_idx];
      while (remote_pkt.load_flag() != kLl128ReadyToWrite) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "ll128_forward: waiting for READY_TO_WRITE on remote packet %llu (current=%lld)",
            (unsigned long long)pkt_idx,
            (long long)remote_pkt.load_flag());
      }
    }

    warp.sync();

    // --- Phase 3: Forward to remote + copy to local + ACK ---
    if (packet_active) {
      // Forward to successor's remote LL128 buffer
      // Thread 7 forwards with same flag_value (pass-through flag).
      volatile uint64_t* remote_slot =
          ll128_slot_ptr(remote_ll128_buf[pkt_idx], lane_in_group);
      comms::device::store128_volatile_global(remote_slot, v0, v1);

      // Copy payload to local output buffer
      const size_t valid_payload = ll128_packet_payload_size(pkt_idx, nbytes);
      const size_t byte_offset_in_payload = lane_in_group * 16;

      if (lane_in_group < 7) {
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
        const size_t t7_payload_offset = 7 * 16; // = 112
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

    // Batched ACK predecessor: ensure all 4 packets are forwarded and
    // copied before ACKing. 4x reduction in per-packet feedback traffic.
    warp.sync();
    if (packet_active && lane_in_group == 7) {
      local_ll128_buf[pkt_idx].ack();
    }
  }
#else
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)local_ll128_buf;
  (void)remote_ll128_buf;
  (void)flag_value;
  (void)timeout;
#endif
}

} // namespace comms::pipes
