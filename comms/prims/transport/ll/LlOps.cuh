// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// @lint-ignore-every CLANGTIDY facebook-modularize-issue-check

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/ll/LlPacket.cuh"

namespace comms::prims {

// =============================================================================
// LL Send/Recv/Forward Operations
// =============================================================================
//
// Low-level LL operations for point-to-point communication.
//
// THREAD ORGANIZATION:
//   - Warp-based (32 threads per warp)
//   - Each thread handles one LlLine (16 bytes) per iteration independently
//   - One warp handles 32 lines (256 bytes of payload) per iteration
//   - No sub-warp grouping needed (unlike LL128's 8-thread groups)
//
// THREAD-TO-DATA MAPPING:
//   Each warp independently handles all lines in the message.
//   Thread tid handles lines at offsets: tid, tid + kLlLinesPerWarp,
//   tid + 2*kLlLinesPerWarp, ... (stride = 32 lines per iteration)
//
// MEMORY ORDERING:
//   All operations use ld.volatile.global.v4.u32 / st.volatile.global.v4.u32
//   (bypass L1, NVLink-visible). The 16-byte volatile store is atomic on
//   NVLink.
//
// PER-THREAD POLLING:
//   Each thread independently polls its own line's flags (flag1 and flag2).
//   Dual-flag checking confirms the full 16B store landed intact.
//   No __shfl_sync needed.
//
// WINDOWED BUFFER (when buffer_num_lines > 0):
//   When the LL buffer is smaller than the message, the single warp loops
//   over the message using modular buffer indexing (line_idx %
//   aligned_buf_lines) and per-line flag increments (line_idx /
//   aligned_buf_lines) to prevent ABA issues.
//
// CHUNKING / WINDOWED BUFFER PROTOCOL:
//   Same two-state invariant as LL128:
//   - Sender polls for kLlReadyToWrite (slot free) -> writes data with
//     pkt_flag_value
//   - Receiver polls for pkt_flag_value (data ready) -> reads data ->
//     ACKs with kLlReadyToWrite
//   - Per-round flag increments prevent ABA within a step
//   - ACK reset prevents ABA between steps
//
// MULTI-STEP USAGE:
//   flag_value is hardcoded to 1. Multi-step works because the receiver ACKs
//   every slot with kLlReadyToWrite after reading, resetting the state machine.

/**
 * ll_send — Send data to a remote LL buffer.
 *
 * The sender reads user data from a local source buffer, packs it into
 * LlLines with flag values, and volatile-stores each line to the remote
 * (receiver's) LL buffer.
 *
 * Each invocation independently processes the entire message using a single
 * warp. When multiple warps call concurrently, the caller must provide each
 * warp with its own buffer partition to avoid races.
 *
 * @param group     ThreadGroup (auto-converted to warp scope)
 * @param src       Local source buffer (8-byte aligned)
 * @param nbytes    Total message size in bytes (must be a multiple of 8)
 * @param remote_ll_buf  Pointer to receiver's LL line buffer
 * @param timeout   Timeout for flag polling
 * @param buffer_num_lines  Number of lines in the LL buffer.
 *                          0 = buffer is pre-sized to fit the entire message.
 *                          >0 and < total lines = windowed/chunked mode.
 *                          Must be >= kLlLinesPerWarp (32) when chunking.
 */
__device__ __forceinline__ void ll_send(
    const ThreadGroup& group,
    const char* __restrict__ src,
    size_t nbytes,
    LlLine* __restrict__ remote_ll_buf,
    const Timeout& timeout,
    size_t buffer_num_lines = 0) {
#ifdef __CUDA_ARCH__
  const uint32_t flag_value = 1;
  const int tid = group.to_warp_group().thread_id_in_group;

  if (nbytes == 0) {
    return;
  }

  PIPES_DEVICE_CHECK(can_use_ll(src, nbytes, buffer_num_lines));

  const size_t total_lines = ll_num_lines(nbytes);

  const size_t buf_lines =
      (buffer_num_lines > 0 && buffer_num_lines < total_lines)
      ? buffer_num_lines
      : total_lines;

  // Align buf_lines down to warp boundary for modular indexing.
  // Safe: PIPES_DEVICE_CHECK above guarantees buf_lines >= kLlLinesPerWarp
  // when chunking.
  const size_t aligned_buf_lines = (buf_lines < total_lines)
      ? (buf_lines / kLlLinesPerWarp) * kLlLinesPerWarp
      : buf_lines;

  for (size_t base = 0; base < total_lines; base += kLlLinesPerWarp) {
    const size_t line_idx = base + tid;
    const bool active = line_idx < total_lines;

    const size_t buf_idx = active ? (line_idx % aligned_buf_lines) : 0;
    const uint32_t pkt_flag_value = flag_value +
        static_cast<uint32_t>(active ? (line_idx / aligned_buf_lines) : 0);

    // Poll for kLlReadyToWrite (slot free).
    // The || condition loops until BOTH flag1 and flag2 equal kLlReadyToWrite,
    // confirming the slot is in READY_TO_WRITE state. Dual-flag checking
    // matches NCCL LL protocol convention and guards against torn reads.
    if (active) {
      LlLine poll;
      do {
        ll_load_line(&remote_ll_buf[buf_idx], poll);
        if (poll.flag1 != kLlReadyToWrite || poll.flag2 != kLlReadyToWrite) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "ll_send: waiting for READY_TO_WRITE on line %llu (buf_idx=%llu)",
              (unsigned long long)line_idx,
              (unsigned long long)buf_idx);
        }
      } while (poll.flag1 != kLlReadyToWrite || poll.flag2 != kLlReadyToWrite);
    }

    // Pack user data into LlLine and write
    if (active) {
      const size_t payload_offset = line_idx * kLlPayloadPerLine;
      const auto* src_u32 =
          reinterpret_cast<const uint32_t*>(src + payload_offset);

      LlLine out;
      out.data1 = src_u32[0];
      out.flag1 = pkt_flag_value;
      out.data2 = src_u32[1];
      out.flag2 = pkt_flag_value;

      ll_store_line(&remote_ll_buf[buf_idx], out);
    }
  }
#else
  (void)group;
  (void)src;
  (void)nbytes;
  (void)remote_ll_buf;
  (void)timeout;
  (void)buffer_num_lines;
#endif
}

/**
 * ll_recv — Receive data from a local LL buffer.
 *
 * The receiver polls the local LL buffer (which the sender wrote to
 * remotely), reads the payload to the output buffer, and ACKs by writing
 * kLlReadyToWrite back to both flags.
 *
 * Each invocation independently processes the entire message using a single
 * warp. When multiple warps call concurrently, the caller must provide each
 * warp with its own buffer partition to avoid races.
 *
 * @param group     ThreadGroup (auto-converted to warp scope)
 * @param dst       Local output buffer (8-byte aligned)
 * @param nbytes    Total message size in bytes (must be a multiple of 8)
 * @param local_ll_buf  Pointer to local LL line buffer
 * @param timeout   Timeout for flag polling
 * @param buffer_num_lines  Number of lines in the LL buffer.
 *                          0 = buffer is pre-sized to fit the entire message.
 *                          >0 and < total lines = windowed/chunked mode.
 *                          Must be >= kLlLinesPerWarp (32) when chunking.
 */
__device__ __forceinline__ void ll_recv(
    const ThreadGroup& group,
    char* __restrict__ dst,
    size_t nbytes,
    LlLine* __restrict__ local_ll_buf,
    const Timeout& timeout,
    size_t buffer_num_lines = 0) {
#ifdef __CUDA_ARCH__
  const uint32_t flag_value = 1;
  const int tid = group.to_warp_group().thread_id_in_group;

  if (nbytes == 0) {
    return;
  }

  PIPES_DEVICE_CHECK(can_use_ll(dst, nbytes, buffer_num_lines));

  const size_t total_lines = ll_num_lines(nbytes);

  const size_t buf_lines =
      (buffer_num_lines > 0 && buffer_num_lines < total_lines)
      ? buffer_num_lines
      : total_lines;

  // Align buf_lines down to warp boundary for modular indexing.
  // Safe: PIPES_DEVICE_CHECK above guarantees buf_lines >= kLlLinesPerWarp
  // when chunking.
  const size_t aligned_buf_lines = (buf_lines < total_lines)
      ? (buf_lines / kLlLinesPerWarp) * kLlLinesPerWarp
      : buf_lines;

  for (size_t base = 0; base < total_lines; base += kLlLinesPerWarp) {
    const size_t line_idx = base + tid;
    const bool active = line_idx < total_lines;

    const size_t buf_idx = active ? (line_idx % aligned_buf_lines) : 0;
    const uint32_t pkt_flag_value = flag_value +
        static_cast<uint32_t>(active ? (line_idx / aligned_buf_lines) : 0);

    // Poll for pkt_flag_value (data ready).
    // The || condition loops until BOTH flag1 and flag2 match the expected
    // value, confirming the sender's full 16B data store landed.
    LlLine in;
    if (active) {
      do {
        ll_load_line(&local_ll_buf[buf_idx], in);
        if (in.flag1 != pkt_flag_value || in.flag2 != pkt_flag_value) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "ll_recv: waiting for flag=%u on line %llu (buf_idx=%llu, got flag1=%u flag2=%u)",
              (unsigned)pkt_flag_value,
              (unsigned long long)line_idx,
              (unsigned long long)buf_idx,
              (unsigned)in.flag1,
              (unsigned)in.flag2);
        }
      } while (in.flag1 != pkt_flag_value || in.flag2 != pkt_flag_value);
    }

    // Copy payload to output
    if (active) {
      const size_t payload_offset = line_idx * kLlPayloadPerLine;
      auto* dst_u32 = reinterpret_cast<uint32_t*>(dst + payload_offset);
      dst_u32[0] = in.data1;
      dst_u32[1] = in.data2;
    }

    // ACK: no sync needed because each thread handles exactly one LlLine per
    // iteration, so this thread's ACK only depends on its own copy above.
    if (active) {
      ll_store_ack(&local_ll_buf[buf_idx]);
    }
  }
#else
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)local_ll_buf;
  (void)timeout;
  (void)buffer_num_lines;
#endif
}

/**
 * ll_forward — Receive from predecessor and forward to successor.
 *
 * For intermediate ranks in a broadcast ring. Reads data from the local
 * LL buffer (written by predecessor), forwards it to the successor's
 * remote LL buffer, copies payload to the local output buffer, and
 * ACKs the predecessor.
 *
 * Each invocation independently processes the entire message using a single
 * warp. When multiple warps call concurrently, the caller must provide each
 * warp with its own buffer partition to avoid races.
 *
 * @param group     ThreadGroup (auto-converted to warp scope)
 * @param dst       Local output buffer (8-byte aligned)
 * @param nbytes    Total message size in bytes (must be a multiple of 8)
 * @param local_ll_buf   Pointer to local LL buffer (predecessor wrote)
 * @param remote_ll_buf  Pointer to successor's LL buffer
 * @param timeout   Timeout for flag polling
 * @param buffer_num_lines  Number of lines in the LL buffer.
 *                          0 = buffer is pre-sized to fit the entire message.
 *                          >0 and < total lines = windowed/chunked mode.
 *                          Must be >= kLlLinesPerWarp (32) when chunking.
 */
__device__ __forceinline__ void ll_forward(
    const ThreadGroup& group,
    char* __restrict__ dst,
    size_t nbytes,
    LlLine* __restrict__ local_ll_buf,
    LlLine* __restrict__ remote_ll_buf,
    const Timeout& timeout,
    size_t buffer_num_lines = 0) {
#ifdef __CUDA_ARCH__
  const uint32_t flag_value = 1;
  const int tid = group.to_warp_group().thread_id_in_group;

  if (nbytes == 0) {
    return;
  }

  PIPES_DEVICE_CHECK(can_use_ll(dst, nbytes, buffer_num_lines));

  const size_t total_lines = ll_num_lines(nbytes);

  const size_t buf_lines =
      (buffer_num_lines > 0 && buffer_num_lines < total_lines)
      ? buffer_num_lines
      : total_lines;
  const size_t aligned_buf_lines = (buf_lines < total_lines)
      ? (buf_lines / kLlLinesPerWarp) * kLlLinesPerWarp
      : buf_lines;

  for (size_t base = 0; base < total_lines; base += kLlLinesPerWarp) {
    const size_t line_idx = base + tid;
    const bool active = line_idx < total_lines;

    const size_t buf_idx = active ? (line_idx % aligned_buf_lines) : 0;
    const uint32_t pkt_flag_value = flag_value +
        static_cast<uint32_t>(active ? (line_idx / aligned_buf_lines) : 0);

    // Phase 1: Poll local buffer for data from predecessor
    LlLine in;
    if (active) {
      do {
        ll_load_line(&local_ll_buf[buf_idx], in);
        if (in.flag1 != pkt_flag_value || in.flag2 != pkt_flag_value) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "ll_forward: waiting for flag=%u on local line %llu (buf_idx=%llu)",
              (unsigned)pkt_flag_value,
              (unsigned long long)line_idx,
              (unsigned long long)buf_idx);
        }
      } while (in.flag1 != pkt_flag_value || in.flag2 != pkt_flag_value);
    }

    // Phase 2: Poll remote buffer for kLlReadyToWrite (slot free)
    if (active) {
      LlLine poll;
      do {
        ll_load_line(&remote_ll_buf[buf_idx], poll);
        if (poll.flag1 != kLlReadyToWrite || poll.flag2 != kLlReadyToWrite) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "ll_forward: waiting for READY_TO_WRITE on remote line %llu (buf_idx=%llu)",
              (unsigned long long)line_idx,
              (unsigned long long)buf_idx);
        }
      } while (poll.flag1 != kLlReadyToWrite || poll.flag2 != kLlReadyToWrite);
    }

    // Phase 3: Forward to remote + copy to local output + ACK local
    if (active) {
      // in.flag1/flag2 already == pkt_flag_value (guaranteed by poll loop)
      ll_store_line(&remote_ll_buf[buf_idx], in);

      // Copy payload to local output
      const size_t payload_offset = line_idx * kLlPayloadPerLine;
      auto* dst_u32 = reinterpret_cast<uint32_t*>(dst + payload_offset);
      dst_u32[0] = in.data1;
      dst_u32[1] = in.data2;

      // ACK predecessor's local buffer
      ll_store_ack(&local_ll_buf[buf_idx]);
    }
  }
#else
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)local_ll_buf;
  (void)remote_ll_buf;
  (void)timeout;
  (void)buffer_num_lines;
#endif
}

} // namespace comms::prims
