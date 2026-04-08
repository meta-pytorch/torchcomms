// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

namespace comms::pipes {

// =============================================================================
// LL Protocol Constants
// =============================================================================

/// Total size of one LL line (packet unit).
static constexpr size_t kLlLineSize = 16;

/// Usable payload bytes per line (data1 + data2 = 8 bytes).
static constexpr size_t kLlPayloadPerLine = 8;

/// Number of lines processed by one warp per iteration (one per thread).
static constexpr int kLlLinesPerWarp = 32;

/// LL flag sentinel: both flag1 and flag2 are set to this value to indicate
/// the slot is free for the sender to overwrite.
/// Valid flag values are in [1, 0xFFFFFFFE], so 0xFFFFFFFF never collides.
static constexpr uint32_t kLlReadyToWrite = 0xFFFFFFFF;

/// Byte value used with cudaMemset to initialize all LL line fields
/// (including flags) to kLlReadyToWrite.
static constexpr int kLlMemsetInitByte = 0xFF;

// =============================================================================
// LlLine — 16-byte packet unit with inline flags
// =============================================================================

/**
 * LlLine - A 16-byte packet unit for the LL protocol.
 *
 * Layout:
 *   data1 (4B) — lower 4 bytes of payload
 *   flag1 (4B) — flag word (must match expected value)
 *   data2 (4B) — upper 4 bytes of payload
 *   flag2 (4B) — flag word (must match expected value)
 *   Total: 16B = 8B payload + 8B flags
 *
 * The two flag words carry the same value. The receiver polls until both
 * flag1 and flag2 match the expected value. This provides the atomicity
 * indicator for the 16-byte volatile store over NVLink.
 *
 * Each thread handles one LlLine independently — no sub-warp coordination.
 */
struct LlLine {
  uint32_t data1;
  uint32_t flag1;
  uint32_t data2;
  uint32_t flag2;
};

static_assert(sizeof(LlLine) == 16, "LlLine must be exactly 16B");

// =============================================================================
// Device-side Volatile Load/Store (v4.u32)
// =============================================================================

/**
 * Volatile-load a 16-byte LL line from global memory.
 *
 * Uses ld.volatile.global.v4.u32 to bypass L1 cache and ensure NVLink
 * visibility. The load is atomic at the 16-byte level on NVLink.
 *
 * @param src  Pointer to the LlLine in global memory
 * @param line Output LlLine to populate
 */
__device__ __forceinline__ void ll_load_line(const LlLine* src, LlLine& line) {
#ifdef __CUDA_ARCH__
  asm volatile(
      "ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(line.data1), "=r"(line.flag1), "=r"(line.data2), "=r"(line.flag2)
      : "l"(src)
      : "memory");
#else
  (void)src;
  (void)line;
#endif
}

/**
 * Volatile-store a 16-byte LL line to global memory.
 *
 * Uses st.volatile.global.v4.u32 to bypass L1 cache and ensure NVLink
 * visibility. The store is atomic at the 16-byte level on NVLink.
 *
 * @param dst  Pointer to the LlLine in global memory
 * @param line LlLine to store
 */
__device__ __forceinline__ void ll_store_line(LlLine* dst, const LlLine& line) {
#ifdef __CUDA_ARCH__
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(dst),
                 "r"(line.data1),
                 "r"(line.flag1),
                 "r"(line.data2),
                 "r"(line.flag2)
               : "memory");
#else
  (void)dst;
  (void)line;
#endif
}

/**
 * ACK a buffer slot by writing kLlReadyToWrite to both flags and zeroing data.
 *
 * @param dst  Pointer to the LlLine in global memory
 */
__device__ __forceinline__ void ll_store_ack(LlLine* dst) {
  LlLine ack;
  ack.data1 = 0;
  ack.flag1 = kLlReadyToWrite;
  ack.data2 = 0;
  ack.flag2 = kLlReadyToWrite;
  ll_store_line(dst, ack);
}

// =============================================================================
// Host/Device Utility Functions
// =============================================================================

/**
 * Compute the number of LL lines needed for a message of @p nbytes.
 *
 * @param nbytes Message size in bytes (should be a multiple of 8)
 * @return Number of lines (0 if nbytes == 0)
 */
__host__ __device__ __forceinline__ size_t ll_num_lines(size_t nbytes) {
  if (nbytes == 0) {
    return 0;
  }
  return (nbytes + kLlPayloadPerLine - 1) / kLlPayloadPerLine;
}

/**
 * Compute the LL buffer size in bytes needed for a given max message size.
 *
 * @param max_message_size Maximum message size in bytes
 * @return Buffer size in bytes (multiple of 16)
 */
__host__ __device__ __forceinline__ size_t
ll_buffer_size(size_t max_message_size) {
  return ll_num_lines(max_message_size) * kLlLineSize;
}

/**
 * Compute the max payload bytes that fit in an LL buffer of a given size.
 *
 * Result is rounded down to an 8-byte multiple (LL alignment requirement).
 *
 * @param buffer_size_bytes Size of the LL buffer in bytes
 * @return Maximum payload capacity in bytes (8-byte aligned)
 */
__host__ __device__ __forceinline__ size_t
ll_buffer_payload_capacity(size_t buffer_size_bytes) {
  size_t num_lines = buffer_size_bytes / kLlLineSize;
  size_t raw_capacity = num_lines * kLlPayloadPerLine;
  return (raw_capacity / 8) * 8;
}

/**
 * Check whether the given pointer and byte count are eligible for LL.
 *
 * LL requires:
 *   - nbytes is a multiple of 8
 *   - ptr is 8-byte aligned (or nullptr when nbytes == 0)
 *
 * @param ptr    Pointer to user data buffer (src or dst)
 * @param nbytes Message size in bytes
 * @return true if the arguments satisfy LL requirements
 */
__host__ __device__ __forceinline__ bool can_use_ll(
    const void* ptr,
    size_t nbytes) {
  if (nbytes == 0) {
    return true;
  }
  return (nbytes % 8 == 0) && (reinterpret_cast<uintptr_t>(ptr) % 8 == 0);
}

/**
 * Compute the flag value for a given step.
 *
 * Returns step + 1 to ensure flag values are always >= 1, so the sentinel
 * kLlReadyToWrite (0xFFFFFFFF) can never collide with valid flags.
 *
 * @param step Zero-based step index
 * @return Flag value (always >= 1)
 */
__host__ __device__ __forceinline__ uint32_t ll_flag(size_t step) {
  return static_cast<uint32_t>(step) + 1;
}

} // namespace comms::pipes
