// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include "comms/common/AtomicUtils.cuh"
#include "comms/common/DeviceConstants.cuh"

using comms::device::kWarpSize;

namespace comms::pipes {

// =============================================================================
// LL128 Protocol Constants
// =============================================================================

/// Total size of one LL128 packet (one cache line).
static constexpr size_t kLl128PacketSize = 128;

/// Usable payload bytes per packet (128B - 8B flag).
static constexpr size_t kLl128PayloadSize = 120;

/// Byte offset of the 8-byte flag within a packet.
static constexpr size_t kLl128FlagOffset = 120;

/// Number of threads that cooperate to read/write one packet.
static constexpr int kLl128ThreadsPerPacket = 8;

/// Number of packets processed simultaneously by one warp (32 / 8).
static constexpr int kLl128PacketsPerWarp = kWarpSize / 8;

/// Payload bytes processed by one warp per iteration (4 * 120).
static constexpr size_t kLl128PayloadPerWarp = kLl128PacketsPerWarp * 120;

/// LL128 flag protocol states.
/// Positive flag_value values indicate "data ready for step N".
enum class Ll128FlagState : int64_t {
  READY_TO_WRITE = -1,
};

/// Convenience constant — avoids verbose static_cast in comparisons.
static constexpr int64_t kLl128ReadyToWrite =
    static_cast<int64_t>(Ll128FlagState::READY_TO_WRITE);

/// Byte value used with cudaMemset to initialize LL128 packet flags to
/// kLl128ReadyToWrite (-1). All-ones = -1 in two's complement int64_t.
static constexpr int kLl128MemsetInitByte = 0xFF;

// =============================================================================
// Ll128Packet — 128-byte aligned packet with inline flag
// =============================================================================

/**
 * Ll128Packet - A 128-byte cache-line-aligned packet for the LL128 protocol.
 *
 * Layout:
 *   data[0..6] — 7 × 16B = 112B pure payload
 *   data[7]    — 8B payload + 8B flag (int64_t)
 *   Total: 128B = 120B payload + 8B flag
 *
 * The 8-byte flag (last 8 bytes of data[7]) serves as inline control:
 *   Sender → Receiver: flag = flag_value (positive) means "data ready"
 *   Receiver → Sender: flag = kLl128ReadyToWrite (-1) means "safe to overwrite"
 *
 * NVLink writes of 128 bytes to a 128-byte-aligned address are atomic at the
 * cache-line level: the receiver sees either the complete old or complete new
 * cache line. The flag being part of the same transaction is the atomicity
 * indicator.
 */
struct alignas(128) Ll128Packet {
  uint4 data[8]; // 8 × 16B = 128B

  /// Volatile-load the flag value.
  __device__ __forceinline__ int64_t load_flag() const;

  /// Volatile-store the flag value.
  __device__ __forceinline__ void store_flag(int64_t flag);

  /// ACK: signal that this packet's buffer is safe to overwrite.
  __device__ __forceinline__ void ack();
};

static_assert(sizeof(Ll128Packet) == 128, "Ll128Packet must be exactly 128B");
static_assert(
    alignof(Ll128Packet) == 128,
    "Ll128Packet must be 128-byte aligned");

// =============================================================================
// Device-side Helpers
// =============================================================================

/**
 * Return a pointer to the 16B slot for the given lane within a packet.
 *
 * Each 8-thread group writes one slot:
 *   lane 0 → data[0], lane 1 → data[1], ..., lane 7 → data[7]
 *
 * The returned pointer is suitable for load128_volatile_global /
 * store128_volatile_global (which operate on uint64_t pairs).
 *
 * @param pkt Reference to the packet
 * @param lane_in_group Lane index within the 8-thread group [0, 7]
 * @return Pointer to two consecutive uint64_t values (16B)
 */
__device__ __forceinline__ volatile uint64_t* ll128_slot_ptr(
    Ll128Packet& pkt,
    int lane_in_group) {
  return reinterpret_cast<volatile uint64_t*>(&pkt.data[lane_in_group]);
}

__device__ __forceinline__ const volatile uint64_t* ll128_slot_ptr(
    const Ll128Packet& pkt,
    int lane_in_group) {
  return reinterpret_cast<const volatile uint64_t*>(&pkt.data[lane_in_group]);
}

/**
 * Return a pointer to the 8-byte flag field within a packet.
 *
 * The flag occupies the last 8 bytes of data[7] (bytes 120-127).
 * data[7] is a uint4 = {x, y, z, w}. As two uint64_t values, the flag
 * is the second uint64_t (bytes 8-15 of data[7] = bytes 120-127 of packet).
 *
 * @param pkt Reference to the packet
 * @return Pointer to the flag as volatile int64_t
 */
__device__ __forceinline__ volatile int64_t* ll128_flag_ptr(Ll128Packet& pkt) {
  // data[7] starts at byte 112. As uint64_t*, [0] = bytes 112-119 (payload),
  // [1] = bytes 120-127 (flag).
  auto* base = reinterpret_cast<volatile int64_t*>(&pkt.data[7]);
  return base + 1;
}

__device__ __forceinline__ const volatile int64_t* ll128_flag_ptr(
    const Ll128Packet& pkt) {
  auto* base = reinterpret_cast<const volatile int64_t*>(&pkt.data[7]);
  return base + 1;
}

/**
 * Volatile-load the 8-byte flag from a packet (polling-efficient).
 *
 * Uses an 8-byte volatile load rather than a 16-byte load, halving
 * polling bandwidth compared to loading the full data[7] slot.
 *
 * @param pkt The packet to read from
 * @return The current flag value
 */
__device__ __forceinline__ int64_t ll128_load_flag(const Ll128Packet& pkt) {
#ifdef __CUDA_ARCH__
  return static_cast<int64_t>(comms::device::ld_volatile_global(
      reinterpret_cast<const volatile uint64_t*>(ll128_flag_ptr(pkt))));
#else
  (void)pkt;
  return 0;
#endif
}

/**
 * Volatile-store the 8-byte flag in a packet.
 *
 * @param pkt The packet to write to
 * @param flag The flag value to store
 */
__device__ __forceinline__ void ll128_store_flag(
    Ll128Packet& pkt,
    int64_t flag) {
#ifdef __CUDA_ARCH__
  comms::device::st_volatile_global(
      reinterpret_cast<volatile uint64_t*>(ll128_flag_ptr(pkt)),
      static_cast<uint64_t>(flag));
#else
  (void)pkt;
  (void)flag;
#endif
}

// =============================================================================
// Ll128Packet method implementations
// =============================================================================

__device__ __forceinline__ int64_t Ll128Packet::load_flag() const {
  return ll128_load_flag(*this);
}

__device__ __forceinline__ void Ll128Packet::store_flag(int64_t flag) {
  ll128_store_flag(*this, flag);
}

__device__ __forceinline__ void Ll128Packet::ack() {
  store_flag(static_cast<int64_t>(Ll128FlagState::READY_TO_WRITE));
}

// =============================================================================
// Host/Device Utility Functions
// =============================================================================

/**
 * Compute the number of valid payload bytes for a given packet index.
 *
 * @param packet_idx Zero-based packet index
 * @param total_bytes Total message size in bytes
 * @return Valid payload bytes for this packet [0, kLl128PayloadSize]
 */
__host__ __device__ __forceinline__ size_t
ll128_packet_payload_size(size_t packet_idx, size_t total_bytes) {
  size_t offset = packet_idx * kLl128PayloadSize;
  if (offset >= total_bytes) {
    return 0;
  }
  size_t remaining = total_bytes - offset;
  return remaining < kLl128PayloadSize ? remaining : kLl128PayloadSize;
}

/**
 * Compute the total number of LL128 packets needed for a message.
 *
 * @param nbytes Message size in bytes
 * @return Number of packets (0 if nbytes == 0)
 */
__host__ __device__ __forceinline__ size_t ll128_num_packets(size_t nbytes) {
  if (nbytes == 0) {
    return 0;
  }
  return (nbytes + kLl128PayloadSize - 1) / kLl128PayloadSize;
}

/**
 * Compute the LL128 buffer size needed for a given max message size.
 *
 * @param max_message_size Maximum message size in bytes
 * @return Buffer size in bytes (multiple of 128)
 */
__host__ __device__ __forceinline__ size_t
ll128_buffer_size(size_t max_message_size) {
  return ll128_num_packets(max_message_size) * kLl128PacketSize;
}

/**
 * Check whether the given pointer and byte count are eligible for LL128.
 *
 * LL128 requires:
 *   - nbytes is a multiple of 16
 *   - ptr is 16-byte aligned (or nullptr when nbytes == 0)
 *
 * @param ptr    Pointer to user data buffer (src or dst)
 * @param nbytes Message size in bytes
 * @return true if the arguments satisfy LL128 requirements
 */
__host__ __device__ __forceinline__ bool can_use_ll128(
    const void* ptr,
    size_t nbytes) {
  if (nbytes == 0) {
    return true;
  }
  return (nbytes % 16 == 0) && (reinterpret_cast<uintptr_t>(ptr) % 16 == 0);
}

} // namespace comms::pipes
