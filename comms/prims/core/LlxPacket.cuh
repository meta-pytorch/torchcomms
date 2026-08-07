// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// For the __host__ / __device__ execution-space macros used below. In CUDA/HIP
// device passes these are compiler builtins, but this header is also reachable
// from pure host translation units (it is pulled in transitively via the
// transport headers), where the macros come from the CUDA runtime headers.
// Matches the other core geometry/util headers (ThreadGroup.cuh, CopyUtils.cuh,
// TiledBuffer.cuh, AtomicUtils.cuh), which include <cuda_runtime.h> the same
// way.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "comms/common/DeviceConstants.cuh"

namespace comms::prims {

// =============================================================================
// LlxPacket<X, Y> — compile-time low-latency packet-geometry policy
// =============================================================================
//
// A packet is the atomic data+flag unit of the LL protocol: X (`kData`)
// payload bytes followed by Y (`kFlag`) trailing flag bytes, written and read
// as ONE hardware-atomic transfer so the trailing flag is never observed
// before its data. A receiver that sees the flag equal to the current
// flagVal therefore knows the packet's payload is already present -- this
// removes the separate DATA_READY signal / receiver fence of the Simple path.
//
//   LlxPacketGeometry = LlxPacket<4, 4>  ->  8 B packet (universal 8B
//   atomicity)
//
// This type is PURE geometry + layout: it owns packet sizing (the
// payload<->wire conversion the pipeline engine needs) and per-packet
// addressing (data slots + flag). It holds no state and no algorithm -- the
// flagVal lifecycle, the pack/unpack loops, and the readiness poll live in
// the LL send/recv implementation, not here.
template <int kDataBytes, int kFlagBytes>
struct LlxPacket {
  // ---- identity ----
  static constexpr int kData = kDataBytes; // X: payload bytes per packet
  static constexpr int kFlag = kFlagBytes; // Y: trailing flag bytes per packet
  static constexpr int kPacketBytes = kDataBytes + kFlagBytes;

  // ---- thread <-> packet mapping (drives pack/unpack parallelism) ----
  // One CUDA lane moves one 16 B vector slot. A packet that fits in a slot is
  // owned by a single lane; a larger packet is striped across several lanes.
  static constexpr int kSlotBytes = 16;
  static constexpr int kWordsPerSlot =
      kSlotBytes / static_cast<int>(sizeof(uint64_t));
  static constexpr int kThreadsPerPacket =
      kPacketBytes <= kSlotBytes ? 1 : kPacketBytes / kSlotBytes;
  static constexpr int kPacketsPerWarp =
      static_cast<int>(comms::device::kWarpSize) / kThreadsPerPacket;
  static constexpr int kFlagLane =
      kThreadsPerPacket - 1; // last lane owns the flag tail

  static_assert(
      kFlagBytes % static_cast<int>(sizeof(uint32_t)) == 0,
      "LlxPacket flag must be a multiple of 4 bytes");
  static constexpr int kFlagWords =
      kFlagBytes / static_cast<int>(sizeof(uint32_t));
  using FlagType = uint32_t;

  // ---- correctness guardrails (the atomicity story per packet size) ----
  static_assert(
      kPacketBytes % static_cast<int>(sizeof(uint64_t)) == 0,
      "packet size (kData + kFlag) must be a multiple of 8 B");
  static_assert(
      kPacketBytes == 8,
      "packet size must be 8 B (universal atomicity)");
  static_assert(
      kPacketBytes <= kSlotBytes ? (kSlotBytes % kPacketBytes == 0)
                                 : (kPacketBytes % kSlotBytes == 0),
      "packet must tile a lane slot or be tiled by lane slots");
  static_assert(kThreadsPerPacket >= 1, "kThreadsPerPacket must be >= 1");

  // Ring slots / chunk boundaries align to whole packets.
  static constexpr std::size_t kAlignBytes =
      static_cast<std::size_t>(kPacketBytes);

  // ---------------------------------------------------------------------------
  // Sizing: the payload <-> wire conversion the pipeline engine consumes.
  // All host + device constexpr; pure arithmetic (no min/reinterpret_cast).
  // ---------------------------------------------------------------------------

  /// Number of packets needed to carry `payload` user bytes.
  __host__ __device__ static constexpr std::size_t packet_count(
      std::size_t payload) {
    return payload == 0 ? 0 : (payload + kData - 1) / kData;
  }

  /// Staging (wire) bytes occupied by `payload` user bytes, rounded up to whole
  /// packets. This is the packet's `wireBytes(payload)`.
  __host__ __device__ static constexpr std::size_t wire_bytes(
      std::size_t payload) {
    return packet_count(payload) * static_cast<std::size_t>(kPacketBytes);
  }

  /// User payload bytes that fit in `wire` staging bytes. Inverse of
  /// `wire_bytes`; this is the packet's `maxPayloadForWire(wire)`.
  __host__ __device__ static constexpr std::size_t max_payload(
      std::size_t wire) {
    return wire / static_cast<std::size_t>(kPacketBytes) *
        static_cast<std::size_t>(kData);
  }

  /// Valid payload bytes carried by packet `packetIdx` of a `totalBytes`
  /// message (the last packet may be partially filled).
  __host__ __device__ static constexpr std::size_t valid_payload(
      std::size_t packetIdx,
      std::size_t totalBytes) {
    const std::size_t offset = packetIdx * static_cast<std::size_t>(kData);
    if (offset >= totalBytes) {
      return 0;
    }
    const std::size_t remaining = totalBytes - offset;
    return remaining < static_cast<std::size_t>(kData)
        ? remaining
        : static_cast<std::size_t>(kData);
  }

  /// 64-bit word pointer to lane `laneInPacket`'s 16 B data slot in `pkt`.
  __device__ __forceinline__ static uint64_t* slot(
      void* pkt,
      int laneInPacket) {
    return reinterpret_cast<uint64_t*>(pkt) + laneInPacket * kWordsPerSlot;
  }
  __device__ __forceinline__ static const uint64_t* slot(
      const void* pkt,
      int laneInPacket) {
    return reinterpret_cast<const uint64_t*>(pkt) +
        laneInPacket * kWordsPerSlot;
  }

  /// Pointer to the packet's trailing flag (at byte offset `kData`).
  /// Points to the first 32-bit flag word; the flag occupies kFlagWords.
  __device__ __forceinline__ static FlagType* flag_ptr(void* pkt) {
    return reinterpret_cast<FlagType*>(reinterpret_cast<char*>(pkt) + kData);
  }
  __device__ __forceinline__ static const FlagType* flag_ptr(const void* pkt) {
    return reinterpret_cast<const FlagType*>(
        reinterpret_cast<const char*>(pkt) + kData);
  }
};

// The IBGDA LL packet tier.
using LlxPacketGeometry = LlxPacket<4, 4>; // 8 B packet
} // namespace comms::prims
