// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "comms/common/AtomicUtils.cuh"
#include "comms/common/DeviceConstants.cuh"

using comms::device::kWarpSize;

namespace comms::prims {

// =============================================================================
// Packet<X, Y> — compile-time LL packet-geometry policy
// =============================================================================
//
// A packet is the atomic data+flag unit of the LL/LL128 protocol: X (`kData`)
// payload bytes + Y (`kFlag`) flag bytes, written/read as ONE hardware-atomic
// transfer so the trailing flag is never visible before its data.
//
//   LlPacketGeometry    = Packet<4, 4>    -> 8 B packet
//   Ll128PacketGeometry = Packet<120, 8>  -> 128 B packet
//
// Everything in the generic path (build_packet, ll_recv, the IBGDA wrapper,
// buffer sizing) is templated on packet geometry. The legacy fixed
// `Ll128Packet` / `ll128_*` symbols are untouched.
template <int kDataBytes, int kFlagBytes>
struct Packet {
  static constexpr int kData = kDataBytes; // X — payload bytes / packet
  static constexpr int kFlag = kFlagBytes; // Y — flag bytes / packet tail
  static constexpr int kPacketBytes = kDataBytes + kFlagBytes;
  static constexpr int kSlotBytes = 16; // one CUDA lane stores one 16 B slot
  static constexpr int kThreadsPerPacket =
      kPacketBytes <= kSlotBytes ? 1 : kPacketBytes / kSlotBytes;
  static constexpr int kWordsPerSlot = kSlotBytes / sizeof(uint64_t);
  static constexpr int kPacketsPerWarp = kWarpSize / kThreadsPerPacket;
  static constexpr int kFlagLane =
      kThreadsPerPacket - 1; // last lane owns flag tail
  using Flag = std::conditional_t<kFlagBytes == 4, int32_t, int64_t>;

  static_assert(
      kPacketBytes % static_cast<int>(sizeof(uint64_t)) == 0,
      "Packet size (kData+kFlag) must be a multiple of 8 B");
  static_assert(
      kPacketBytes == 8 || kPacketBytes == 128,
      "Packet size (kData+kFlag) must be 8 B (universal) or 128 B (PIX-gated)");
  static_assert(
      kPacketBytes <= kSlotBytes ? (kSlotBytes % kPacketBytes == 0)
                                 : (kPacketBytes % kSlotBytes == 0),
      "Packet must tile a lane slot or be tiled by lane slots");

  /// Number of packets needed for a message of `nbytes`.
  __host__ __device__ static constexpr size_t num_packets(size_t nbytes) {
    return nbytes == 0 ? 0 : (nbytes + kData - 1) / kData;
  }

  /// Valid payload bytes carried by packet `packetIdx` of a `totalBytes`
  /// message.
  __host__ __device__ static constexpr size_t payload_size(
      size_t packetIdx,
      size_t totalBytes) {
    const size_t offset = packetIdx * static_cast<size_t>(kData);
    if (offset >= totalBytes) {
      return 0;
    }
    const size_t remaining = totalBytes - offset;
    return remaining < static_cast<size_t>(kData) ? remaining
                                                  : static_cast<size_t>(kData);
  }

  /// Registered buffer bytes needed to hold a `maxBytes` message (multiple of
  /// `kPacketBytes`).
  __host__ __device__ static constexpr size_t buffer_size(size_t maxBytes) {
    return num_packets(maxBytes) * static_cast<size_t>(kPacketBytes);
  }
};

using LlPacketGeometry = Packet<4, 4>; // 8 B packet
using Ll128PacketGeometry = Packet<120, 8>; // 128 B packet — PIX-gated

// =============================================================================
// Device helpers (generic over packet geometry)
// =============================================================================

/// Pointer to lane `laneInPacket`'s slot within a packet.
template <typename P>
__device__ __forceinline__ volatile uint64_t* ll_slot_ptr(
    void* pkt,
    int laneInPacket) {
  auto* base = reinterpret_cast<volatile uint64_t*>(pkt);
  return base + laneInPacket * P::kWordsPerSlot;
}

template <typename P>
__device__ __forceinline__ const volatile uint64_t* ll_slot_ptr(
    const void* pkt,
    int laneInPacket) {
  auto* base = reinterpret_cast<const volatile uint64_t*>(pkt);
  return base + laneInPacket * P::kWordsPerSlot;
}

/// Pointer to the packet's flag tail.
template <typename P>
__device__ __forceinline__ volatile typename P::Flag* ll_flag_ptr(void* pkt) {
  auto* base = reinterpret_cast<volatile char*>(pkt);
  return reinterpret_cast<volatile typename P::Flag*>(base + P::kData);
}

template <typename P>
__device__ __forceinline__ const volatile typename P::Flag* ll_flag_ptr(
    const void* pkt) {
  auto* base = reinterpret_cast<const volatile char*>(pkt);
  return reinterpret_cast<const volatile typename P::Flag*>(base + P::kData);
}

/// Volatile-load the flag.
template <typename P>
__device__ __forceinline__ typename P::Flag ll_load_flag(const void* pkt) {
#ifdef __CUDA_ARCH__
  if constexpr (P::kFlag == 8) {
    return static_cast<typename P::Flag>(comms::device::ld_volatile_global(
        reinterpret_cast<const volatile uint64_t*>(ll_flag_ptr<P>(pkt))));
  } else {
    return *ll_flag_ptr<P>(pkt);
  }
#else
  (void)pkt;
  return 0;
#endif
}

/// Volatile-store the flag.
template <typename P>
__device__ __forceinline__ void ll_store_flag(
    void* pkt,
    typename P::Flag flag) {
#ifdef __CUDA_ARCH__
  if constexpr (P::kFlag == 8) {
    comms::device::st_volatile_global(
        reinterpret_cast<volatile uint64_t*>(ll_flag_ptr<P>(pkt)),
        static_cast<uint64_t>(flag));
  } else {
    *ll_flag_ptr<P>(pkt) = flag;
  }
#else
  (void)pkt;
  (void)flag;
#endif
}

} // namespace comms::prims
