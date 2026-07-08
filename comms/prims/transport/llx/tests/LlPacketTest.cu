// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/prims/tests/Checks.h"
#include "comms/prims/transport/ll128/Ll128Packet.cuh"
#include "comms/prims/transport/llx/LlPacket.cuh"

namespace comms::prims::test {

using namespace comms::prims;

// =============================================================================
// Packet geometry for both tiers
// =============================================================================

__global__ void test_ll_packet_geometry_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // LL128 packet = 128 B, 8 threads, flag lane 7.
    if (Ll128PacketGeometry::kPacketBytes != 128 ||
        Ll128PacketGeometry::kThreadsPerPacket != 8 ||
        Ll128PacketGeometry::kFlagLane != 7 ||
        Ll128PacketGeometry::kData != 120 ||
        Ll128PacketGeometry::kPacketsPerWarp !=
            comms::device::kWarpSize / Ll128PacketGeometry::kThreadsPerPacket) {
      atomicAdd(errorCount, 1);
    }

    // LL packet = 8 B: one 4 B data + 4 B flag.
    if (LlPacketGeometry::kThreadsPerPacket != 1 ||
        LlPacketGeometry::kFlagLane != 0 || LlPacketGeometry::kData != 4 ||
        LlPacketGeometry::kFlag != 4 || LlPacketGeometry::kPacketBytes != 8 ||
        LlPacketGeometry::kPacketsPerWarp !=
            comms::device::kWarpSize / LlPacketGeometry::kThreadsPerPacket) {
      atomicAdd(errorCount, 1);
    }
  }
}

void test_ll_packet_geometry(uint32_t* errorCount_d) {
  test_ll_packet_geometry_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Slot pointers + flag round-trip (both tiers)
// =============================================================================

// `pkt` is in GLOBAL memory — ll_load/store_flag use global volatile ops.
template <typename P>
__device__ void check_slot_flag(void* pkt, uint32_t* errorCount) {
  auto* base = reinterpret_cast<volatile uint64_t*>(pkt);

  // Each lane's slot points to data[lane * P::kWordsPerSlot].
  for (int lane = 0; lane < P::kThreadsPerPacket; ++lane) {
    if (ll_slot_ptr<P>(pkt, lane) != base + lane * P::kWordsPerSlot) {
      atomicAdd(errorCount, 1);
    }
  }

  auto* expectedFlag = reinterpret_cast<volatile typename P::Flag*>(
      reinterpret_cast<volatile char*>(pkt) + P::kData);
  if (ll_flag_ptr<P>(pkt) != expectedFlag) {
    atomicAdd(errorCount, 1);
  }

  // Round-trip a few flag values.
  ll_store_flag<P>(pkt, static_cast<typename P::Flag>(kLl128ReadyToWrite));
  if (ll_load_flag<P>(pkt) !=
      static_cast<typename P::Flag>(kLl128ReadyToWrite)) {
    atomicAdd(errorCount, 1);
  }
  ll_store_flag<P>(pkt, static_cast<typename P::Flag>(1));
  if (ll_load_flag<P>(pkt) != static_cast<typename P::Flag>(1)) {
    atomicAdd(errorCount, 1);
  }
  ll_store_flag<P>(pkt, static_cast<typename P::Flag>(42));
  if (ll_load_flag<P>(pkt) != static_cast<typename P::Flag>(42)) {
    atomicAdd(errorCount, 1);
  }
}

__global__ void
test_ll_packet_slot_flag_kernel(void* p128, void* p8, uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    check_slot_flag<Ll128PacketGeometry>(p128, errorCount);
    check_slot_flag<LlPacketGeometry>(p8, errorCount);
  }
}

void test_ll_packet_slot_flag(
    void* p128_d,
    void* p8_d,
    uint32_t* errorCount_d) {
  test_ll_packet_slot_flag_kernel<<<1, 1>>>(p128_d, p8_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
