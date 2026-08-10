// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/prims/core/LlxPacket.cuh"
#include "comms/prims/tests/Checks.h"

namespace comms::prims::test {

// =============================================================================
// Packet geometry constants for the 8 B tier.
// =============================================================================

__global__ void test_llpacket_geometry_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // LL packet = 8 B: 4 B data + 4 B flag, one thread/packet, flag lane 0.
    if (LlxPacketGeometry::kData != 4 || LlxPacketGeometry::kFlag != 4 ||
        LlxPacketGeometry::kPacketBytes != 8 ||
        LlxPacketGeometry::kThreadsPerPacket != 1 ||
        LlxPacketGeometry::kFlagLane != 0 ||
        LlxPacketGeometry::kPacketsPerWarp !=
            static_cast<int>(comms::device::kWarpSize) /
                LlxPacketGeometry::kThreadsPerPacket) {
      atomicAdd(errorCount, 1);
    }
  }
}

void test_llpacket_geometry(uint32_t* errorCount_d) {
  test_llpacket_geometry_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// slot() + flag_ptr() addressing. Pure geometry: the flag I/O
// (store/load/is_flag_set) is LLImpl's policy and is exercised in LLImplTest.
// =============================================================================

template <typename P>
__device__ void check_addressing(void* pkt, uint32_t* errorCount) {
  auto* base = reinterpret_cast<uint64_t*>(pkt);

  // Each lane's slot points to word[lane * kWordsPerSlot].
  for (int lane = 0; lane < P::kThreadsPerPacket; ++lane) {
    if (P::slot(pkt, lane) != base + lane * P::kWordsPerSlot) {
      atomicAdd(errorCount, 1);
    }
  }

  // Flag lives at byte offset kData.
  auto* expectedFlag = reinterpret_cast<typename P::FlagType*>(
      reinterpret_cast<char*>(pkt) + P::kData);
  if (P::flag_ptr(pkt) != expectedFlag) {
    atomicAdd(errorCount, 1);
  }
}

__global__ void test_llpacket_addressing_kernel(
    void* p8,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    check_addressing<LlxPacketGeometry>(p8, errorCount);
  }
}

void test_llpacket_addressing(void* p8_d, uint32_t* errorCount_d) {
  test_llpacket_addressing_kernel<<<1, 1>>>(p8_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
