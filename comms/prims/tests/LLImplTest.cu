// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/prims/core/LLImpl.cuh"
#include "comms/prims/core/LlxPacket.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"

namespace comms::prims::test {

// A single block packs then unpacks the same staging region and verifies the
// payload round-trips. pack/unpack use the same grid-stride mapping, so a
// thread reads back the packets it wrote; block syncs order the phases.
template <typename P>
__global__ void pack_unpack_kernel(
    const char* src,
    char* staging,
    char* dst,
    std::size_t nbytes,
    uint32_t* errorCount) {
  ThreadGroup g{
      threadIdx.x,
      blockDim.x,
      blockIdx.x,
      blockIdx.x,
      gridDim.x,
      SyncScope::BLOCK};

  const typename P::FlagType flagVal = static_cast<typename P::FlagType>(7);

  LLImpl<P>::pack(g, staging, src, nbytes, flagVal);
  g.sync();
  LLImpl<P>::unpack(g, dst, staging, nbytes, flagVal);
  g.sync();

  for (std::size_t i = threadIdx.x; i < nbytes; i += blockDim.x) {
    if (dst[i] != src[i]) {
      atomicAdd(errorCount, 1u);
    }
  }
}

void test_ll_pack_unpack(
    const char* src_d,
    char* staging_d,
    char* dst_d,
    std::size_t nbytes,
    uint32_t* errorCount_d) {
  pack_unpack_kernel<LlxPacketGeometry>
      <<<1, 256>>>(src_d, staging_d, dst_d, nbytes, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// LLImpl::store_flag/load_flag/is_flag_set round-trip over a single global
// packet. `pkt` is GLOBAL memory -> flag I/O uses global volatile ops.
template <typename P>
__device__ void check_flag(void* pkt, uint32_t* errorCount) {
  for (typename P::FlagType g :
       {typename P::FlagType(1),
        typename P::FlagType(42),
        static_cast<typename P::FlagType>(0xABCDu)}) {
    LLImpl<P>::store_flag(pkt, g);
    if (LLImpl<P>::load_flag(pkt) != g) {
      atomicAdd(errorCount, 1);
    }
    // All flag words should be replicated.
    const auto* flagWords = P::flag_ptr(pkt);
    for (int i = 0; i < P::kFlagWords; ++i) {
      if (flagWords[i] != g) {
        atomicAdd(errorCount, 1);
      }
    }
    // is_flag_set should agree.
    if (!LLImpl<P>::is_flag_set(pkt, g)) {
      atomicAdd(errorCount, 1);
    }
    if (LLImpl<P>::is_flag_set(pkt, g + 1)) {
      atomicAdd(errorCount, 1);
    }
  }
}

__global__ void test_ll_flag_roundtrip_kernel(void* p8, uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    check_flag<LlxPacketGeometry>(p8, errorCount);
  }
}

void test_ll_flag_roundtrip(void* p8_d, uint32_t* errorCount_d) {
  test_ll_flag_roundtrip_kernel<<<1, 1>>>(p8_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
