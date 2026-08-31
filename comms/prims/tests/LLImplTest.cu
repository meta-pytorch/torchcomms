// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/prims/core/LLImpl.cuh"
#include "comms/prims/core/LlxPacket.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Tile.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/LLImplTest.cuh"

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

// pack the chunk, optionally break one packet's flag, then ask all_flags_set.
template <typename P>
__global__ void all_flags_set_kernel(
    const char* src,
    char* staging,
    std::size_t nbytes,
    int corruptPacket,
    uint32_t* ready) {
  ThreadGroup g{
      threadIdx.x,
      blockDim.x,
      blockIdx.x,
      blockIdx.x,
      gridDim.x,
      SyncScope::BLOCK};

  const auto flagVal = static_cast<typename P::FlagType>(7);

  LLImpl<P>::pack(g, staging, src, nbytes, flagVal);
  g.sync();
  if (corruptPacket >= 0 && threadIdx.x == 0) {
    // A different generation is exactly what an un-arrived packet looks like:
    // the memory holds whatever the previous ring pass left there.
    LLImpl<P>::store_flag(
        staging +
            static_cast<std::size_t>(corruptPacket) *
                static_cast<std::size_t>(P::kPacketBytes),
        static_cast<typename P::FlagType>(flagVal + 1));
  }
  g.sync();

  const bool r = LLImpl<P>::all_flags_set(g, staging, nbytes, flagVal);
  if (threadIdx.x == 0) {
    *ready = r ? 1u : 0u;
  }
}

void test_ll_all_flags_set(
    const char* src_d,
    char* staging_d,
    std::size_t nbytes,
    int corruptPacket,
    uint32_t* ready_d) {
  all_flags_set_kernel<LlxPacketGeometry>
      <<<1, 256>>>(src_d, staging_d, nbytes, corruptPacket, ready_d);
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

// pack() the source into staging, then reduce it into `accum` with
// unpack_reduce. Building staging with pack() rather than by hand is
// deliberate: it also pins that the two agree on packet layout and flag
// placement, so a change to one without the other fails here.
// LLImpl takes a Combine functor rather than a reduce op, so the test supplies
// its own -- which also checks that the seam is usable without VecOps.
template <typename T, typename Op>
struct TestCombine {
  __device__ __forceinline__ void operator()(T& accum, const T& value) const {
    VecOps<T>::reduce_scalar(Op{}, accum, value);
  }
};

template <typename P, typename T, typename Op>
__global__ void unpack_reduce_kernel(
    const T* src,
    char* staging,
    T* accum,
    std::size_t nelems) {
  ThreadGroup g{
      threadIdx.x,
      blockDim.x,
      blockIdx.x,
      blockIdx.x,
      gridDim.x,
      SyncScope::BLOCK};

  const std::size_t nbytes = nelems * sizeof(T);
  const typename P::FlagType flagVal = static_cast<typename P::FlagType>(7);

  LLImpl<P>::pack(
      g, staging, reinterpret_cast<const char*>(src), nbytes, flagVal);
  g.sync();
  // unpack_reduce polls each packet's flag itself and syncs on the way out.
  LLImpl<P>::template unpack_reduce<T, TestCombine<T, Op>>(
      g, accum, staging, nbytes, flagVal);
}

namespace {

template <typename T>
void launchByKind(
    const T* src,
    char* staging,
    T* accum,
    std::size_t nelems,
    ReduceKind kind) {
  switch (kind) {
    case ReduceKind::Sum:
      unpack_reduce_kernel<LlxPacketGeometry, T, SumOp>
          <<<1, 256>>>(src, staging, accum, nelems);
      break;
    case ReduceKind::Max:
      unpack_reduce_kernel<LlxPacketGeometry, T, MaxOp>
          <<<1, 256>>>(src, staging, accum, nelems);
      break;
    case ReduceKind::Min:
      unpack_reduce_kernel<LlxPacketGeometry, T, MinOp>
          <<<1, 256>>>(src, staging, accum, nelems);
      break;
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace

void test_ll_unpack_reduce_f32(
    const float* src_d,
    char* staging_d,
    float* accum_d,
    std::size_t nelems,
    ReduceKind kind) {
  launchByKind<float>(src_d, staging_d, accum_d, nelems, kind);
}

void test_ll_unpack_reduce_f16(
    const void* src_d,
    char* staging_d,
    void* accum_d,
    std::size_t nelems) {
  launchByKind<__half>(
      static_cast<const __half*>(src_d),
      staging_d,
      static_cast<__half*>(accum_d),
      nelems,
      ReduceKind::Sum);
}

void test_ll_unpack_reduce_i64(
    const int64_t* src_d,
    char* staging_d,
    int64_t* accum_d,
    std::size_t nelems) {
  launchByKind<int64_t>(src_d, staging_d, accum_d, nelems, ReduceKind::Sum);
}

// pack(src) under the upstream generation, relay it into a second staging
// region under the downstream one, then check the two halves of the result
// separately. See test_ll_repack() in the header for why the flag pass does
// not go through unpack().
template <typename P>
__global__ void repack_kernel(
    const char* src,
    char* recvStaging,
    char* fwdStaging,
    char* dst,
    char* packetOut,
    std::size_t nbytes,
    bool useDst,
    uint32_t* flagErrors) {
  ThreadGroup g{
      threadIdx.x,
      blockDim.x,
      blockIdx.x,
      blockIdx.x,
      gridDim.x,
      SyncScope::BLOCK};

  const auto recvFlag = static_cast<typename P::FlagType>(7);
  const auto fwdFlag = static_cast<typename P::FlagType>(9);

  LLImpl<P>::pack(g, recvStaging, src, nbytes, recvFlag);
  g.sync();
  LLImpl<P>::repack(
      g, useDst ? dst : nullptr, fwdStaging, recvStaging, nbytes, fwdFlag);
  g.sync();

  // (1) Flag pass. One read per packet, no spin.
  const std::size_t nPackets = P::packet_count(nbytes);
  for (std::size_t i = threadIdx.x; i < nPackets; i += blockDim.x) {
    const char* pkt =
        fwdStaging + i * static_cast<std::size_t>(P::kPacketBytes);
    if (!LLImpl<P>::is_flag_set(pkt, fwdFlag)) {
      atomicAdd(flagErrors, 1u);
    }
  }
  g.sync();

  // (2) Payload pass. Force every flag to the downstream generation first so
  // unpack() is guaranteed to terminate even when the relay mis-stamped -- the
  // flag verdict is already recorded above, and this isolates the data half.
  for (std::size_t i = threadIdx.x; i < nPackets; i += blockDim.x) {
    LLImpl<P>::store_flag(
        fwdStaging + i * static_cast<std::size_t>(P::kPacketBytes), fwdFlag);
  }
  g.sync();
  LLImpl<P>::unpack(g, packetOut, fwdStaging, nbytes, fwdFlag);
}

void test_ll_repack(
    const char* src_d,
    char* recvStaging_d,
    char* fwdStaging_d,
    char* dst_d,
    char* packetOut_d,
    std::size_t nbytes,
    bool useDst,
    uint32_t* flagErrors_d) {
  repack_kernel<LlxPacketGeometry><<<1, 256>>>(
      src_d,
      recvStaging_d,
      fwdStaging_d,
      dst_d,
      packetOut_d,
      nbytes,
      useDst,
      flagErrors_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
