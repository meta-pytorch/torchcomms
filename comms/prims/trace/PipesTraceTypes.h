// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <cuda/atomic>
#endif

namespace comms::prims {

enum class PipesTraceEventType : uint8_t {
  kUnknown = 0,
  kHierAgIbChunkBegin = 1,
  kHierAgIbChunkReady = 2,
  kHierAgNvlWaitBegin = 3,
  kHierAgNvlChunkReady = 4,
  kHierAgNvlTaskDone = 5,

  kIbSendBegin = 6,
  kIbSendEnd = 7,
  kIbRecvBegin = 8,
  kIbRecvEnd = 9,
  kIbForwardBegin = 10,
  kIbForwardEnd = 11,
};

struct PipesTraceEvent {
  uint32_t step;
  uint16_t detail;
  uint8_t type;
  uint8_t rank;
};

static_assert(sizeof(PipesTraceEvent) == 8);

struct alignas(16) PipesTraceEntry {
  uint32_t timestamp;
  uint32_t epoch;
  PipesTraceEvent data;
};

static_assert(sizeof(PipesTraceEntry) == 16);
static_assert(alignof(PipesTraceEntry) == 16);

struct PipesTraceHandle {
  PipesTraceEntry* ring{nullptr};
  uint64_t* writeIndex{nullptr};
  uint32_t mask{0};
  uint32_t shift{0};
};

#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ __forceinline__ uint64_t read_pipes_trace_globaltimer() {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  return wall_clock64();
#elif defined(__CUDA_ARCH__)
  uint64_t timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
#else
  return 0;
#endif
}

__device__ __forceinline__ void write_pipes_trace(
    PipesTraceHandle trace,
    PipesTraceEventType type,
    uint32_t step,
    uint16_t detail,
    uint8_t rank) {
  if (trace.ring == nullptr || trace.writeIndex == nullptr) {
    return;
  }

#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900)
  // The backing ring uses 128-bit atomics, which are only available on
  // Hopper+. Keep traced kernels buildable for other device targets.
  (void)type;
  (void)step;
  (void)detail;
  (void)rank;
  return;
#else
  uint64_t slot =
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*trace.writeIndex)
          .fetch_add(1ULL, cuda::memory_order_relaxed);
  uint64_t idx = slot & trace.mask;

  PipesTraceEntry packed{};
  packed.timestamp =
      static_cast<uint32_t>(read_pipes_trace_globaltimer() >> 10);
  packed.epoch = static_cast<uint32_t>(slot >> trace.shift) + 1u;
  packed.data = PipesTraceEvent{
      .step = step,
      .detail = detail,
      .type = static_cast<uint8_t>(type),
      .rank = rank};

  uint64_t packed_lo, packed_hi;
  __builtin_memcpy(&packed_lo, &packed, sizeof(packed_lo));
  __builtin_memcpy(
      &packed_hi,
      reinterpret_cast<const char*>(&packed) + sizeof(packed_lo),
      sizeof(packed_hi));

  [[maybe_unused]] uint64_t prev_lo, prev_hi;
  asm volatile(
      "{ .reg .b128 _src, _dst;\n\t"
      "  mov.b128 _src, {%2, %3};\n\t"
      "  atom.exch.relaxed.sys.b128 _dst, [%4], _src;\n\t"
      "  mov.b128 {%0, %1}, _dst; }"
      : "=l"(prev_lo), "=l"(prev_hi)
      : "l"(packed_lo), "l"(packed_hi), "l"(&trace.ring[idx])
      : "memory");
#endif
}
#endif

} // namespace comms::prims
