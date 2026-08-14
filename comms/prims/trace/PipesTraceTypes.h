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

  kAllReducePhase1Begin = 12,
  kAllReducePhase1End = 13,
  kAllReducePhase2Begin = 14,
  kAllReducePhase2End = 15,
  kAllReducePhase3Begin = 16,
  kAllReducePhase3End = 17,
  kAllReduceRingReduceScatterBegin = 18,
  kAllReduceRingReduceScatterEnd = 19,
  kAllReduceRingAllGatherBegin = 20,
  kAllReduceRingAllGatherEnd = 21,
  kAllReduceSendSyncBegin = 22,
  kAllReduceSendSyncEnd = 23,
  kAllReduceSlotPrepareBegin = 24,
  kAllReduceSlotPrepareEnd = 25,
  kAllReduceWqeSubmitBegin = 26,
  kAllReduceWqeSubmitEnd = 27,
  kAllReduceDataReadyWaitBegin = 28,
  kAllReduceDataReadyWaitEnd = 29,
  kAllReduceReduceCopyBegin = 30,
  kAllReduceReduceCopyEnd = 31,
  kAllReduceDrainBegin = 32,
  kAllReduceDrainEnd = 33,
  kAllReduceBookkeepingBegin = 34,
  kAllReduceBookkeepingEnd = 35,
  kAllReduceLocalCompletionWaitBegin = 36,
  kAllReduceLocalCompletionWaitEnd = 37,
  kAllReduceRemoteSlotFreeWaitBegin = 38,
  kAllReduceRemoteSlotFreeWaitEnd = 39,
  kAllReduceStageCopyBegin = 40,
  kAllReduceStageCopyEnd = 41,
  kAllReducePathStaged = 42,
  kAllReducePathRegisteredProgress = 43,
  kAllReduceTreeSchedulerIdleBegin = 44,
  kAllReduceTreeSchedulerIdleEnd = 45,
};

enum class PipesTraceAllReducePhase : uint8_t {
  RingReduceScatter = 0,
  RingAllGather = 1,
  TreeReduce = 2,
  TreeBroadcast = 3,
};

enum class PipesTraceAllReduceRole : uint8_t {
  Send = 0,
  RecvCopy = 1,
  RecvReduce = 2,
  ForwardCopy = 3,
  ForwardReduce = 4,
  Scheduler = 5,
  Envelope = 6,
  Reserved = 7,
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

struct PipesTraceAllReduceContext {
  PipesTraceHandle trace;
  uint32_t traceStep{0};
  uint8_t phase{0};
  uint8_t dependencyStep{0};
  uint8_t block{0};
  uint8_t lane{0};
  uint8_t chunk{0};
  uint8_t role{0};
  uint8_t peer{0};
  uint8_t qpLane{0};
  uint32_t bytes{0};
};

struct PipesTraceProgressState {
  bool localCompletionWaitOpen{false};
  bool remoteSlotFreeWaitOpen{false};
  bool dataReadyWaitOpen{false};
};

inline constexpr uint32_t kPipesTraceFineSchemaVersion = 2;
inline constexpr uint32_t kPipesTraceFineSamplePeriod = 64;
inline constexpr uint8_t kPipesTracePeerNone = UINT8_MAX;
inline constexpr uint32_t kPipesTraceChunkShift = 0;
inline constexpr uint32_t kPipesTraceDependencyStepShift = 2;
inline constexpr uint32_t kPipesTraceBlockShift = 8;
inline constexpr uint32_t kPipesTraceLaneShift = 16;
inline constexpr uint32_t kPipesTracePhaseShift = 17;
inline constexpr uint32_t kPipesTraceRoleShift = 19;
inline constexpr uint32_t kPipesTraceQpLaneShift = 22;
inline constexpr uint32_t kPipesTraceOpTagShift = 28;
inline constexpr uint32_t kPipesTraceChunkMask = 0x03;
inline constexpr uint32_t kPipesTraceDependencyStepMask = 0x3f;
inline constexpr uint32_t kPipesTraceBlockMask = 0xff;
inline constexpr uint32_t kPipesTraceLaneMask = 0x01;
inline constexpr uint32_t kPipesTracePhaseMask = 0x03;
inline constexpr uint32_t kPipesTraceRoleMask = 0x07;
inline constexpr uint32_t kPipesTraceQpLaneMask = 0x3f;
inline constexpr uint32_t kPipesTraceOpTagMask = 0x0f;
inline constexpr uint32_t kPipesTraceBytesQuantum = 32;

#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    constexpr uint32_t pack_pipes_trace_allreduce_step(
        const PipesTraceAllReduceContext& context) {
  const uint32_t opTag =
      (context.traceStep / kPipesTraceFineSamplePeriod) & kPipesTraceOpTagMask;
  return ((static_cast<uint32_t>(context.chunk) & kPipesTraceChunkMask)
          << kPipesTraceChunkShift) |
      ((static_cast<uint32_t>(context.dependencyStep) &
        kPipesTraceDependencyStepMask)
       << kPipesTraceDependencyStepShift) |
      ((static_cast<uint32_t>(context.block) & kPipesTraceBlockMask)
       << kPipesTraceBlockShift) |
      ((static_cast<uint32_t>(context.lane) & kPipesTraceLaneMask)
       << kPipesTraceLaneShift) |
      ((static_cast<uint32_t>(context.phase) & kPipesTracePhaseMask)
       << kPipesTracePhaseShift) |
      ((static_cast<uint32_t>(context.role) & kPipesTraceRoleMask)
       << kPipesTraceRoleShift) |
      ((static_cast<uint32_t>(context.qpLane) & kPipesTraceQpLaneMask)
       << kPipesTraceQpLaneShift) |
      (opTag << kPipesTraceOpTagShift);
}

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

__device__ __forceinline__ void write_pipes_trace_allreduce(
    const PipesTraceAllReduceContext& context,
    PipesTraceEventType type) {
  if (context.trace.ring == nullptr || context.trace.writeIndex == nullptr) {
    return;
  }

  const uint32_t packed = pack_pipes_trace_allreduce_step(context);
  const uint64_t byteQuanta =
      (static_cast<uint64_t>(context.bytes) + kPipesTraceBytesQuantum - 1) /
      kPipesTraceBytesQuantum;
  const uint16_t packedBytes =
      static_cast<uint16_t>(byteQuanta > UINT16_MAX ? UINT16_MAX : byteQuanta);
  write_pipes_trace(context.trace, type, packed, packedBytes, context.peer);
}
#endif

} // namespace comms::prims
