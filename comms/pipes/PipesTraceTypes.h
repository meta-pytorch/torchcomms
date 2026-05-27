// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"

namespace comms::pipes {

enum class PipesTraceEventType : uint8_t {
  kUnknown = 0,
  kHierAgIbChunkBegin = 1,
  kHierAgIbChunkReady = 2,
  kHierAgNvlWaitBegin = 3,
  kHierAgNvlChunkReady = 4,
  kHierAgNvlTaskDone = 5,
};

struct PipesTraceEvent {
  uint32_t step;
  uint16_t detail;
  uint8_t type;
  uint8_t rank;
};

static_assert(sizeof(PipesTraceEvent) == 8);

using PipesTraceHandle =
    ::hrdw_ring_buffer::HRDWRingBufferDeviceHandle<PipesTraceEvent>;

#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ __forceinline__ void write_pipes_trace(
    PipesTraceHandle trace,
    PipesTraceEventType type,
    uint32_t step,
    uint16_t detail,
    uint8_t rank) {
  if (trace.ring == nullptr) {
    return;
  }
  trace.write(PipesTraceEvent{step, detail, static_cast<uint8_t>(type), rank});
}
#endif

} // namespace comms::pipes
