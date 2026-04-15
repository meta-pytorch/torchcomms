// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Explicit template instantiation declarations for HRDWRingBuffer.
// Include this header from host code that uses HRDWRingBuffer::write().
// Add new DataT types here and in HRDWRingBufferInstantiations.cu.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include "comms/utils/HRDWRingBuffer.h"
#include "comms/utils/colltrace/GraphCollTraceEvent.h"

namespace meta::comms::colltrace {

// --- GraphCollTraceEvent (graph-initiated colltrace) ---
extern template cudaError_t launchRingBufferWrite<GraphCollTraceEvent>(
    cudaStream_t,
    HRDWEntry<GraphCollTraceEvent>*,
    uint64_t*,
    uint32_t,
    GraphCollTraceEvent);

} // namespace meta::comms::colltrace
