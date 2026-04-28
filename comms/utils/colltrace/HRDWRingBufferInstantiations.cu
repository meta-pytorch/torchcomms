// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Explicit template instantiations for HRDWRingBuffer typed kernel launches.
// The template definition lives in HRDWRingBuffer.h.
// Add new DataT types here and in HRDWRingBufferInstantiations.cuh.

#include "comms/utils/colltrace/HRDWRingBufferInstantiations.cuh"

namespace meta::comms::colltrace {

template cudaError_t launchRingBufferWrite<GraphCollTraceEvent>(
    cudaStream_t,
    HRDWEntry<GraphCollTraceEvent>*,
    uint64_t*,
    uint32_t,
    GraphCollTraceEvent,
    HRDWMemoryStrategy);

} // namespace meta::comms::colltrace
