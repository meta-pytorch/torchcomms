// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Explicit instantiations for test event types.
// The template definition lives in HRDWRingBuffer.h.

#include "comms/utils/HRDWRingBuffer.h"
#include "comms/utils/tests/HRDWRingBufferTestTypes.h"

namespace meta::comms::colltrace {

template cudaError_t launchRingBufferWrite<TestEvent>(
    cudaStream_t,
    HRDWEntry<TestEvent>*,
    uint64_t*,
    uint32_t,
    TestEvent,
    HRDWMemoryStrategy);

} // namespace meta::comms::colltrace
