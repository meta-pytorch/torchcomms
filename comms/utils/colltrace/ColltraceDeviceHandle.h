// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#include "comms/utils/colltrace/GraphCollTraceEvent.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"

namespace meta::comms::colltrace {

// The device-side handle colltrace hands to a collective kernel so it can
// publish its own start/end timestamps into the graph ring from inside the
// kernel, replacing the host-launched timestamp kernels on the graph path.
struct ColltraceDeviceHandle {
  // NOTE: these scalar fields are intentionally placed BEFORE the embedded
  // HRDWRingBufferDeviceHandle. That handle uses [[no_unique_address]] empty
  // members in its Overwrite specialization; when it is the first member, nvcc
  // computes a different offset for a following field on the device than on the
  // host -- it places the field past the end of the object (observed: host
  // offsetof(collId) == 24 but the device reads it at 32, with sizeof == 24 on
  // both), so the kernel reads collId/emit flags back as 0. Keeping the scalars
  // in the head bytes keeps the host and device offsets in agreement.
  uint32_t collId{0};
  // Which boundaries this kernel emits, set host-side per the kernel's role in
  // the collective: a single-kernel collective emits both; a multi-kernel one
  // emits start on its first kernel and end on its last; interior kernels emit
  // neither. Read by ColltraceEventScope (ctor→start, dtor→end).
  bool emitStart{false};
  bool emitEnd{false};
  ::hrdw_ring_buffer::HRDWRingBufferDeviceHandle<GraphCollTraceEvent> ring{};

  // Usable only when a ring is attached (the graph in-kernel path). A default
  // (null-ring) handle means "not armed": host callers keep the host-launched
  // timestamps, and the in-kernel emit is a no-op. Callable from device code
  // since the collective kernel gates its ring write on it.
#if defined(__CUDACC__) || defined(__HIPCC__)
  __host__ __device__
#endif
      bool valid() const {
    return ring.ring != nullptr;
  }
};

} // namespace meta::comms::colltrace
