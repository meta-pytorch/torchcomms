// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

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
  // No default member initializers: this keeps the handle trivially
  // default-constructible so it can be embedded in ncclx's ncclDevKernelArgs,
  // which is copied into __shared__ ncclShmem (dynamic init is illegal for
  // __shared__ variables). Producers value-initialize (`{}`, which zeroes every
  // member -> valid()==false) or assign an armed handle; there are no bare
  // default-constructed-then-used handles.
  uint32_t collId;
  // Which boundaries this kernel emits, set host-side per the kernel's role in
  // the collective: a single-kernel collective emits both; a multi-kernel one
  // emits start on its first kernel and end on its last; interior kernels emit
  // neither. Read by ColltraceEventScope (ctor→start, dtor→end).
  bool emitStart;
  bool emitEnd;
  ::hrdw_ring_buffer::HRDWRingBufferDeviceHandle<GraphCollTraceEvent> ring;

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

// Pin the scalar-before-ring layout the note above depends on. The host/device
// offsetof agreement (and thus in-kernel emit working at all) requires collId
// and the emit flags to stay ahead of the embedded ring handle; a future
// reorder that moves a scalar past `ring` would silently make the kernel read
// collId/emit flags back as 0. Fail the build instead of shipping that.
static_assert(
    offsetof(ColltraceDeviceHandle, collId) <
            offsetof(ColltraceDeviceHandle, emitStart) &&
        offsetof(ColltraceDeviceHandle, emitStart) <
            offsetof(ColltraceDeviceHandle, emitEnd) &&
        offsetof(ColltraceDeviceHandle, emitEnd) <
            offsetof(ColltraceDeviceHandle, ring),
    "ColltraceDeviceHandle scalars (collId/emitStart/emitEnd) must precede the "
    "embedded ring handle; see the layout note on the struct.");

// Must stay trivially default-constructible so it can be embedded in ncclx's
// ncclDevKernelArgs, which lands in __shared__ memory (no dynamic init
// allowed).
static_assert(
    std::is_trivially_default_constructible_v<ColltraceDeviceHandle>,
    "ColltraceDeviceHandle must be trivially default-constructible (no default "
    "member initializers) so it can live in __shared__ ncclShmem.");

} // namespace meta::comms::colltrace
