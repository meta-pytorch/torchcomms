// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/common/GpeRing.h" // ctran::gpe::KernelFlagDev
#include "comms/utils/colltrace/ColltraceDeviceHandle.h"

// In-kernel colltrace timestamp emission: a collective kernel publishes its own
// start/end timestamps into the graph colltrace ring, replacing the host-
// launched <<<1,1>>> timestamp kernels on the CUDA-graph path.

namespace ctran::device {

// Publish a colltrace start/end timestamp for this logical collective from
// inside the kernel. Single-writer election (block 0, thread 0); no-op on an
// unarmed handle (valid() == false). The timestamp is captured at the true
// kernel boundary. The ring write is the HRDWRingBuffer System-scope 128b
// atomic path, which requires sm_90+; the host therefore only arms this when
// the device supports it, falling back to the host-launched writer otherwise.
static __forceinline__ __device__ void ColltraceEmitEvent(
    meta::comms::colltrace::ColltraceDeviceHandle hdr,
    meta::comms::colltrace::GraphCollTracePhase phase) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && hdr.valid()) {
    hdr.ring.write(
        meta::comms::colltrace::GraphCollTraceEvent{hdr.collId, phase});
  }
}

// One RAII colltrace timing scope for every GPE kernel: emits the start
// timestamp on construction when the armed handle enables start, and the end
// timestamp on destruction when it enables end. The host arms the emitStart/
// emitEnd flags per the kernel's role, so a single uniform line covers all
// shapes: a single-kernel collective arms both; a multi-kernel one arms start
// on its first kernel and end on its last; interior kernels arm neither.
//
// Threading: both the start (ctor) and the end (dtor) are recorded by the SAME
// single elected writer -- block 0, thread 0 -- inside ColltraceEmitEvent;
// every other block/thread constructs the scope but its emits are no-ops. So it
// is safe (and required, for a correct start boundary) to place at the very top
// of any kernel, before any per-thread work. Both emits are also valid()-gated,
// so an unarmed handle makes the whole scope a no-op.
struct ColltraceEventScope {
  meta::comms::colltrace::ColltraceDeviceHandle hdr;

  __forceinline__ __device__ explicit ColltraceEventScope(
      meta::comms::colltrace::ColltraceDeviceHandle handle)
      : hdr(handle) {
    if (hdr.emitStart) {
      ColltraceEmitEvent(
          hdr, meta::comms::colltrace::GraphCollTracePhase::kStart);
    }
  }

  // Convenience overload: take the kernel flag directly so call sites don't
  // repeat the null-check ternary. A null flag yields an unarmed (no-op) scope.
  __forceinline__ __device__ explicit ColltraceEventScope(
      const ctran::gpe::KernelFlagDev* f)
      : ColltraceEventScope(
            f ? f->colltraceHdr
              : meta::comms::colltrace::ColltraceDeviceHandle{}) {}

  __forceinline__ __device__ ~ColltraceEventScope() {
    if (hdr.emitEnd) {
      ColltraceEmitEvent(
          hdr, meta::comms::colltrace::GraphCollTracePhase::kEnd);
    }
  }

  // Non-copyable and non-movable: the scope's lifetime must match the enclosing
  // kernel body so kEnd fires exactly once at kernel exit.
  ColltraceEventScope(const ColltraceEventScope&) = delete;
  ColltraceEventScope& operator=(const ColltraceEventScope&) = delete;
  ColltraceEventScope(ColltraceEventScope&&) = delete;
  ColltraceEventScope& operator=(ColltraceEventScope&&) = delete;
};

} // namespace ctran::device
