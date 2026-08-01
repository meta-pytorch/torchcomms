// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/utils/colltrace/ColltraceDeviceHandle.h"

// In-kernel colltrace timestamp emission: a collective kernel publishes its own
// start/end timestamps into the graph colltrace ring, replacing the host-
// launched <<<1,1>>> timestamp kernels on the CUDA-graph path. This header is
// backend-agnostic (it depends only on the shared ColltraceDeviceHandle), so it
// is used by both the ctran GPE kernels and the baseline ncclx kernels; the
// ctran-specific KernelFlagDev convenience overload lives in ctran's own
// ColltraceDeviceEventScope.cuh which wraps this.

namespace meta::comms::colltrace {

// Publish a colltrace start/end timestamp for this logical collective from
// inside the kernel. Single-writer election: the elected writer is the one
// thread in block 0 with threadIdx.y == threadIdx.z == 0 and threadIdx.x equal
// to writerThreadIdxX, so a 2D/3D launch still elects exactly one writer and
// never duplicates an event. writerThreadIdxX defaults to 0 (thread 0); a
// caller may elect a different lane (e.g. thread 1) so this emit runs in
// parallel with other thread-0 work such as the GPE-ring kick. If the requested
// lane does not exist in this launch (writerThreadIdxX >= blockDim.x, e.g. a
// single-thread kernel), it falls back to thread 0 so the event is still
// emitted exactly once and never dropped. No-op on an unarmed handle (valid()
// == false). The timestamp reflects when the elected writer crosses this scope
// boundary, not a grid-wide barrier. The ring write is the HRDWRingBuffer
// System-scope 128b atomic path, which requires sm_90+; the host therefore only
// arms this when the device supports it, falling back to the host-launched
// writer otherwise.
static __forceinline__ __device__ void ColltraceEmitEvent(
    ColltraceDeviceHandle hdr,
    GraphCollTracePhase phase,
    unsigned int writerThreadIdxX = 0) {
  const unsigned int electedThreadIdxX =
      (writerThreadIdxX < blockDim.x) ? writerThreadIdxX : 0u;
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == electedThreadIdxX && threadIdx.y == 0 &&
      threadIdx.z == 0 && hdr.valid()) {
    hdr.ring.write(GraphCollTraceEvent{hdr.collId, phase});
  }
}

// One RAII colltrace timing scope for a collective kernel: emits the start
// timestamp on construction when the armed handle enables start, and the end
// timestamp on destruction when it enables end. The host arms the emitStart/
// emitEnd flags per the kernel's role, so a single uniform line covers all
// shapes: a single-kernel collective arms both; a multi-kernel one arms start
// on its first kernel and end on its last; interior kernels arm neither.
//
// Threading: the start (ctor) and end (dtor) emits are each recorded by a
// single elected writer in block 0 -- by default thread 0 for both, but the
// start and end lanes are independently configurable via startWriterThreadIdxX
// / endWriterThreadIdxX (see ColltraceEmitEvent). Every other block/thread
// constructs the scope but its emits are no-ops. So it is safe (and required,
// for a correct start boundary) to place at the very top of any kernel, before
// any per-thread work. Both emits are also valid()-gated, so an unarmed handle
// makes the whole scope a no-op.
//
// NOTE: the end lane must be a thread that stays alive until kernel exit. In
// ctran GPE kernels only thread 0 waits for collective completion; every other
// thread exits early, so endWriterThreadIdxX must stay 0 there or kEnd would
// record a too-early timestamp.
struct ColltraceDeviceEventScope {
  ColltraceDeviceHandle hdr;
  unsigned int startWriterThreadIdxX;
  unsigned int endWriterThreadIdxX;

  __forceinline__ __device__ explicit ColltraceDeviceEventScope(
      ColltraceDeviceHandle handle,
      unsigned int startWriterThreadIdxX = 0,
      unsigned int endWriterThreadIdxX = 0)
      : hdr(handle),
        startWriterThreadIdxX(startWriterThreadIdxX),
        endWriterThreadIdxX(endWriterThreadIdxX) {
    if (hdr.emitStart) {
      ColltraceEmitEvent(
          hdr, GraphCollTracePhase::kStart, startWriterThreadIdxX);
    }
  }

  __forceinline__ __device__ ~ColltraceDeviceEventScope() {
    if (hdr.emitEnd) {
      ColltraceEmitEvent(hdr, GraphCollTracePhase::kEnd, endWriterThreadIdxX);
    }
  }

  // Non-copyable and non-movable: the scope's lifetime must match the enclosing
  // kernel body so kEnd fires exactly once at kernel exit.
  ColltraceDeviceEventScope(const ColltraceDeviceEventScope&) = delete;
  ColltraceDeviceEventScope& operator=(const ColltraceDeviceEventScope&) =
      delete;
  ColltraceDeviceEventScope(ColltraceDeviceEventScope&&) = delete;
  ColltraceDeviceEventScope& operator=(ColltraceDeviceEventScope&&) = delete;
};

} // namespace meta::comms::colltrace
