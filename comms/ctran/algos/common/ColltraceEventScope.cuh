// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/common/GpeRing.h" // ctran::gpe::KernelFlagDev
#include "comms/utils/colltrace/ColltraceDeviceEventScope.cuh"

// In-kernel colltrace timestamp emission for ctran GPE kernels. The generic
// device writer (single-writer election + RAII start/end) is shared with the
// baseline ncclx kernels in ColltraceDeviceEventScope.cuh; this
// header only adds the ctran-specific convenience of arming directly from a
// KernelFlagDev so call sites don't repeat the null-check ternary.

namespace ctran::device {

// Re-expose the shared emitter under the ctran::device namespace that existing
// ctran call sites use.
using meta::comms::colltrace::ColltraceEmitEvent;

struct ColltraceEventScope : meta::comms::colltrace::ColltraceDeviceEventScope {
  __forceinline__ __device__ explicit ColltraceEventScope(
      meta::comms::colltrace::ColltraceDeviceHandle handle,
      unsigned int startWriterThreadIdxX = 0,
      unsigned int endWriterThreadIdxX = 0)
      : meta::comms::colltrace::ColltraceDeviceEventScope(
            handle,
            startWriterThreadIdxX,
            endWriterThreadIdxX) {}

  // Convenience overload: take the kernel flag directly. A null flag yields an
  // unarmed (no-op) scope. startWriterThreadIdxX / endWriterThreadIdxX pick the
  // elected writer lane for the start (ctor) and end (dtor) emits; both default
  // to thread 0. Keep the end lane at 0 in GPE kernels where only thread 0
  // waits for completion (see the base ColltraceEventScope note).
  __forceinline__ __device__ explicit ColltraceEventScope(
      const ctran::gpe::KernelFlagDev* f,
      unsigned int startWriterThreadIdxX = 0,
      unsigned int endWriterThreadIdxX = 0)
      : meta::comms::colltrace::ColltraceDeviceEventScope(
            f ? f->colltraceHdr
              : meta::comms::colltrace::ColltraceDeviceHandle{},
            startWriterThreadIdxX,
            endWriterThreadIdxX) {}
};

} // namespace ctran::device
