// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/colltrace/ColltraceDeviceEventScope.cuh"

// In-kernel colltrace timestamp emission for ctran GPE kernels. The device
// writer (single-writer election + RAII start/end) is shared with the baseline
// ncclx kernels in ColltraceDeviceEventScope.cuh; every GPE kernel receives its
// ColltraceDeviceHandle by value as its third argument and constructs this
// scope from it. A default (null-ring) handle makes the scope a no-op.

namespace ctran::device {

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
};

} // namespace ctran::device
