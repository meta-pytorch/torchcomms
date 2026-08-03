// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <mutex>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/ctran/algos/common/GpeRing.h" // ctran::gpe::KernelFlagDev
#include "comms/utils/colltrace/CollTraceHandle.h" // ICollTraceHandle
#include "comms/utils/colltrace/ColltraceDeviceHandle.h" // ColltraceDeviceHandle

namespace ctran::gpe {

// In-kernel colltrace (the collective kernel publishing its own start/end
// timestamps into the colltrace ring) requires sm_90+ for the ring's 128b
// atomic device write — the same hardware requirement as the GPE device ring,
// but determined independently of it. The compute capability is queried once
// per process (call_once); NCCL pins one device per process, so a single cached
// value is correct.
inline bool inKernelColltraceSupported() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return false;
#else
  static std::once_flag onceFlag;
  static int ccMajor = -1; // -1 = unknown/query failed
  std::call_once(onceFlag, [] {
    int dev = 0;
    int major = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess) {
      ccMajor = major;
    }
  });
  return ccMajor >= 9;
#endif
}

// Arm a kernel flag for in-kernel colltrace start/end emission and maintain the
// cross-submit grouping state (`pendingGroup`) for multi-kernel collectives.
//
// A multi-kernel collective (e.g. AllGatherP's PipeStart..PipeSync..PipeEnd)
// records one CollTrace event: its Begin kernel writes the start and stashes
// the device handle (ring + collId) in `pendingGroup`, and its End kernel
// reuses that handle to write the end. A single-kernel collective arms both
// boundaries on the same kernel; interior kernels emit neither and leave
// `pendingGroup` untouched.
//
// Precondition: the caller has already gated on inKernelColltraceSupported()
// and a live kernel flag. `colltraceHandle` may be null (no record was
// created).
inline void armInKernelColltrace(
    meta::comms::colltrace::ColltraceDeviceHandle& pendingGroup,
    KernelFlagDev& flagDev,
    meta::comms::colltrace::ICollTraceHandle* colltraceHandle,
    bool emitStart,
    bool emitEnd) {
  if (emitStart) {
    // Begin (or single-kernel) collective: this kernel opens the record and
    // writes the start. Drop any pending group left behind by a Begin whose End
    // was never submitted (early-return, exception, or a future non-contiguous
    // algo path), so a stale collId can never be consumed by a later unrelated
    // End.
    pendingGroup = {};
    auto devHandle = colltraceHandle
        ? colltraceHandle->getColltraceDeviceHandle()
        : meta::comms::colltrace::ColltraceDeviceHandle{};
    if (devHandle.valid()) {
      // A single-kernel collective also emits the end here (emitEnd == true); a
      // multi-kernel begin hands the end to the group's End kernel below.
      devHandle.emitStart = true;
      devHandle.emitEnd = emitEnd;
      flagDev.colltraceHdr = devHandle;
      // Tell this collective's graph wait event it self-emits, so it skips the
      // host-launched timestamp path (the two must never both write the ring).
      // TEMPORARY: the host path is deleted at the in-kernel cutover.
      colltraceHandle->markInKernelEmit();
      if (!emitEnd) {
        // Multi-kernel begin: the End kernel reuses the same ring/collId.
        pendingGroup = devHandle;
        pendingGroup.emitStart = false;
        pendingGroup.emitEnd = true;
      }
    }
  } else if (emitEnd) {
    // End kernel of a multi-kernel collective: reuse the ring/collId stashed by
    // the Begin and emit only the end.
    if (pendingGroup.valid()) {
      flagDev.colltraceHdr = pendingGroup;
      // TEMPORARY (removed at the in-kernel cutover): keep the host path off
      // for the End kernel's wait event too, if it has one.
      if (colltraceHandle) {
        colltraceHandle->markInKernelEmit();
      }
    }
    // The group is closed once its End is armed; clear it so a later unrelated
    // End (a Begin whose End was dropped, then this collective's End) can never
    // consume this collective's stale collId.
    pendingGroup = {};
  }
  // Otherwise (interior kernel: no emit) there is nothing to arm.
}

} // namespace ctran::gpe
