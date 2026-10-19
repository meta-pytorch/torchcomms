// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/utils/colltrace/common/ColltraceDeviceHandle.h"
#include "comms/utils/colltrace/v1/CollTraceHandle.h"

namespace meta::comms::colltrace {

// Kernel-side tracing state the GPE keeps across submits: a multi-kernel
// collective (AllGatherP's pipeline) arms its Begin and End kernels on
// separate submits and both must carry the same collective id. Written and
// read only on the submit thread; a group's submits are contiguous, so one
// slot per implementation suffices.
struct KernelGroupState {
  KernelGroupState();
  // colltrace v1: ring + collId stashed by a Begin for its End.
  ColltraceDeviceHandle pendingGroup{};
  // colltrace v2: id of the collective being submitted.
  uint64_t v2CollId;
};

// Sets up tracing for one kernel submit, for whichever colltrace the process
// runs: under v2 it binds the kernel to the collective's v2 id; under v1 it
// records the collective and, for graph-captured kernels on sm_90+, arms the
// kernel to publish its own start/end. `hdr` receives the kernel's by-value
// colltrace argument (a default handle emits nothing). Returns the v1 record
// handle the GPE command carries (null or a dummy when nothing is recorded).
std::shared_ptr<ICollTraceHandle> traceKernelSubmit(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    bool isCapturing,
    bool ifchecksum,
    KernelGroupState& state,
    ColltraceDeviceHandle& hdr);

// CollTrace tracing logic for RMA. In RMA, sometimes it will not go through
// gpe->submit, so we need to record manually in the algo. When shouldRecord is
// false, we will return a dummy handle and any operation on the handle will be
// ignored.
std::shared_ptr<ICollTraceHandle> getCollTraceHandleRMA(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    bool shouldRecord);

std::shared_ptr<ICollTraceHandle> getCollTraceHandle(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const bool ifchecksum = false);

bool testOnlyClearCollTraceRecords(CtranComm* comm);

} // namespace meta::comms::colltrace
