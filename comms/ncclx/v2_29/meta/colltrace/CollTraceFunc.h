// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <csignal>
#include <stdexcept>

#include "comm.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "info.h"
#include "meta/colltrace/CollTrace.h"

namespace ncclx::colltrace {

class CollTraceError : public std::runtime_error {
 public:
  explicit CollTraceError(const std::string& what) : std::runtime_error(what) {}
};

// We need to have visibility of ncclComm to check async error.
ncclResult_t collTraceInit(ncclComm* comm);

ncclResult_t collTraceDestroy(ncclComm* comm);

// For baseline transport
std::unique_ptr<CollTraceEvent> collTraceAquireEventBaseline(
    ncclKernelPlan* plan,
    cudaStream_t stream);

// For CtranGpe
std::unique_ptr<CollTraceEvent> collTraceAquireEventCtran(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const bool ifchecksum = false);

// TODO: remove this function after refactoring done
// !!! DO NOT USE THIS FUNCTION !!!
inline std::unique_ptr<CollTraceEvent> collTraceAquireEventCtran(
    ncclComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const bool ifchecksum = false) {
  return collTraceAquireEventCtran(
      comm->ctranComm_.get(), opGroup, kernelConfig, ifchecksum);
}

ncclResult_t collTraceRecordStartEvent(
    cudaStream_t launchStream,
    CollTraceEvent* event);

ncclResult_t collTraceRecordEndEvent(
    CtranComm* comm,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event);

// TODO: remove this function after refactoring done
// !!! DO NOT USE THIS FUNCTION !!!
inline ncclResult_t collTraceRecordEndEvent(
    ncclComm* comm,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event) {
  return collTraceRecordEndEvent(
      comm->ctranComm_.get(), launchStream, std::move(event));
}

ncclResult_t collTraceCtranRecordEndEvent(
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event);

ncclResult_t collTraceCtranSubmitEvent(std::unique_ptr<CollTraceEvent> event);

CpuWaitEvent* collTraceGetCpuStartWaitEvent(CollTraceEvent& event);

CpuWaitEvent* collTraceGetCpuEndWaitEvent(CollTraceEvent& event);

std::shared_ptr<meta::comms::colltrace::ICollTraceHandle>
collTraceBaselineGetHandle(ncclKernelPlan* plan, cudaStream_t stream);

} // namespace ncclx::colltrace
