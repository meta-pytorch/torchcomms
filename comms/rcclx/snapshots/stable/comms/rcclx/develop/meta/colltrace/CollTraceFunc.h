// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "CollTraceEvent.h"
#include "comm.h"
#include "comms/rcclx/develop/meta/lib/CollTraceUtils.h"

namespace meta::colltrace {
class CollTraceError : public std::runtime_error {
 public:
  explicit CollTraceError(const std::string& what) : std::runtime_error(what) {}
};

bool enableGranularScuba();

ncclResult_t collTraceInit(ncclComm* comm);

ncclResult_t collTraceDestroy(ncclComm* comm);

std::unique_ptr<CollTraceEvent> collTraceAquireEventBaseline(
    ncclKernelPlan* plan,
    cudaStream_t stream);

ncclResult_t collTraceRecordStartEvent(
    ncclComm* comm,
    cudaStream_t launchStream,
    CollTraceEvent* event);

ncclResult_t collTraceRecordEndEvent(
    ncclComm* comm,
    ncclKernelPlan* plan,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event);

CollTraceInfo parseCollInfoFromCollTask(const ncclTaskColl& collTask);

CollTraceInfo parseCollInfoFromP2PTasks(
    const ncclTaskP2p& p2pTaskHead,
    int myRank);

} // namespace meta::colltrace
