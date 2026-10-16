// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/utils/colltrace/v1/CollMetadata.h"
#include "comms/utils/colltrace/v1/CollTraceHandle.h"

#include "comm.h"
#include "nccl.h"

namespace meta::comms::ncclx {

ncclResult_t newCollTraceInit(ncclComm* comm);

ncclResult_t newCollTraceDestroy(ncclComm* comm);

std::unique_ptr<meta::comms::colltrace::ICollMetadata>
getMetadataFromNcclKernelPlan(ncclKernelPlan& plan, cudaStream_t stream);

std::shared_ptr<meta::comms::colltrace::ICollTraceHandle>
getHandleFromNcclKernelPlan(ncclKernelPlan& plan, cudaStream_t stream);

std::unordered_map<std::string, std::string> collTraceGetInfo();

} // namespace meta::comms::ncclx

namespace ncclx::colltrace {

// Called at the top of every collective/p2p API entry point, before any early
// return, so every user call gets a colltrace v2 id even if it launches no
// kernel. No-op under v1.
void onCollectiveEnqueue(ncclComm* comm);

std::shared_ptr<meta::comms::colltrace::ICollTraceHandle>
prepareNcclKernelColltrace(
    ncclKernelPlan* plan,
    cudaStream_t stream,
    int compCap);

} // namespace ncclx::colltrace
