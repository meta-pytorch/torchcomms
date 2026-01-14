// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/utils/colltrace/CollMetadata.h"

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
