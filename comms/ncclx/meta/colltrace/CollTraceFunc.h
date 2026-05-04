// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comm.h"
#include "comms/utils/colltrace/CollTraceHandle.h"

namespace ncclx::colltrace {

std::shared_ptr<meta::comms::colltrace::ICollTraceHandle>
collTraceBaselineGetHandle(ncclKernelPlan* plan, cudaStream_t stream);

} // namespace ncclx::colltrace
