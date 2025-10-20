// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/utils/colltrace/CollTraceHandle.h"

namespace meta::comms::colltrace {

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

// See getCollTraceHandle for the signature of func
void setCollTraceLegacyHandleFunc(
    std::function<std::unique_ptr<ICollTraceHandle>(
        CtranComm*,
        const std::vector<std::unique_ptr<struct OpElem>>&,
        const KernelConfig&,
        const bool)> func);

} // namespace meta::comms::colltrace
