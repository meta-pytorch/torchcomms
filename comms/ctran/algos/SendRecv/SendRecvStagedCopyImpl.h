// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/SendRecv/Types.h"

namespace ctran::sendrecv {

commResult_t setupStagedCopyKernelConfig(
    CtranComm* comm,
    const std::vector<OpElem*>& nvlOps,
    KernelConfig& config,
    ctran::sendrecv::KernArgs& kernArgs);

} // namespace ctran::sendrecv
