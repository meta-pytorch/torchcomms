// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"

namespace ctran::device_alltoallv_pipes {

commResult_t setupKernelConfig(
    const void* sendbuff,
    void* recvbuff,
    const int64_t* sendcounts_d,
    const int64_t* recvcounts_d,
    commDataType_t datatype,
    CtranComm* comm,
    KernelConfig& config,
    ctran::device_alltoallv_pipes::KernArgs& kernArgs,
    int64_t sendcountsMultiplier = 1,
    int64_t recvcountsMultiplier = 1);

} // namespace ctran::device_alltoallv_pipes
