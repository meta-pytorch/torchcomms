// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"

namespace ctran::alltoall {
extern void* alltoallKerns[commNumTypes];

// Configure kernel launch parameters for AllToAll collectives.
// When nLocalRanks == 1 (no NVL peers), zeros args.count so the kernel
// short-circuits (skipping self-copy and NVL send/recv) and issues the
// self D2D copy via cudaMemcpyAsync. Otherwise launches the full
// multi-block alltoall kernel.
commResult_t setupKernelConfig(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    KernelConfig& config);
} // namespace ctran::alltoall
