// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"

namespace ctran::alltoall {
extern void* alltoallKerns[commNumTypes];

// Stub kernel for pure-IB path (no local NVL peers). Synchronizes with
// GPE thread using 1 block / 1 thread to minimize SM occupancy.
extern __global__ void ncclKernelAllToAllStub(
    int* flag,
    CtranAlgoDeviceState* devState);

// Configure kernel launch parameters for AllToAll collectives.
// When nLocalRanks == 1 (no NVL peers), uses the lightweight stub kernel
// and issues self D2D copy via cudaMemcpyAsync. Otherwise uses the full
// multi-block alltoall kernel.
// Sets `kernel` to the kernel function pointer to launch.
commResult_t setupKernelConfig(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    KernelConfig& config,
    void*& kernel);
} // namespace ctran::alltoall
