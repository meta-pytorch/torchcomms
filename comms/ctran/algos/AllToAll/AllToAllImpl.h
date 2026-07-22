// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"

// Cudagraph-aware path: transparently converts a regular alltoall to the
// persistent window-based AllToAllP algorithm during CUDA graph capture.
commResult_t ctranAllToAllCudagraphAware(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLTOALL_ALGO algo = NCCL_ALLTOALL_ALGO::ctgraph);

namespace ctran::alltoall {
extern void* alltoallKerns[commNumTypes];

// Configure kernel launch parameters for AllToAll collectives.
// Passes the comm's nLocalRanks to the kernel. When nLocalRanks == 1 (no NVL
// peers), the kernel short-circuits (skipping self-copy and NVL send/recv) and
// the self D2D copy is issued via cudaMemcpyAsync.
commResult_t setupKernelConfig(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    KernelConfig& config);
} // namespace ctran::alltoall
