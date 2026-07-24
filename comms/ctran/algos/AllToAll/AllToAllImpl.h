// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"

namespace ctran {
struct CtranWin;
} // namespace ctran

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

// Window-based persistent alltoall (ctwin): converts an alltoall whose recvbuff
// lives inside a registered symmetric CtranWin into a persistent AllToAllP
// request that reuses the window's already-exchanged NVL/IPC state, caching the
// request on the window for reuse across calls. Supports CUDA graph capture
// inline: the window-owned request is built once and reused across replays (the
// window must outlive any graph that captured over it).
commResult_t ctranAllToAllCtwin(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLTOALL_ALGO algo);

// True if a ctwin alltoall over [recvbuff, recvBytes) is supported: recvbuff
// sits in a registered symmetric window that supports AllToAllP. On success
// sets *winOut (if non-null) to the resolving window. A null recvbuff returns
// false (ctwin is dormant until a caller opts in).
bool checkCtranAllToAllCtwinSupport(
    CtranComm* comm,
    const void* recvbuff,
    size_t recvBytes,
    enum NCCL_ALLTOALL_ALGO algo,
    ctran::CtranWin** winOut = nullptr);

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
