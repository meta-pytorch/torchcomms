// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_ALLGATHER_IMPL_H_
#define CTRAN_ALLGATHER_IMPL_H_

#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran {
struct CtranWin;
} // namespace ctran

commResult_t ctranAllGatherPDirect(CtranPersistentRequest* req);

commResult_t ctranAllGatherDirect(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranAllGatherStreamedRd(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranAllGatherRing(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranAllGatherBrucksFF(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

// Window-based persistent allgather (ctwin): converts an allgather whose
// recvbuff lives inside a registered symmetric CtranWin into a persistent
// AllGatherP request that reuses the window's already-exchanged NVL/IPC state,
// caching the request on the window for reuse across calls. Supports CUDA graph
// capture inline: the window-owned request is built once and reused across
// replays (the window must outlive any graph that captured over it).
commResult_t ctranAllGatherCtwin(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLGATHER_ALGO algo);

// True if a ctwin-family allgather for `algo` over [recvbuff, recvBytes) is
// supported: recvbuff sits in a registered symmetric window that supports
// AllGatherP, and the topology satisfies the forced variant's constraints
// (ctwin_ring/ctwin_srd need nLocalRanks==1; ctwin_srd needs power-of-2 nRanks;
// ctwin_rdpipeline needs power-of-2 nNodes). On success sets *winOut (if
// non-null) to the resolving window. A null recvbuff returns false (ctwin is
// dormant until a caller opts in).
bool checkCtranAllGatherCtwinSupport(
    CtranComm* comm,
    const void* recvbuff,
    size_t recvBytes,
    enum NCCL_ALLGATHER_ALGO algo,
    ctran::CtranWin** winOut = nullptr);

static inline const std::string allGatherAlgoName(
    enum NCCL_ALLGATHER_ALGO algo) {
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctdirect:
      return "CtranAllGatherDirect";
    case NCCL_ALLGATHER_ALGO::ctsrd:
      return "CtranAllGatherStreamedRd";
    case NCCL_ALLGATHER_ALGO::ctring:
      return "CtranAllGatherRing";
    case NCCL_ALLGATHER_ALGO::ctbrucks:
      return "CtranBrucksFF";
    case NCCL_ALLGATHER_ALGO::cthierarchical_ring:
      return "AllGatherHierarchicalRing";
    case NCCL_ALLGATHER_ALGO::ctran:
      return "CtranAuto";
    case NCCL_ALLGATHER_ALGO::ctgraph:
      return "CtranCudagraphAware";
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
      return "CtranCudagraphPipeline";
    case NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline:
      return "CtranCudagraphRdPipeline";
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
      return "CtranCudagraphRing";
    case NCCL_ALLGATHER_ALGO::ctgraph_rd:
      return "CtranCudagraphRd";
    case NCCL_ALLGATHER_ALGO::ctwin:
      return "CtranAllGatherWin";
    case NCCL_ALLGATHER_ALGO::ctwin_ring:
      return "CtranAllGatherWinRing";
    case NCCL_ALLGATHER_ALGO::ctwin_srd:
      return "CtranAllGatherWinSrd";
    case NCCL_ALLGATHER_ALGO::ctwin_pipeline:
      return "CtranAllGatherWinPipeline";
    case NCCL_ALLGATHER_ALGO::ctwin_rdpipeline:
      return "CtranAllGatherWinRdPipeline";
    case NCCL_ALLGATHER_ALGO::orig:
      return "Baseline";
    default:
      return "Unknown";
  }
}

// Cudagraph-aware path: transparently converts a regular allgather to the
// persistent window-based AGP algorithm during CUDA graph capture.
commResult_t ctranAllGatherCudagraphAware(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLGATHER_ALGO algo = NCCL_ALLGATHER_ALGO::ctgraph);

commResult_t prepareAllGatherArgs(
    std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    KernelConfig& config,
    void** extraCopyBuff,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

#endif
