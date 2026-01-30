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

commResult_t ctranAllGatherRd(
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

// Window-based AllGather using RMA put+signal operations.
// The window's data buffer (win->winDataPtr) serves as the recv buffer.
// Window layout: [rank0_data | rank1_data | ... | rankN-1_data]
commResult_t ctranAllGatherWindow(
    const void* sendbuff,
    size_t sendcount,
    commDataType_t datatype,
    ctran::CtranWin* win,
    cudaStream_t stream);

static inline const std::string allGatherAlgoName(
    enum NCCL_ALLGATHER_ALGO algo) {
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctdirect:
      return "CtranAllGatherDirect";
    case NCCL_ALLGATHER_ALGO::ctrd:
      return "CtranAllGatherRd";
    case NCCL_ALLGATHER_ALGO::ctring:
      return "CtranAllGatherRing";
    case NCCL_ALLGATHER_ALGO::ctbrucks:
      return "CtranBrucksFF";
    case NCCL_ALLGATHER_ALGO::ctwindow:
      return "CtranAllGatherWindow";
    case NCCL_ALLGATHER_ALGO::ctran:
      return "CtranAuto";
    case NCCL_ALLGATHER_ALGO::orig:
      return "Baseline";
    default:
      return "Unknown";
  }
}

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
