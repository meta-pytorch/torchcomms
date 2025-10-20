// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_REDUCESCATTER_IMPL_H_
#define CTRAN_REDUCESCATTER_IMPL_H_

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/CudaWrap.h"

commResult_t ctranReduceScatterDirect(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranReduceScatterRing(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranReduceScatterRHD(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream);

static inline commResult_t reduceScatterSingleRankImpl(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream) {
  // if out-of-place and count > 0, copy data to recvbuff; otherwise no-op
  if (sendbuff != recvbuff && recvcount > 0) {
    FB_COMMCHECK(comm->ctran_->mapper->icopy(
        recvbuff, sendbuff, recvcount * commTypeSize(datatype), stream));
  }
  return commSuccess;
}

static inline const std::string reduceScatterAlgoName(
    enum NCCL_REDUCESCATTER_ALGO algo) {
  switch (algo) {
    case NCCL_REDUCESCATTER_ALGO::ctdirect:
      return "CtranReduceScatterDirect";
    case NCCL_REDUCESCATTER_ALGO::ctring:
      return "CtranReduceScatterRing";
    case NCCL_REDUCESCATTER_ALGO::ctrhd:
      return "CtranReduceScatterRHD";
    case NCCL_REDUCESCATTER_ALGO::ctran:
      return "CtranAuto";
    case NCCL_REDUCESCATTER_ALGO::orig:
      return "Baseline";
    default:
      return "Unknown";
  }
}
#endif
