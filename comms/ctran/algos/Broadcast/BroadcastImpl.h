// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_BROADCAST_IMPL_H_
#define CTRAN_BROADCAST_IMPL_H_
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/CtranExImpl.h"
#include "comms/utils/cvars/nccl_cvars.h"

commResult_t ctranBroadcastDirect(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int root,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranBroadcastBinomialTree(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int root,
    CtranComm* comm,
    cudaStream_t stream);

// API to broadcast between host memory.
// Use request to track completion
commResult_t ctranBroadcastBinomialTree(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int root,
    CtranComm* comm,
    ::ctran::CtranExRequestImpl* req);

static inline const std::string broadcastAlgoName(
    enum NCCL_BROADCAST_ALGO algo) {
  switch (algo) {
    case NCCL_BROADCAST_ALGO::ctran:
      return "CtranBroadcastAuto";
    case NCCL_BROADCAST_ALGO::ctdirect:
      return "CtranBroadcastDirect";
    case NCCL_BROADCAST_ALGO::ctbtree:
      return "CtranBroadcastBinomialTree";
    default:
      return "Unknown";
  }
}

#endif
