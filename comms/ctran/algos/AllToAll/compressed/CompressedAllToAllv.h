// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef ENABLE_META_COMPRESSION

#pragma once

#include "comms/ctran/CtranComm.h"

#include <ai_codesign/comms/compression/CompressionManager.h>

enum class IbImplType {
  // Use slow NCCL socket to bootstrap compressed buffer sizes, slow but allow
  // efficient memory use
  Bootstrap,
  // Directly embed comprssed buffer into IB control message, fast but may
  // results in memory fragmentation due since we assume fixed memory size for
  // each rank
  IbExchange,
};

bool ctranCompressedAllToAllvSupport(CtranComm* comm);

commResult_t ctranCompressedAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranBootstrapCompressedAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranCompressedAllToAllvWrapper(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    IbImplType ibImplType,
    CtranComm* comm,
    cudaStream_t stream);
#endif
