// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "MetaFactory.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "nccl.h"

namespace ncclx {

__attribute__((visibility("default"))) ncclResult_t alltoallvDynamic(
    const void* const* sendbuffs,
    const size_t* sendcounts,
    void* const* recvbuffs,
    size_t maxSendcount,
    size_t maxRecvcount,
    size_t* actualRecvcounts,
    const Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  NCCLCHECK(metaCommToNccl(ctranAllToAllvDynamicSupport(
      comm->ctranComm_.get(),
      ncclToMetaComm(hints),
      maxSendcount,
      maxRecvcount,
      ncclToMetaComm(datatype))));

  return metaCommToNccl(ctranAllToAllvDynamic(
      sendbuffs,
      sendcounts,
      recvbuffs,
      maxSendcount,
      maxRecvcount,
      actualRecvcounts,
      ncclToMetaComm(hints),
      ncclToMetaComm(datatype),
      comm->ctranComm_.get(),
      stream));
}

__attribute__((visibility("default"))) ncclResult_t alltoallvDynamicSplit(
    const void* sendbuff,
    const size_t* sendSplitLengths,
    void* const* recvbuffs,
    size_t maxSendcount,
    size_t maxRecvcount,
    size_t* actualRecvcounts,
    const Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  NCCLCHECK(metaCommToNccl(ctranAllToAllvDynamicSupport(
      comm->ctranComm_.get(),
      ncclToMetaComm(hints),
      maxSendcount,
      maxRecvcount,
      ncclToMetaComm(datatype))));

  return metaCommToNccl(ctranAlltoallvDynamicSplit(
      sendbuff,
      sendSplitLengths,
      recvbuffs,
      maxSendcount,
      maxRecvcount,
      actualRecvcounts,
      ncclToMetaComm(hints),
      ncclToMetaComm(datatype),
      comm->ctranComm_.get(),
      stream));
}

__attribute__((visibility("default"))) ncclResult_t
alltoallvDynamicSplitNonContig(
    const void* sendbuff,
    const size_t* sendSplitLengths,
    size_t numSendSplitLengths,
    const size_t* sendIndices,
    const size_t* sendIndicesBlockLengths,
    void* const* recvbuffs,
    size_t* recvAllSplitLengths,
    size_t* recvIndices,
    size_t* recvIndicesBlockLengths,
    size_t maxSendcount,
    size_t maxRecvcount,
    const Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  NCCLCHECK(metaCommToNccl(ctranAllToAllvDynamicSupport(
      comm->ctranComm_.get(),
      ncclToMetaComm(hints),
      maxSendcount,
      maxRecvcount,
      ncclToMetaComm(datatype))));

  return metaCommToNccl(ctranAlltoallvDynamicSplitNonContig(
      sendbuff,
      sendSplitLengths,
      numSendSplitLengths,
      sendIndices,
      sendIndicesBlockLengths,
      recvbuffs,
      nullptr,
      maxSendcount,
      maxRecvcount,
      ncclToMetaComm(hints),
      ncclToMetaComm(datatype),
      comm->ctranComm_.get(),
      stream,
      false,
      recvAllSplitLengths));
}

} // namespace ncclx
