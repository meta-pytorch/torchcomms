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

} // namespace ncclx
