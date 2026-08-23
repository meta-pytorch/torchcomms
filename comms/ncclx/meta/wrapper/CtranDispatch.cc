// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/wrapper/CtranDispatch.h"
#include "meta/wrapper/NcclCommCtran.h"

#include "checks.h"
#include "comm.h"
#include "nccl.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/NcclxConfig.h"
#include "meta/wrapper/MetaFactory.h"

namespace ncclx {

std::optional<ncclResult_t> ctranTryAllGather(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    cudaStream_t stream) {
  const auto algo = NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo);
  if (algo != NCCL_ALLGATHER_ALGO::orig &&
      ctranAllGatherSupport(
          meta::comms::ncclx::ncclCommCtran(comm).get(),
          algo,
          stream,
          recvbuff,
          sendcount * comm->nRanks * ncclTypeSize(datatype))) {
    return metaCommToNccl(ctranAllGather(
        sendbuff,
        recvbuff,
        sendcount,
        ncclToMetaComm(datatype),
        meta::comms::ncclx::ncclCommCtran(comm).get(),
        stream,
        algo));
  }
  return std::nullopt;
}

std::optional<ncclResult_t> ctranTryAllReduce(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    cudaStream_t stream) {
  const auto algo = NCCLX_CONFIG_FIELD(comm->config, allreduceAlgo);
  // [NCCLX] Redirect to CTRAN if enabled and applicable
  if (algo != NCCL_ALLREDUCE_ALGO::orig &&
      ctranAllReduceSupport(
          meta::comms::ncclx::ncclCommCtran(comm).get(), algo)) {
    return metaCommToNccl(ctranAllReduce(
        sendbuff,
        recvbuff,
        count,
        ncclToMetaComm(datatype),
        ncclToMetaComm(op),
        meta::comms::ncclx::ncclCommCtran(comm).get(),
        stream,
        algo));
  }
  return std::nullopt;
}

std::optional<ncclResult_t> ctranTryBroadcast(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    cudaStream_t stream) {
  if (NCCL_BROADCAST_ALGO != NCCL_BROADCAST_ALGO::orig &&
      ctranBroadcastSupport(
          meta::comms::ncclx::ncclCommCtran(comm).get(), NCCL_BROADCAST_ALGO)) {
    return metaCommToNccl(ctranBroadcast(
        sendbuff,
        recvbuff,
        count,
        ncclToMetaComm(datatype),
        root,
        meta::comms::ncclx::ncclCommCtran(comm).get(),
        stream,
        NCCL_BROADCAST_ALGO));
  }
  return std::nullopt;
}

std::optional<ncclResult_t> ctranTryReduceScatter(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    cudaStream_t stream) {
  if (NCCL_REDUCESCATTER_ALGO != NCCL_REDUCESCATTER_ALGO::orig &&
      ctranReduceScatterSupport(
          meta::comms::ncclx::ncclCommCtran(comm).get(),
          NCCL_REDUCESCATTER_ALGO)) {
    return metaCommToNccl(ctranReduceScatter(
        sendbuff,
        recvbuff,
        recvcount,
        ncclToMetaComm(datatype),
        ncclToMetaComm(op),
        meta::comms::ncclx::ncclCommCtran(comm).get(),
        stream,
        NCCL_REDUCESCATTER_ALGO));
  }
  return std::nullopt;
}

std::optional<ncclResult_t> ctranTrySend(
    ncclComm* comm,
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    cudaStream_t stream) {
  const auto algo =
      NCCLX_CONFIG_FIELD(comm->config, sendrecvAlgo); // [META:PER_COMM_CONFIG]
  if ((algo != NCCL_SENDRECV_ALGO::orig) &&
      ctranSendRecvSupport(
          peer, meta::comms::ncclx::ncclCommCtran(comm).get(), algo, stream)) {
    // ctran send/recvs are enqueued within ctran wherease other non-ctran ones
    // are enqueued in the original queue. When reaching group end, these two
    // groups of ops will be issued separately.
    ncclResult_t ret;
    NCCLCHECK(ncclGroupStart());
    ret = metaCommToNccl(ctranSend(
        sendbuff,
        count,
        ncclToMetaComm(datatype),
        peer,
        meta::comms::ncclx::ncclCommCtran(comm).get(),
        stream,
        algo));
    NCCLCHECK(ncclGroupEnd());
    return ret;
  }
  return std::nullopt;
}

std::optional<ncclResult_t> ctranTryRecv(
    ncclComm* comm,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    cudaStream_t stream) {
  const auto algo =
      NCCLX_CONFIG_FIELD(comm->config, sendrecvAlgo); // [META:PER_COMM_CONFIG]
  if ((algo != NCCL_SENDRECV_ALGO::orig) &&
      ctranSendRecvSupport(
          peer, meta::comms::ncclx::ncclCommCtran(comm).get(), algo, stream)) {
    // ctran send/recvs are enqueued within ctran wherease other non-ctran ones
    // are enqueued in the original queue. When reaching group end, these two
    // groups of ops will be issued separately.
    ncclResult_t ret;
    NCCLCHECK(ncclGroupStart());
    ret = metaCommToNccl(ctranRecv(
        recvbuff,
        count,
        ncclToMetaComm(datatype),
        peer,
        meta::comms::ncclx::ncclCommCtran(comm).get(),
        stream,
        algo));
    NCCLCHECK(ncclGroupEnd());
    return ret;
  }
  return std::nullopt;
}

std::optional<ncclResult_t> ctranTryAllToAll(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    cudaStream_t stream) {
  const auto algo = NCCLX_CONFIG_FIELD(comm->config, alltoallAlgo);
  if (algo != NCCL_ALLTOALL_ALGO::orig &&
      ctranAllToAllSupport(
          count,
          ncclToMetaComm(datatype),
          meta::comms::ncclx::ncclCommCtran(comm).get(),
          algo,
          stream,
          recvbuff)) {
    return metaCommToNccl(ctranAllToAll(
        sendbuff,
        recvbuff,
        count,
        ncclToMetaComm(datatype),
        meta::comms::ncclx::ncclCommCtran(comm).get(),
        stream,
        algo));
  }
  return std::nullopt;
}

std::optional<ncclResult_t> ctranTryAllToAllv(
    ncclComm* comm,
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    cudaStream_t stream) {
  if ((NCCLX_CONFIG_FIELD(comm->config, alltoallvAlgo) ==
       NCCL_ALLTOALLV_ALGO::ctran) &&
      ctranAllToAllvSupport(meta::comms::ncclx::ncclCommCtran(comm).get())) {
    return metaCommToNccl(ctranAllToAllv(
        sendbuff,
        sendcounts,
        sdispls,
        recvbuff,
        recvcounts,
        rdispls,
        ncclToMetaComm(datatype),
        meta::comms::ncclx::ncclCommCtran(comm).get(),
        stream));
  }
  return std::nullopt;
}

void ctranTrackDefaultSendRecv(ncclComm* comm) {
  ctranGroupTrackDefaultOp(meta::comms::ncclx::ncclCommCtran(comm).get());
}

ncclResult_t ctranRunDeviceAllToAllv(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    const int64_t* sendcounts_d,
    const int64_t* recvcounts_d,
    ncclDataType_t datatype,
    cudaStream_t stream,
    int64_t sendcountsMultiplier,
    int64_t recvcountsMultiplier,
    const std::unordered_map<std::string, std::string>& hints) {
#if defined(ENABLE_PRIMS)
  if (!ctranDeviceAllToAllvSupport(
          meta::comms::ncclx::ncclCommCtran(comm).get())) {
    CERR(
        commInvalidUsage,
        "deviceAllToAllv requires ctran with pipes transport support");
    return ncclInvalidUsage;
  }
  return metaCommToNccl(ctranDeviceAllToAllv(
      sendbuff,
      recvbuff,
      sendcounts_d,
      recvcounts_d,
      ncclToMetaComm(datatype),
      meta::comms::ncclx::ncclCommCtran(comm).get(),
      stream,
      sendcountsMultiplier,
      recvcountsMultiplier,
      hints));
#else
  return ncclInvalidUsage;
#endif // ENABLE_PRIMS
}

} // namespace ncclx
