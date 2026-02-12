/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "nccl.h"
#include "nvtx_payload_schemas.h"

#include "comms/ctran/Ctran.h"
#include "meta/algoconf/AlgoConfig.h"
#include "meta/collectives/PatAvgHelper.h"
#include "comms/ctran/utils/Checks.h"
#include "meta/wrapper/MetaFactory.h"

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncAlltoAll: return "AlltoAll";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncGather: return "Gather";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncScatter: return "Scatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  default: return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum: return "Sum";
  case ncclDevProd: return "Prod";
  case ncclDevMinMax: return "MinMax";
  case ncclDevPreMulSum: return "PreMulSum";
  case ncclDevSumPostDiv: return "SumPostDiv";
  case ncclDevPatSumPostDiv: return "PatSumPostDiv";
  default: return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: return "ncclInt8";
  case ncclInt32: return "ncclInt32";
  case ncclUint32: return "ncclUint32";
  case ncclInt64: return "ncclInt64";
  case ncclUint64: return "ncclUint64";
  case ncclFloat16: return "ncclFloat16";
  case ncclFloat32: return "ncclFloat32";
  case ncclFloat64: return "ncclFloat64";
  case ncclBfloat16: return "ncclBfloat16";
  case ncclFloat8e4m3: return "ncclFloat8e4m3";
  case ncclFloat8e5m2: return "ncclFloat8e5m2";
  default: return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS: return "NVLS";
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
  case NCCL_ALGO_PAT: return "PAT";
  default: return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  default: return "Unknown";
  }
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (sendcount == 0) {
    return ncclSuccess;
  }
  SetCudaDevRAII setCudaDev(comm->cudaDev);
  // Just pass the size of one message and not the total bytes sent/received.
  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype)));

  // Set algo to global config
  auto algo = NCCL_ALLGATHER_ALGO;
  // Override algo if comm config is set
  if (ctranInitialized(comm->ctranComm_.get())) {
    algo = comm->ctranComm_->ctran_->algo->getAllGatherAlgo();
  }

  // Use ctran allgather if user specified and ctran is supported
  if (algo != NCCL_ALLGATHER_ALGO::orig && ctranAllGatherSupport(comm->ctranComm_.get(), algo)) {
    return metaCommToNccl(ctranAllGather(
        sendbuff, recvbuff, sendcount, ncclToMetaComm(datatype), comm->ctranComm_.get(), stream, algo));
  }

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype)));

  struct ncclInfo info = { ncclFuncAlltoAll, "AlltoAll",
    sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  auto algo = ncclx::algoconf::getAllReduceAlgo();

  // [NCCLX] Redirect to CTRAN if enabled and applicable
  if (algo != NCCL_ALLREDUCE_ALGO::orig && ctranAllReduceSupport(comm->ctranComm_.get(), algo)) {
    return metaCommToNccl(ctranAllReduce(
        sendbuff, recvbuff, count, ncclToMetaComm(datatype), ncclToMetaComm(op), comm->ctranComm_.get(), stream, algo));
  }

  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  if (count == 0) {
    return ncclSuccess;
  }

  SetCudaDevRAII setCudaDev(comm->cudaDev);
  if (NCCL_BROADCAST_ALGO != NCCL_BROADCAST_ALGO::orig &&
      ctranBroadcastSupport(comm->ctranComm_.get(), NCCL_BROADCAST_ALGO)) {
    return metaCommToNccl(ctranBroadcast(
        sendbuff, recvbuff, count, ncclToMetaComm(datatype), root, comm->ctranComm_.get(), stream, NCCL_BROADCAST_ALGO));
  }

  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncGather, "Gather",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    GATHER_CHUNKSTEPS, GATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  SetCudaDevRAII setCudaDev(comm->cudaDev);

  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op));

  if (NCCL_REDUCESCATTER_ALGO != NCCL_REDUCESCATTER_ALGO::orig &&
      ctranReduceScatterSupport(comm->ctranComm_.get(), NCCL_REDUCESCATTER_ALGO)) {
    return metaCommToNccl(ctranReduceScatter(
        sendbuff, recvbuff, recvcount, ncclToMetaComm(datatype), ncclToMetaComm(op), comm->ctranComm_.get(), stream, NCCL_REDUCESCATTER_ALGO));
  }

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };

  // [META:PAT_AVG] Set up infoExt for per-comm PAT AVG control
  // Only for types with enough exponent range (bf16, f32, f64, integers)
  if (comm->usePatAvg_ && op == ncclAvg &&
      ncclx::isPatAvgSupportedType(datatype)) {
    size_t nBytes = recvcount * ncclTypeSize(datatype) * comm->nRanks;
    info.ext = ncclx::setupPatAvgInfoExt(comm, nBytes, datatype);
  }

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncScatter, "Scatter",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

static ncclResult_t baseSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ctranGroupTrackDefaultOp(comm->ctranComm_.get());

  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, exit);
exit:
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  if (count == 0) {
    return ncclSuccess;
  }
  SetCudaDevRAII setCudaDev(comm->cudaDev);

  if ((ncclx::algoconf::getSendRecvAlgo() == NCCL_SENDRECV_ALGO::ctran) &&
      ctranSendRecvSupport(peer, comm->ctranComm_.get())) {
    // ctran send/recvs are enqueued within ctran wherease other non-ctran ones
    // are enqueued in the original queue. When reaching group end, these two
    // groups of ops will be issued separately.
    ncclResult_t ret;
    NCCLCHECK(ncclGroupStart());
    ret = metaCommToNccl(ctranSend(sendbuff, count, ncclToMetaComm(datatype), peer, comm->ctranComm_.get(), stream));
    NCCLCHECK(ncclGroupEnd());
    return ret;
  }

  return baseSend(sendbuff, count, datatype, peer, comm, stream);
}

static ncclResult_t baseRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ctranGroupTrackDefaultOp(comm->ctranComm_.get());

  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, exit);
exit:
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  if (count == 0) {
    return ncclSuccess;
  }
  SetCudaDevRAII setCudaDev(comm->cudaDev);

  if ((ncclx::algoconf::getSendRecvAlgo() == NCCL_SENDRECV_ALGO::ctran) &&
      ctranSendRecvSupport(peer, comm->ctranComm_.get())) {
    // ctran send/recvs are enqueued within ctran wherease other non-ctran ones
    // are enqueued in the original queue. When reaching group end, these two
    // groups of ops will be issued separately.
    ncclResult_t ret;
    NCCLCHECK(ncclGroupStart());
    ret = metaCommToNccl(ctranRecv(recvbuff, count, ncclToMetaComm(datatype), peer, comm->ctranComm_.get(), stream));
    NCCLCHECK(ncclGroupEnd());
    return ret;
  }

  return baseRecv(recvbuff, count, datatype, peer, comm, stream);
}

NCCL_API(
    ncclResult_t,
    ncclAllToAll,
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclAllToAll(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  // Do nothing if count is 0
  if (count == 0) {
    return ncclSuccess;
  }

  SetCudaDevRAII setCudaDev(comm->cudaDev);
  NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", "ncclAllToAll"));
  NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", "ncclAllToAll"));
  if (sendbuff == recvbuff) {
    FB_ERRORRETURN(
        ncclInvalidArgument,
        "Found sendbuff %p == recvbuff %p. In-place ncclAllToAll is not supported.",
        sendbuff,
        recvbuff);
  }

  if ((NCCL_ALLTOALL_ALGO == NCCL_ALLTOALL_ALGO::ctran) &&
      ctranAllToAllSupport(count, ncclToMetaComm(datatype), comm->ctranComm_.get(), NCCL_ALLTOALL_ALGO)) {
    return metaCommToNccl(ctranAllToAll(sendbuff, recvbuff, count, ncclToMetaComm(datatype), comm->ctranComm_.get(), stream, NCCL_ALLTOALL_ALGO));
  }

  // fallback to baseline send/recv based alltoall

  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < comm->nRanks; r++) {
    if (count) {
      NCCLCHECK(baseSend(
          ((char*)sendbuff) + r * count * ncclTypeSize(datatype),
          count,
          datatype,
          r,
          comm,
          stream));
    }
    if (count) {
      NCCLCHECK(baseRecv(
          ((char*)recvbuff) + r * count * ncclTypeSize(datatype),
          count,
          datatype,
          r,
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}


NCCL_API(
    ncclResult_t,
    ncclAllToAllv,
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  SetCudaDevRAII setCudaDev(comm->cudaDev);
  NCCLCHECK(PtrCheck((void*)sendcounts, "sendcounts", "ncclAllToAllv"));
  NCCLCHECK(PtrCheck((void*)recvcounts, "recvcounts", "ncclAllToAllv"));
  // sendbuff/recvbuff can be NULL only if sendcounts/recvcounts is 0
  size_t totalSendCount = 0, totalRecvCount = 0;
  for (int i = 0; i < comm->nRanks; i++) {
    totalSendCount += sendcounts[i];
    totalRecvCount += recvcounts[i];
  }

  if (totalSendCount > 0) {
    NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", "ncclAllToAllv"));
  }
  if (totalRecvCount > 0) {
    NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", "ncclAllToAllv"));
  }

  if (totalSendCount && totalRecvCount && sendbuff == recvbuff) {
    FB_ERRORRETURN(
        ncclInvalidArgument,
        "Found sendbuff %p == recvbuff %p. In-place ncclAllToAllv is not supported.",
        sendbuff,
        recvbuff);
  }

  if ((ncclx::algoconf::getAllToAllVAlgo() == NCCL_ALLTOALLV_ALGO::ctran) &&
      ctranAllToAllvSupport(comm->ctranComm_.get())) {
    return metaCommToNccl(ctranAllToAllv(
        sendbuff,
        sendcounts,
        sdispls,
        recvbuff,
        recvcounts,
        rdispls,
        ncclToMetaComm(datatype),
        comm->ctranComm_.get(),
        stream));
  }

  // fallback to baseline send/recv based alltoallv
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < comm->nRanks; r++) {
    NCCLCHECK(baseSend(
        ((char*)sendbuff) + sdispls[r] * ncclTypeSize(datatype),
        sendcounts[r],
        datatype,
        r,
        comm,
        stream));
    NCCLCHECK(baseRecv(
        ((char*)recvbuff) + rdispls[r] * ncclTypeSize(datatype),
        recvcounts[r],
        datatype,
        r,
        comm,
        stream));
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

__attribute__((visibility("default")))
ncclResult_t ncclx::alltoallvDynamic(
    const void * const* sendbuffs,
    const size_t* sendcounts,
    void * const* recvbuffs,
    size_t maxSendcount,
    size_t maxRecvcount,
    size_t* actualRecvcounts,
    const ncclx::Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {

  NCCLCHECK(metaCommToNccl(ctranAllToAllvDynamicSupport(comm->ctranComm_.get(), ncclToMetaComm(hints), maxSendcount, maxRecvcount, ncclToMetaComm(datatype))));

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


__attribute__((visibility("default")))
ncclResult_t ncclx::alltoallvDynamicSplit(
    const void* sendbuff,
    const size_t* sendSplitLengths,
    void* const* recvbuffs,
    size_t maxSendcount,
    size_t maxRecvcount,
    size_t* actualRecvcounts,
    const ncclx::Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {

  NCCLCHECK(metaCommToNccl(ctranAllToAllvDynamicSupport(comm->ctranComm_.get(), ncclToMetaComm(hints), maxSendcount, maxRecvcount, ncclToMetaComm(datatype))));

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

__attribute__((visibility("default")))
ncclResult_t ncclx::alltoallvDynamicSplitNonContig(
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
  const ncclx::Hints& hints,
  ncclDataType_t datatype,
  ncclComm_t comm,
  cudaStream_t stream) {

  NCCLCHECK(metaCommToNccl(ctranAllToAllvDynamicSupport(comm->ctranComm_.get(), ncclToMetaComm(hints), maxSendcount, maxRecvcount, ncclToMetaComm(datatype))));

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

__attribute__((visibility("default")))
ncclResult_t ncclx::alltoallvDynamicDispatch(
  const void* sendbuff,
  const size_t* sendSplitLengths,
  size_t numSendSplitLengths,
  const size_t* sendIndices,
  const size_t* sendIndicesBlockLengths,
  void* const* recvbuffs,
  size_t* recvAllSplitLengths,
  size_t maxSendcount,
  size_t maxRecvcount,
  const ncclx::Hints& hints,
  ncclDataType_t datatype,
  ncclComm_t comm,
  cudaStream_t stream) {

  NCCLCHECK(metaCommToNccl(ctranAllToAllvDynamicSupport(comm->ctranComm_.get(), ncclToMetaComm(hints), maxSendcount, maxRecvcount, ncclToMetaComm(datatype))));

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

__attribute__((visibility("default")))
ncclResult_t ncclx::alltoallvDynamicCombine(
  const void* sendbuff,
  const size_t* sendSplitLengths,
  size_t numSendSplitLengths,
  const size_t* sendIndices,
  const size_t* sendIndicesBlockLengths,
  void* recvbuff,
  size_t maxSendcount,
  size_t maxRecvcount,
  const ncclx::Hints& hints,
  ncclDataType_t datatype,
  ncclComm_t comm,
  cudaStream_t stream) {

  NCCLCHECK(metaCommToNccl(ctranAllToAllvDynamicSupport(comm->ctranComm_.get(), ncclToMetaComm(hints), maxSendcount, maxRecvcount, ncclToMetaComm(datatype))));

  return metaCommToNccl(ctranAlltoallvDynamicSplitNonContig(
      sendbuff,
      sendSplitLengths,
      numSendSplitLengths,
      sendIndices,
      sendIndicesBlockLengths,
      nullptr,
      recvbuff,
      maxSendcount,
      maxRecvcount,
      ncclToMetaComm(hints),
      ncclToMetaComm(datatype),
      comm->ctranComm_.get(),
      stream,
      true));
}
