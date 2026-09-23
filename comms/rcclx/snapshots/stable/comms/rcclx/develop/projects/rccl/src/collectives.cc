/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "graph/topo.h"
#include "nccl.h"
#include "api_trace.h"
#include "AlgoUtils.h"
#include "nvtx_payload_schemas.h"
#include "device/hierarchical_ag_shuffle.h"
#include "dda_all_reduce_ipc.h"
#include "dda_reduce_scatter_ipc.h"
#include "dda_all_gather_ipc.h"
#include "dda_alltoall_ipc.h"
#include "meta/lpcoll/low_precision_allgather.h"
#include "meta/lpcoll/low_precision_allreduce.h"
#include "meta/lpcoll/low_precision_alltoall.h"
#include "meta/lpcoll/low_precision_reduce_scatter.h"
#include "meta/lpcoll/p2p_allgather.h"
#include "meta/relay/sharded_relay_allreduce.h"
#include "meta/relay/sharded_relay_reduce_scatter.h"
#include "meta/relay/sharded_relay_all_to_all.h"
#include "meta/relay/sharded_relay_all_gather.h"
#include "meta/relay/relay_control.h"

#include <cstddef>
#include <cstring>
#include <type_traits>
#include "comms/ctran/Ctran.h"
#include "MetaFactory.h"

#ifdef ENABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>
#endif

using namespace rccl;

// 16MB threshold for low precision collectives
#define LOW_PRECISION_MSG_SIZE_THRESHOLD (16 * 1024 * 1024)

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
  case ncclFuncPutSignal: return "PutSignal";
  case ncclFuncSignal: return "Signal";
  case ncclFuncWaitSignal: return "WaitSignal";
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

// Direct AllGather: posts Send/Recv to every peer (including self) for every
// rank using the same iteration order on all ranks. Mirrors the AlltoAll
// scheduling path so that, when RCCL_P2P_BATCH_ENABLE=1, all ranks emit the
// same sequence of (sendRank, recvRank) rounds and therefore the same fused
// ncclDevWorkBatch composition.
//
// We deliberately keep this minimal:
//   - No (rank + r) % nRanks rotation: all ranks visit peers in order 0..N-1.
//   - No in-place self-peer skip: send/recv to self is always posted; the
//     device kernel handles isCopy = (sendRank == self) as a local memcpy
//     (see device/sendrecv.h).
//   - Posting is delegated to taskAppend() via a single ncclEnqueueCheck
//     call. taskAppend() then loops once and calls p2pTaskAppend directly,
//     the same way ncclAlltoAll does, avoiding the per-peer ncclSend/ncclRecv
//     overhead (ArgsCheck, Recorder, profiler events, group start/end
//     internal) that previously made cross-rank batch composition fragile at
//     scale.
static ncclResult_t rcclDirectAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream,
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS, nullptr };
  info.useDirect = true;
  return ncclEnqueueCheck(&info);
}

RCCL_PARAM(DdaEnable, "DDA_ENABLE", 0);
RCCL_PARAM(DdaThreshold, "DDA_THRESHOLD", (size_t)(67108864));

// Returns true when the DDA fast path should be attempted for a collective
// with the given total byte count.  gfx942Default is the per-collective
// threshold for gfx942; gfx950 uses the user-configurable rcclParamDdaThreshold();
// all other architectures return false (threshold 0).
static bool rcclDdaEnabled(const ncclComm* comm, size_t totalBytes, size_t gfx942Default) {
  if (!rcclParamDdaEnable() || ncclParamLaunchOrderImplicit() || ncclGroupDepth != 0 || comm->nRanks < 8 || comm->symmetricSupport) return false;
  size_t threshold;
  if (IsArchMatch(comm->archName, "gfx942")) {
    threshold = gfx942Default;
  } else if (IsArchMatch(comm->archName, "gfx950")) {
    threshold = (size_t)rcclParamDdaThreshold();
  } else {
    return false;
  }
  return threshold > 0 && totalBytes <= threshold;
}

enum rcclAllGatherAlgo {
  RCCL_AG_RING,
  RCCL_AG_DIRECT,
  RCCL_AG_HIERARCHICAL
};

static rcclAllGatherAlgo rcclSelectAllGatherAlgo(struct ncclComm* comm, size_t msgSize) {
  if (ncclGroupDepth == 0 && rcclUseHierarchicalAllGather(comm, msgSize)) {
    return RCCL_AG_HIERARCHICAL;
  }
  if (rcclUseAllGatherDirect(comm, msgSize)) {
    return RCCL_AG_DIRECT;
  }
  return RCCL_AG_RING;
}

static inline int hierarchicalShuffleNumBlocks(size_t totalBytes) {
  if (totalBytes <= (size_t)64 * 1024)
    return 8;
  if (totalBytes <= (size_t)16 * 1024 * 1024)
    return 16;
  return 32;
}

static ncclResult_t ncclHierarchicalAllGather_Impl(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (sendcount == 0) return ncclSuccess;
  ncclComm* intraComm = comm->hierarchicalIntraComm;
  ncclComm* interComm = comm->hierarchicalInterComm;
  int localRanks = intraComm->nRanks; // Ranks per node
  int nNodes = interComm->nRanks; // Number of nodes
  size_t typeSize = ncclTypeSize(datatype);

  void* tempBuffer = comm->hierarchicalAGTempBuffer;
  const void* interSendBuff = sendbuff;
  size_t rankOffset = sendcount * typeSize;
  if (sendbuff == ((char*)recvbuff) + comm->rank * rankOffset) {
    CUDACHECK(hipMemcpyAsync(tempBuffer, sendbuff, rankOffset, hipMemcpyDeviceToDevice, stream));
    interSendBuff = tempBuffer;
  }

  // Step 1: Inter-node AllGather
  size_t interMsgSize = sendcount * nNodes * typeSize;
  if (nNodes <= 16 && rcclUseAllGatherDirect(interComm, interMsgSize)) {
    NCCLCHECK(rcclDirectAllGather(interSendBuff, recvbuff, sendcount, datatype, interComm, stream));
  } else {
    struct ncclInfo infoInterAG = { ncclFuncAllGather, "HierarchicalAllGather-Inter",
      interSendBuff, recvbuff, sendcount, datatype, ncclSum, 0, interComm, stream,
      ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS, nullptr };
    NCCLCHECK(ncclEnqueueCheck(&infoInterAG));
  }

  // Step 2: Intra-node AllGather
  size_t intraSendCount = sendcount * nNodes;
  size_t intraMsgSize = intraSendCount * typeSize * localRanks;
  if (rcclUseAllGatherDirect(intraComm, intraMsgSize)) {
    // Use direct allgather
    NCCLCHECK(rcclDirectAllGather(recvbuff, tempBuffer, intraSendCount, datatype, intraComm, stream));
  } else {
    struct ncclInfo infoIntraAG = { ncclFuncAllGather, "HierarchicalAllGather-Intra",
      recvbuff, tempBuffer, intraSendCount, datatype, ncclSum, 0, intraComm, stream,
      ALLGATHER_CHUNKSTEPS,
      intraComm->rcclUseOneSlice ? ALLGATHER_SLICESTEPS_SINGLE_NODE : ALLGATHER_SLICESTEPS, nullptr
    };
    NCCLCHECK(ncclEnqueueCheck(&infoIntraAG));
  }

  // Step 3: Shuffle tempBuffer to recvbuff
  size_t totalAGBytes = (size_t)nNodes * localRanks * rankOffset;
  int numBlocks = hierarchicalShuffleNumBlocks(totalAGBytes);
  int threadsPerBlock = 1024;
  hierarchicalAGShuffle<<<numBlocks, threadsPerBlock, 0, stream>>>(
    (const char*)tempBuffer, (char*)recvbuff, rankOffset, nNodes, localRanks);
  CUDACHECK(hipGetLastError());

  return ncclSuccess;
}

ncclResult_t ncclAllGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // Check if low precision is enabled
  if (isLowPrecisionFp8E4M3AllGatherEnabled()) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t messageSize = nRanks * sendcount * ncclTypeSize(datatype);

    if ((messageSize >= LOW_PRECISION_MSG_SIZE_THRESHOLD) &&
        (datatype == ncclFloat32 || datatype == ncclBfloat16)) {
      // Use low precision (quantized) allgather for large float32 messages
      TRACE(
          NCCL_COLL,
          "Using quantized ARG allgather (FP8 E4M3) for float32 data");
      return ncclLowPrecisionAllGather(
          sendbuff, recvbuff, sendcount, datatype, comm, stream);
    } else {
      // Use P2P allgather for all other cases when low precision is enabled
      TRACE(
          NCCL_COLL,
          "Using P2P AllGather (low precision enabled but using P2P: msg_size=%zu, threshold=%zu)",
          messageSize,
          LOW_PRECISION_MSG_SIZE_THRESHOLD);
      return ncclP2PAllGather(
          sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }
  }

#ifdef BUILD_META_INTERNAL
  if (comm->algoFactory) {
    // try to get meta customized algo
    auto algo = comm->algoFactory->getAllGatherAlgo(
        sendbuff, recvbuff, sendcount, meta::comms::ncclToMetaComm(datatype), stream);
    if (algo) {
      try {
        algo->allGather();
      } catch (const std::exception& e) {
        WARN("failed to launch custom all gather: %s", e.what());
        return ncclInternalError;
      }
      return ncclSuccess;
    }
  }
#endif

  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype), datatype));
    // RCCL update slice steps for AllGather if single node
    const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");

    int chunkSteps = (isGfx950 && comm->rcclUseOneSlice)? 1 : ALLGATHER_CHUNKSTEPS;
    int sliceSteps = comm->rcclUseOneSlice
      ? (isGfx950 ? 1 : ALLGATHER_SLICESTEPS_SINGLE_NODE)
      : ALLGATHER_SLICESTEPS;
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    chunkSteps, sliceSteps, nullptr };
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  size_t msgSize = sendcount * ncclTypeSize(datatype) * nRanks;

  NCCLCHECK(Recorder::instance().record(rrAllGather, info));

  if (rcclDdaEnabled(comm, nRanks * sendcount * ncclTypeSize(datatype), 8388608) &&
      ncclAllGatherDdaIpcEligible(comm, sendbuff, recvbuff, sendcount, datatype)) {
    NCCLCHECK(ncclAllGatherDdaIpc(
        sendbuff,
        recvbuff,
        sendcount,
        datatype,
        comm,
        stream));
    return ncclSuccess;
  }
  rcclAllGatherAlgo algo = rcclSelectAllGatherAlgo(comm, msgSize);
  switch (algo) {
    case RCCL_AG_HIERARCHICAL:
    return ncclHierarchicalAllGather_Impl(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    case RCCL_AG_DIRECT:
    INFO(NCCL_INIT, "RCCL DIRECT ALLGATHER count = %zu, msgSize = %zu, comm = %p, stream = %p, rank = %d, sendbuff = %p, recvbuff = %p",
      sendcount, msgSize, comm, stream, rank, sendbuff, recvbuff);
    // Use direct allgather (only when not in a group; in-group use Ring so
    // ncclGroupSimulateEnd gets estimatedTime).
    if (sendcount == 0) return ncclSuccess;
    // Mark the info so taskAppend posts this as A2A-style per-peer Send/Recv
    // P2P tasks (no peer rotation, no in-place self skip).
    info.useDirect = true;
    return ncclEnqueueCheck(&info);
    case RCCL_AG_RING:
    default:
      return ncclEnqueueCheck(&info);
  }
}

RCCL_PARAM(AlltoAllPivotEnable, "ALL_TO_ALL_PIVOT_ENABLE", 0);

NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  // Check for quantized ARG alltoall via environment variable
  if (isLowPrecisionFp8E4M3AllToAllEnabled() && (datatype == ncclFloat32 || datatype == ncclBfloat16)) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    if (nRanks * count * ncclTypeSize(datatype) >=
        LOW_PRECISION_MSG_SIZE_THRESHOLD) {
      TRACE(
          NCCL_COLL,
          "Using quantized ARG alltoall (FP8 E4M3) for float32 data");
      return ncclLowPrecisionAllToAll(
          sendbuff, recvbuff, count, datatype, comm, stream);
    }
  }

#ifdef BUILD_META_INTERNAL
  if (comm->algoFactory) {
    // try to get meta customized algo
    auto algo = comm->algoFactory->getAllToAllAlgo(
        sendbuff, recvbuff, count, meta::comms::ncclToMetaComm(datatype), stream);
    if (algo) {
      try {
        algo->allToAll();
      } catch (const std::exception& e) {
        WARN("failed to launch custom all-to-all: %s", e.what());
        return ncclInternalError;
      }
      return ncclSuccess;
    }
  }
#endif

  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), datatype));
  
  NCCLCHECK(Recorder::instance().record(rrAllToAll, sendbuff, recvbuff, count, datatype, comm, stream));

  size_t rankOffset = count * ncclTypeSize(datatype);
  size_t rankAlign = rankOffset & ((~rankOffset) + 1);

  struct ncclInfo info;
  if (comm->topo->pivotA2AEnabled && comm->nChannels >= comm->topo->pivotA2ANumBiRings * 2 &&
      rankOffset >= 744 * 1024 && rankAlign != 4 && rcclParamAlltoAllPivotEnable()) {
      info = { ncclFuncAlltoAllPivot, "AlltoAllPivot",
        sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
        ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS, nullptr };
  } else {
      #ifdef ENABLE_ROCSHMEM
      size_t msgSize = count * ncclTypeSize(datatype) * comm->nRanks;
      if (rcclUseAlltoAllGda(comm) && msgSize <= comm->rocshmemThreshold) {	
        struct ncclInfo info = { ncclFuncAlltoAllGda, "AlltoAllGda",
              sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream,
              ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS, nullptr };
            
        return ncclEnqueueCheck(&info);
      }
      #endif // ENABLE_ROCSHMEM

    if (rcclDdaEnabled(comm, comm->nRanks * count * ncclTypeSize(datatype), 4194304) &&
        ncclAllToAllDdaIpcEligible(comm, sendbuff, recvbuff, count, datatype)) {
      NCCLCHECK(ncclAllToAllDdaIpc(
        sendbuff,
        recvbuff,
        count,
        datatype,
        comm,
        stream));
      return ncclSuccess;
    }

    info = { ncclFuncAlltoAll, "AlltoAll",
      sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS };
  }
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAlltoAllv, const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclAlltoAllv_impl(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAllv, NcclNvtxParamsAlltoAllv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcounts[comm->rank] * ncclTypeSize(datatype),
      recvcounts[comm->rank] * ncclTypeSize(datatype), datatype));

  NCCLCHECK(Recorder::instance().record(rrAllToAllv, sendbuff, recvbuff, 0, datatype, comm, stream, -1, sendcounts, sdispls, recvcounts, rdispls));

  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  std::vector<size_t> sdispls1(nRanks);
  std::vector<size_t> rdispls1(nRanks);
  std::vector<size_t> sendcounts1(nRanks);
  std::vector<size_t> recvcounts1(nRanks);

  std::vector<size_t> sizes(4*nRanks);	//4 for sdispl, rdispl, scount, rcount
#ifdef ENABLE_ROCSHMEM
    for (int i = 0; i < nRanks; i++) {
       sdispls1[i] = sdispls[i] * ncclTypeSize(datatype);
       rdispls1[i] = rdispls[i] * ncclTypeSize(datatype);
       sendcounts1[i] = sendcounts[i] * ncclTypeSize(datatype);
       recvcounts1[i] = recvcounts[i] * ncclTypeSize(datatype);
    }

    size_t count = sdispls1[nRanks - 1] + sendcounts1[nRanks - 1];

    if (comm->enableRocshmem && comm->nNodes > 1 && (comm->nRanks/comm->nNodes == 8)) {
        INFO(NCCL_INIT, "GDA alltoallv is supported for up to 128MB message size; Use ROCSHMEM_HEAP_SIZE=3GB for GDA support till 512MB");  

        for (int i = 0; i < nRanks; i++) {
            sizes[i] = sendcounts1[i];
            sizes[nRanks + i] = sdispls1[i];
            sizes[2*nRanks + i] = recvcounts1[i];
            sizes[3*nRanks + i] = rdispls1[i];
        }
        count = count / ncclTypeSize(datatype);

	//use CU for copy-in/copy-out for small <= 128KB sizes
	//TODO: the threshold could be different for different number of nodes
	if ((count * ncclTypeSize(datatype)) > 131072) {
	    void *dest = (char*)comm->sourceRshmem + comm->symId * comm->bufThreshold;
            CUDACHECK(hipMemcpyAsync(dest, sendbuff, count * ncclTypeSize(datatype),
               hipMemcpyDeviceToDevice, stream));
        }
        struct ncclInfo info = { ncclFuncAlltoAllvGda, "AlltoAllvGda",
        sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream,
        ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS, nullptr };
#ifdef ENABLE_ROCSHMEM
        info.sizes = sizes.data();
#endif

        ncclResult_t ret = ncclEnqueueCheck(&info);

        if (ret == ncclSuccess && ((count * ncclTypeSize(datatype)) > 131072)) {
	    void *src = (char*)comm->destRshmem + comm->symId * comm->bufThreshold;
            CUDACHECK(hipMemcpyAsync(recvbuff, src, count * ncclTypeSize(datatype),
                    hipMemcpyDeviceToDevice, stream));
            comm->symId = (comm->symId + 1) % comm->numSymBuf;
        }
        return ret;
    }
#endif

  Recorder::instance().skip(true);
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nRanks; r++) {
    NCCLCHECK(ncclSend(
        ((char*)sendbuff) + sdispls[r]*ncclTypeSize(datatype),
        sendcounts[r],
        datatype,
        r,
        comm,
        stream));
    NCCLCHECK(ncclRecv(
        ((char*)recvbuff) + rdispls[r]*ncclTypeSize(datatype),
        recvcounts[r],
        datatype,
        r,
        comm,
        stream));
  }
  NCCLCHECK(ncclGroupEnd());
  Recorder::instance().skip(false);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);


ncclResult_t ncclAllReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
// Check for quantized ARG allreduce via environment variable
if (isLowPrecisionFp8E4M3AllReduceEnabled() && (datatype == ncclFloat32 || datatype == ncclBfloat16) &&
    op == ncclSum &&
    count * ncclTypeSize(datatype) >= LOW_PRECISION_MSG_SIZE_THRESHOLD) {
  TRACE(
      NCCL_COLL,
      "Using quantized ARG allreduce (FP8 E4M3) for float32 sum reduction");
  return ncclLowPrecisionAllReduce(
      sendbuff, recvbuff, count, datatype, op, comm, stream);
}

  if (comm->algoFactory && op == ncclSum) {
    // try to get meta customized algo
    auto algo = comm->algoFactory->getAllReduceAlgo(
        sendbuff, recvbuff, count, meta::comms::ncclToMetaComm(datatype), stream);
    if (algo) {
      try {
        algo->allReduce();
      } catch (const std::exception& e) {
        WARN("failed to launch custom all reduce: %s", e.what());
        return ncclInternalError;
      }
      return ncclSuccess;
    }
  }

  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op, datatype));

  // RCCL update slice steps for AllReduce if single node
  const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");
  int chunkSteps = (isGfx950 && comm->rcclUseOneSlice)? 1 : ALLREDUCE_CHUNKSTEPS;
  int sliceSteps = comm->rcclUseOneSlice
      ? (isGfx950 ? 1 : ALLREDUCE_SLICESTEPS_SINGLE_NODE)
      : ALLREDUCE_SLICESTEPS;

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    chunkSteps, sliceSteps, nullptr };

  NCCLCHECK(Recorder::instance().record(rrAllReduce, info));

  if (rcclDdaEnabled(comm, count * ncclTypeSize(datatype), 8388608) &&
      ncclAllReduceDdaIpcEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    NCCLCHECK(ncclAllReduceDdaIpc(
        sendbuff,
        recvbuff,
        count,
        datatype,
        op,
        comm,
        stream));
    return ncclSuccess;
  }

  return ncclEnqueueCheck(&info);
}

ncclResult_t ncclAllReduceWithBias_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, const void* acc) {
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op, datatype));

  if (acc == nullptr) {
    WARN("ncclAllReduceWithBias : acc cannot be nullptr");
    return ncclInvalidArgument;
  }
  // DDA (Direct Device Access) custom algo is only used for plain AllReduce
  // (acc == nullptr). When a bias accumulator is provided (acc != nullptr),
  // DDA applies the bias as the initial accumulator before summing all ranks,
  // while the RCCL ring kernel adds it after the reduction. These two orderings
  // produce different bfloat16 rounding results, causing test failures.
  // Always fall through to the RCCL kernel path for AllReduceWithBias.
  if (comm->algoFactory && op == ncclSum && acc == nullptr) {
    // try to get meta customized algo
    auto algo = comm->algoFactory->getAllReduceAlgo(
        sendbuff, recvbuff, count, meta::comms::ncclToMetaComm(datatype), stream, acc);
    if (algo) {
      try {
        algo->allReduce();
      } catch (const std::exception& e) {
        WARN("failed to launch custom all reduce: %s", e.what());
        return ncclInternalError;
      }
      return ncclSuccess;
    }
  }

  // RCCL update slice steps for AllReduceBias if single node
  // similar to changes made to AllReduce earlier
  const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");
  int chunkSteps = (isGfx950 && comm->rcclUseOneSlice) ? 1 : ALLREDUCE_CHUNKSTEPS;
  int sliceSteps = comm->rcclUseOneSlice
      ? (isGfx950 ? 1 : ALLREDUCE_SLICESTEPS_SINGLE_NODE)
      : ALLREDUCE_SLICESTEPS;

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    chunkSteps, sliceSteps, acc };

  NCCLCHECK(Recorder::instance().record(rrAllReduceWithBias, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclShardedRelayMultiGroupAllReduce, 
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* counts,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision);

ncclResult_t ncclShardedRelayMultiGroupAllReduce_impl(
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* counts,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision) {
  // Validate operation - only SUM and AVG are supported
  if (op != ncclSum && op != ncclAvg) {
    WARN("ncclShardedRelayMultiGroupAllReduce: only ncclSum and ncclAvg operations are supported");
    return ncclInvalidArgument;
  }

  // Validate buffer pointers
  if (recvBuffs == nullptr || allActiveRanks == nullptr || counts == nullptr ||
      sendBuffs == nullptr) {
    WARN("ncclShardedRelayMultiGroupAllReduce: buffer, counts, and activeRanks pointers must be non-null");
    return ncclInvalidArgument;
  }

  // Validate group parameters
  if (nGroups < 1 || nGroups > 8) {
    WARN("ncclShardedRelayMultiGroupAllReduce: nGroups must be between 1 and 8, got %d", nGroups);
    return ncclInvalidArgument;
  }

  if (nActiveRanksPerGroup != 2 && nActiveRanksPerGroup != 4) {
    WARN("ncclShardedRelayMultiGroupAllReduce: nActiveRanksPerGroup must be 2 or 4, got %d", 
         nActiveRanksPerGroup);
    return ncclInvalidArgument;
  }

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  
  // Validate we have enough ranks for sharded relay (need helpers)
  if (nRanks < nActiveRanksPerGroup + 1) {
    WARN("ncclShardedRelayMultiGroupAllReduce: need at least %d ranks for %d active ranks (need helpers)", 
         nActiveRanksPerGroup + 1, nActiveRanksPerGroup);
    return ncclInvalidArgument;
  }

  TRACE(NCCL_COLL, "Using Sharded Relay Multi-Group AllReduce (nRanks=%d, nActiveRanksPerGroup=%d, nGroups=%d)", 
        nRanks, nActiveRanksPerGroup, nGroups);
  return ncclShardedRelayMultiGroupAllReduceImpl(
      sendBuffs, recvBuffs, counts,
      datatype, op, comm, stream, allActiveRanks, nActiveRanksPerGroup, nGroups,
      lowPrecision);
}

ncclResult_t ncclShardedRelayMultiGroupAllReduce(
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* counts,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision) {
  return ncclShardedRelayMultiGroupAllReduce_impl(
      sendBuffs, recvBuffs, counts,
      datatype, op, comm, stream, allActiveRanks, nActiveRanksPerGroup, nGroups,
      lowPrecision);
}

NCCL_API(ncclResult_t, ncclShardedRelayMultiGroupReduceScatter,
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* recvCounts,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision);

ncclResult_t ncclShardedRelayMultiGroupReduceScatter_impl(
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* recvCounts,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision) {
  // Validate operation - only SUM and AVG are supported
  if (op != ncclSum && op != ncclAvg) {
    WARN("ncclShardedRelayMultiGroupReduceScatter: only ncclSum and ncclAvg operations are supported");
    return ncclInvalidArgument;
  }

  // Validate buffer pointers
  if (recvBuffs == nullptr || allActiveRanks == nullptr || recvCounts == nullptr ||
      sendBuffs == nullptr) {
    WARN("ncclShardedRelayMultiGroupReduceScatter: buffer, recvCounts, and activeRanks pointers must be non-null");
    return ncclInvalidArgument;
  }

  // Validate group parameters
  if (nGroups < 1 || nGroups > 8) {
    WARN("ncclShardedRelayMultiGroupReduceScatter: nGroups must be between 1 and 8, got %d", nGroups);
    return ncclInvalidArgument;
  }

  if (nActiveRanksPerGroup != 2 && nActiveRanksPerGroup != 4) {
    WARN("ncclShardedRelayMultiGroupReduceScatter: nActiveRanksPerGroup must be 2 or 4, got %d",
         nActiveRanksPerGroup);
    return ncclInvalidArgument;
  }

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));

  // Validate we have enough ranks for sharded relay (need helpers)
  if (nRanks < nActiveRanksPerGroup + 1) {
    WARN("ncclShardedRelayMultiGroupReduceScatter: need at least %d ranks for %d active ranks (need helpers)",
         nActiveRanksPerGroup + 1, nActiveRanksPerGroup);
    return ncclInvalidArgument;
  }

  TRACE(NCCL_COLL, "Using Sharded Relay Multi-Group ReduceScatter (nRanks=%d, nActiveRanksPerGroup=%d, nGroups=%d)",
        nRanks, nActiveRanksPerGroup, nGroups);
  return ncclShardedRelayMultiGroupReduceScatterImpl(
      sendBuffs, recvBuffs, recvCounts,
      datatype, op, comm, stream, allActiveRanks, nActiveRanksPerGroup, nGroups,
      lowPrecision);
}

ncclResult_t ncclShardedRelayMultiGroupReduceScatter(
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* recvCounts,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision) {
  return ncclShardedRelayMultiGroupReduceScatter_impl(
      sendBuffs, recvBuffs, recvCounts,
      datatype, op, comm, stream, allActiveRanks, nActiveRanksPerGroup, nGroups,
      lowPrecision);
}

NCCL_API(ncclResult_t, ncclShardedRelayMultiGroupAllToAll,
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* segmentCounts,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision);

ncclResult_t ncclShardedRelayMultiGroupAllToAll_impl(
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* segmentCounts,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision) {
  // Validate buffer pointers
  if (recvBuffs == nullptr || allActiveRanks == nullptr || segmentCounts == nullptr ||
      sendBuffs == nullptr) {
    WARN("ncclShardedRelayMultiGroupAllToAll: buffer, segmentCounts, and activeRanks pointers must be non-null");
    return ncclInvalidArgument;
  }

  // Validate group parameters
  if (nGroups < 1 || nGroups > 8) {
    WARN("ncclShardedRelayMultiGroupAllToAll: nGroups must be between 1 and 8, got %d", nGroups);
    return ncclInvalidArgument;
  }

  if (nActiveRanksPerGroup != 2 && nActiveRanksPerGroup != 4) {
    WARN("ncclShardedRelayMultiGroupAllToAll: nActiveRanksPerGroup must be 2 or 4, got %d",
         nActiveRanksPerGroup);
    return ncclInvalidArgument;
  }

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));

  // Validate we have enough ranks for sharded relay (need helpers)
  if (nRanks < nActiveRanksPerGroup + 1) {
    WARN("ncclShardedRelayMultiGroupAllToAll: need at least %d ranks for %d active ranks (need helpers)",
         nActiveRanksPerGroup + 1, nActiveRanksPerGroup);
    return ncclInvalidArgument;
  }

  TRACE(NCCL_COLL, "Using Sharded Relay Multi-Group AllToAll (nRanks=%d, nActiveRanksPerGroup=%d, nGroups=%d)",
        nRanks, nActiveRanksPerGroup, nGroups);
  return ncclShardedRelayMultiGroupAllToAllImpl(
      sendBuffs, recvBuffs, segmentCounts,
      datatype, comm, stream, allActiveRanks, nActiveRanksPerGroup, nGroups,
      lowPrecision);
}

ncclResult_t ncclShardedRelayMultiGroupAllToAll(
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* segmentCounts,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision) {
  return ncclShardedRelayMultiGroupAllToAll_impl(
      sendBuffs, recvBuffs, segmentCounts,
      datatype, comm, stream, allActiveRanks, nActiveRanksPerGroup, nGroups,
      lowPrecision);
}

NCCL_API(ncclResult_t, ncclShardedRelayMultiGroupAllGather,
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* sendCounts,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision);

ncclResult_t ncclShardedRelayMultiGroupAllGather_impl(
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* sendCounts,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision) {
  // Validate buffer pointers
  if (recvBuffs == nullptr || allActiveRanks == nullptr || sendCounts == nullptr ||
      sendBuffs == nullptr) {
    WARN("ncclShardedRelayMultiGroupAllGather: buffer, sendCounts, and activeRanks pointers must be non-null");
    return ncclInvalidArgument;
  }

  // Validate group parameters
  if (nGroups < 1 || nGroups > 8) {
    WARN("ncclShardedRelayMultiGroupAllGather: nGroups must be between 1 and 8, got %d", nGroups);
    return ncclInvalidArgument;
  }

  if (nActiveRanksPerGroup != 2 && nActiveRanksPerGroup != 4) {
    WARN("ncclShardedRelayMultiGroupAllGather: nActiveRanksPerGroup must be 2 or 4, got %d",
         nActiveRanksPerGroup);
    return ncclInvalidArgument;
  }

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));

  // Validate we have enough ranks for sharded relay (need helpers)
  if (nRanks < nActiveRanksPerGroup + 1) {
    WARN("ncclShardedRelayMultiGroupAllGather: need at least %d ranks for %d active ranks (need helpers)",
         nActiveRanksPerGroup + 1, nActiveRanksPerGroup);
    return ncclInvalidArgument;
  }

  TRACE(NCCL_COLL, "Using Sharded Relay Multi-Group AllGather (nRanks=%d, nActiveRanksPerGroup=%d, nGroups=%d)",
        nRanks, nActiveRanksPerGroup, nGroups);
  return ncclShardedRelayMultiGroupAllGatherImpl(
      sendBuffs, recvBuffs, sendCounts,
      datatype, comm, stream, allActiveRanks, nActiveRanksPerGroup, nGroups,
      lowPrecision);
}

ncclResult_t ncclShardedRelayMultiGroupAllGather(
    const void* const* sendBuffs, void* const* recvBuffs, const size_t* sendCounts,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
    const int* const* allActiveRanks, int nActiveRanksPerGroup, int nGroups,
    int lowPrecision) {
  return ncclShardedRelayMultiGroupAllGather_impl(
      sendBuffs, recvBuffs, sendCounts,
      datatype, comm, stream, allActiveRanks, nActiveRanksPerGroup, nGroups,
      lowPrecision);
}

// The ABI probe for consumers that bind the four entry points above through
// ctypes, with no compiler and no linker in the chain to notice that their
// signatures changed. See nccl.h.in for why one exists at all.
NCCL_API(int, ncclShardedRelayAbiVersion, void);

int ncclShardedRelayAbiVersion_impl(void) {
  return NCCL_SHARDED_RELAY_ABI_VERSION;
}

int ncclShardedRelayAbiVersion(void) {
  return ncclShardedRelayAbiVersion_impl();
}

// Host control plane for the relay collectives. Two functions and one struct is
// the entire surface: everything else rides hooks that already exist. Setup and
// teardown are not exported because commInitRank and commFree already run them,
// capacity is not queryable because it is validated at attach so a mismatch
// fails at init rather than at runtime, abort is folded into Consume's return,
// and shutdown is an opCode rather than a third entry point.
//
// Both take ncclComm_t instead of a handle, so there is no object for a caller
// to own, thread through, or leak.

NCCL_API(ncclResult_t, ncclRelayControlPublish,
    ncclComm_t comm, uint64_t epoch,
    const ncclRelayPlanInfo* info, const size_t* counts, int64_t timeoutNs);

// The exported record and the internal one are the SAME 32-byte wire format, so
// the conversions below are a memcpy rather than a field list. This is the only
// translation unit where both types are visible, which makes it the only place
// these assertions can live -- and a memcpy behind them cannot silently drop a
// field the way a hand-written field list does the moment one is added to both.
static_assert(
    sizeof(ncclRelayPlanInfo) == sizeof(rcclx::relay::RelayPlanInfo),
    "the exported and internal relay plan records must be the same size");
static_assert(
    offsetof(ncclRelayPlanInfo, nCalls) ==
            offsetof(rcclx::relay::RelayPlanInfo, nCalls) &&
        offsetof(ncclRelayPlanInfo, opCode) ==
            offsetof(rcclx::relay::RelayPlanInfo, opCode) &&
        offsetof(ncclRelayPlanInfo, dtype) ==
            offsetof(rcclx::relay::RelayPlanInfo, dtype) &&
        offsetof(ncclRelayPlanInfo, redOp) ==
            offsetof(rcclx::relay::RelayPlanInfo, redOp) &&
        offsetof(ncclRelayPlanInfo, flags) ==
            offsetof(rcclx::relay::RelayPlanInfo, flags) &&
        offsetof(ncclRelayPlanInfo, reserved) ==
            offsetof(rcclx::relay::RelayPlanInfo, reserved),
    "the exported and internal relay plan records must agree field for field");
static_assert(
    std::is_trivially_copyable<ncclRelayPlanInfo>::value &&
        std::is_trivially_copyable<rcclx::relay::RelayPlanInfo>::value,
    "the relay plan records are memcpy'd between exported and internal form");

ncclResult_t ncclRelayControlPublish_impl(
    ncclComm_t comm, uint64_t epoch,
    const ncclRelayPlanInfo* info, const size_t* counts, int64_t timeoutNs) {
  if (comm == nullptr || info == nullptr) {
    WARN("ncclRelayControlPublish: comm and info must be non-null");
    return ncclInvalidArgument;
  }
  // Validated HERE, at the boundary, even though relayControlPublish checks
  // flags again: the public contract declares flags and reserved "must be 0",
  // and a caller that violates it should see the diagnostic attributed to the
  // symbol it called. reserved had no check at all before, so non-zero bytes in
  // a field documented as reserved were accepted and then silently dropped.
  if (info->flags != 0) {
    WARN(
        "ncclRelayControlPublish: info->flags must be 0, got %u", info->flags);
    return ncclInvalidArgument;
  }
  for (size_t i = 0; i < sizeof(info->reserved) / sizeof(info->reserved[0]);
       i++) {
    if (info->reserved[i] != 0) {
      WARN(
          "ncclRelayControlPublish: info->reserved must be 0, got reserved[%zu]=%u",
          i, info->reserved[i]);
      return ncclInvalidArgument;
    }
  }
  if (info->nCalls > 0 && counts == nullptr) {
    WARN(
        "ncclRelayControlPublish: counts must be non-null when nCalls is %u",
        info->nCalls);
    return ncclInvalidArgument;
  }
  rcclx::relay::RelayPlanInfo plan;
  memcpy(&plan, info, sizeof(plan));
  return rcclx::relay::relayControlPublish(
      comm, epoch, plan, counts, timeoutNs);
}

ncclResult_t ncclRelayControlPublish(
    ncclComm_t comm, uint64_t epoch,
    const ncclRelayPlanInfo* info, const size_t* counts, int64_t timeoutNs) {
  return ncclRelayControlPublish_impl(comm, epoch, info, counts, timeoutNs);
}

NCCL_API(ncclResult_t, ncclRelayControlConsume,
    ncclComm_t comm, uint64_t epoch,
    ncclRelayPlanInfo* info, size_t* counts, uint32_t countsCapacity,
    int64_t timeoutNs);

ncclResult_t ncclRelayControlConsume_impl(
    ncclComm_t comm, uint64_t epoch,
    ncclRelayPlanInfo* info, size_t* counts, uint32_t countsCapacity,
    int64_t timeoutNs) {
  if (comm == nullptr || info == nullptr) {
    WARN("ncclRelayControlConsume: comm and info must be non-null");
    return ncclInvalidArgument;
  }
  rcclx::relay::RelayPlanInfo plan{};
  const ncclResult_t res = rcclx::relay::relayControlConsume(
      comm, epoch, &plan, counts, countsCapacity, timeoutNs);
  // Copied out on success, and on the ONE failure that carries information: an
  // over-capacity plan reports the size the caller needed, which is only
  // actionable if the caller can see it.
  //
  // Every other failure -- timeout, a peer abort, no control plane on this comm
  // -- leaves `plan` untouched, so copying it out unconditionally would hand the
  // caller a zeroed record. That is NOT a neutral value: nCalls 0 with opCode 0
  // is exactly ncclRelayOpShutdown, so a failure would be indistinguishable from
  // a valid instruction to stop. The caller's buffer is therefore left alone,
  // which the header documents.
  if (res == ncclSuccess ||
      (res == ncclInvalidArgument && plan.nCalls > countsCapacity)) {
    memcpy(info, &plan, sizeof(*info));
  }
  return res;
}

ncclResult_t ncclRelayControlConsume(
    ncclComm_t comm, uint64_t epoch,
    ncclRelayPlanInfo* info, size_t* counts, uint32_t countsCapacity,
    int64_t timeoutNs) {
  return ncclRelayControlConsume_impl(
      comm, epoch, info, counts, countsCapacity, timeoutNs);
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclBroadcast_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, datatype));

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS, nullptr };

  NCCLCHECK(Recorder::instance().record(rrBroadcast, info));

  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(Recorder::instance().record(rrBcast, buff, buff, count, datatype, comm, stream, root));
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclGather_impl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  NCCLCHECK(Recorder::instance().record(rrGather, sendbuff, recvbuff, count, datatype, comm, stream, root));

  struct ncclInfo info = { ncclFuncGather, "Gather",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    GATHER_CHUNKSTEPS, GATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op, datatype));

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS, nullptr };

  NCCLCHECK(Recorder::instance().record(rrReduce, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);


ncclResult_t ncclReduceScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  // Check for quantized ARG reduce-scatter via environment variable
  if (isLowPrecisionFp8E4M3ReduceScatterEnabled() && (datatype == ncclFloat32 || datatype == ncclBfloat16) &&
      op == ncclSum) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t totalCount = recvcount * nRanks;
    if (totalCount * ncclTypeSize(datatype) >=
        LOW_PRECISION_MSG_SIZE_THRESHOLD) {
      TRACE(
          NCCL_COLL,
          "Using quantized ARG reduce-scatter (FP8 E4M3) for float32 sum reduction");
      return ncclLowPrecisionReduceScatter(
          sendbuff, recvbuff, totalCount, datatype, op, comm, stream);
    }
  }

#ifdef BUILD_META_INTERNAL
  if (comm->algoFactory && op == ncclSum) {
    // try to get meta customized algo
    auto algo = comm->algoFactory->getReduceScatterAlgo(
        sendbuff, recvbuff, recvcount, meta::comms::ncclToMetaComm(datatype), stream);
    if (algo) {
      try {
        algo->reduceScatter();
      } catch (const std::exception& e) {
        WARN("failed to launch custom reduce scatter: %s", e.what());
        return ncclInternalError;
      }
      return ncclSuccess;
    }
  }
#endif

  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op, datatype));
    // RCCL update slice steps for ReduceScatter if single node
    const bool isGfx950 = IsArchMatch(comm->archName, "gfx950");

    int chunkSteps = (isGfx950 && comm->rcclUseOneSlice)? 1 : REDUCESCATTER_CHUNKSTEPS;
    int sliceSteps = comm->rcclUseOneSlice
      ? (isGfx950 ? 1 : REDUCESCATTER_SLICESTEPS_SINGLE_NODE)
      : REDUCESCATTER_SLICESTEPS;

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    chunkSteps, sliceSteps, nullptr };

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t msgSize = recvcount * ncclTypeSize(datatype) * nRanks;

  NCCLCHECK(Recorder::instance().record(rrReduceScatter, info));

  // Reset value forcing direct reduce scatter algorithm 
  comm->enableDirectReduceScatter = 0;
  if (rcclDdaEnabled(comm, nRanks * recvcount * ncclTypeSize(datatype), 8388608) &&
      ncclReduceScatterDdaIpcEligible(comm, sendbuff, recvbuff, recvcount, datatype, op)) {
    NCCLCHECK(ncclReduceScatterDdaIpc(
        sendbuff,
        recvbuff,
        recvcount,
        datatype,
        op,
        comm,
        stream));
    return ncclSuccess;
  }

  if (rcclUseReduceScatterDirect(comm, msgSize)) {
    INFO(NCCL_INIT, "RCCL DIRECT REDUCE-SCATTER recvcount=%zu msgSize=%zu rank=%d nRanks=%d nNodes=%d comm=%p stream=%p sendbuff=%p recvbuff=%p",
      recvcount, msgSize, comm->rank, nRanks, comm->nNodes, comm, stream, sendbuff, recvbuff);

    // Temporary Buffer to store data from each rank
    void* tempbuff = comm->tempBuff;

    // Use Direct Reduce Scatter Algorithm
    comm->enableDirectReduceScatter = 1;
    
    if (recvcount == 0) return ncclSuccess;
    
    // Calculate offset into buffers
    size_t offset = recvcount * ncclTypeSize(datatype);
    
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nRanks; i++) {
      int peer = (comm->rank + i) % nRanks;
      NCCLCHECK(ncclSend((void*)((char*)sendbuff + peer * offset), recvcount, datatype, peer, comm, stream));
      NCCLCHECK(ncclRecv((void*)((char*)tempbuff + peer * offset), recvcount, datatype, peer, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
  }
  
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclScatter_impl(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, datatype));

  NCCLCHECK(Recorder::instance().record(rrScatter, sendbuff, recvbuff, count, datatype, comm, stream, root));

  struct ncclInfo info = { ncclFuncScatter, "Scatter",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);


ncclResult_t ncclSend_impl(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, datatype));

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1, nullptr };

  NCCLCHECK(Recorder::instance().record(rrSend, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclRecv_impl(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, datatype));

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1, nullptr };

  NCCLCHECK(Recorder::instance().record(rrRecv, info));

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclPutSignal, const void* localbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclPutSignal_impl(const void* localbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(PutSignal, NcclNvtxParamsPut,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, ctx));

  struct ncclInfo info = { ncclFuncPutSignal, "PutSignal",
    localbuff, NULL, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1, nullptr, /* chunkSteps, sliceSteps, acc */
    false, /* useDirect */
    peerWinOffset, peerWin, sigIdx, ctx, flags, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
    0, NULL }; /* nDesc, signalDescs */
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSignal, int peer, int sigIdx, int ctx, unsigned int flags,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSignal_impl(int peer, int sigIdx, int ctx, unsigned int flags,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Signal, NcclNvtxParamsSignal,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, peer, ctx));

  struct ncclInfo info = { ncclFuncSignal, "Signal",
    NULL, NULL, 0, ncclInt8, ncclSum, peer, comm, stream, /* Args */
    1, 1, nullptr, /* chunkSteps, sliceSteps, acc */
    false, /* useDirect */
    0, NULL, sigIdx, ctx, flags, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
    0, NULL }; /* nDesc, signalDescs */
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclWaitSignal, int nDesc, ncclWaitSignalDesc_t* signalDescs,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclWaitSignal_impl(int nDesc, ncclWaitSignalDesc_t* signalDescs,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(WaitSignal, NcclNvtxParamsWaitSignal,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, nDesc, 0));

  struct ncclInfo info = { ncclFuncWaitSignal, "WaitSignal",
    NULL, NULL, 0, ncclInt32, ncclSum, 0, comm, stream, /* Args */
    1, 1, nullptr, /* chunkSteps, sliceSteps, acc */
    false, /* useDirect */
    0, NULL, 0, 0, 0, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
    nDesc, signalDescs }; /* nDesc, signalDescs */
  return ncclEnqueueCheck(&info);
}
