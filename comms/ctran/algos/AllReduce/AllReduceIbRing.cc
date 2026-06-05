// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <chrono>
#include <optional>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/AllReduce/AllReduceV2.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PIPES)
#include "comms/ctran/algos/AllReduce/AllReduceIbRing.cuh"
#endif

namespace fused = ctran::allreduce::fused;

commResult_t ctranAllReduceHierarchicalRing(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout) {
  if (count == 0) {
    return commSuccess;
  }

  const auto* statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  const int nNodes = statex->nNodes();
  const int localRank = statex->localRank();
  const int nodeId = statex->node();

  if (nRanks == 1) {
    if (sendbuff != recvbuff) {
      FB_CUDACHECK(cudaMemcpyAsync(
          recvbuff,
          sendbuff,
          count * commTypeSize(datatype),
          cudaMemcpyDeviceToDevice,
          stream));
    }
    return commSuccess;
  }

  if (fused::is_nccl_tests_sync_comm(comm)) {
    CLOGF(
        DBG,
        "AllReduce cthierarchical_ring delegates nccl-tests sync comm {} to ctdirect",
        statex->commDesc());
    return ctranAllReduceDirect(
        sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
  }

  if (redOp != commSum) {
    CLOGF(ERR, "AllReduce cthierarchical_ring currently supports commSum only");
    return commInvalidArgument;
  }

  if (datatype != commFloat32) {
    CLOGF(
        ERR,
        "AllReduce cthierarchical_ring currently supports commFloat32 only, got {}",
        commDataTypeToString(datatype));
    return commInvalidArgument;
  }

  if (nNodes > 1 && !NCCL_CTRAN_IBGDA_SENDRECV_ENABLE) {
    CLOGF(
        ERR,
        "AllReduce cthierarchical_ring requires NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1 for inter-node IB transfers");
    return commInvalidArgument;
  }
  if (!NCCL_CTRAN_USE_PIPES) {
    CLOGF(
        ERR,
        "AllReduce cthierarchical_ring requires NCCL_CTRAN_USE_PIPES=1 for Pipes transports");
    return commInvalidArgument;
  }

  const int pMin = fused::compute_p_min(comm);

  const size_t segmentElems = (count + pMin - 1) / pMin;
  const bool participatesInIB = (localRank < pMin);

  ctran::allreduce::hierring::RingTopology ring{};
  if (participatesInIB && nNodes > 1) {
    int prevNode = (nodeId - 1 + nNodes) % nNodes;
    int nextNode = (nodeId + 1) % nNodes;
    ring.prevRank = statex->localRankToRank(localRank, prevNode);
    ring.nextRank = statex->localRankToRank(localRank, nextNode);
    ring.nNodes = nNodes;
    ring.myNodeIdx = nodeId;
  }

  CLOGF(
      DBG,
      "AllReduce cthierarchical_ring: rank {} localRank {} nNodes {} pMin {} "
      "segmentElems {} participatesInIB {} ring[prev={} next={} myNode={}]",
      rank,
      localRank,
      nNodes,
      pMin,
      segmentElems,
      participatesInIB,
      ring.prevRank,
      ring.nextRank,
      ring.myNodeIdx);

#if defined(ENABLE_PIPES)
  const int nLocalRanks = statex->nLocalRanks();
  const size_t elementSize = commTypeSize(datatype);
  const size_t totalBytes = count * elementSize;
  const size_t segmentBytes = segmentElems * elementSize;
  const bool hasIbPhase = participatesInIB && nNodes > 1;
  const int numBlocks =
      fused::compute_num_blocks(totalBytes, fused::get_num_block_cap());

  if (hasIbPhase) {
    const int maxIbGroups = NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS;
    const int requiredIbGroups =
        numBlocks * ctran::allreduce::hierring::kRingLanes;
    if (requiredIbGroups > maxIbGroups) {
      CLOGF(
          ERR,
          "AllReduce cthierarchical_ring requires {} IBGDA send/recv groups, "
          "exceeding NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS={}",
          requiredIbGroups,
          maxIbGroups);
      return commInvalidArgument;
    }
  }

  void* phase2Buf = fused::compute_phase2_buf(
      recvbuff, localRank, segmentBytes, participatesInIB);

  CLOGF(
      DBG,
      "AllReduce cthierarchical_ring launch: totalBytes {} segmentBytes {} "
      "numBlocks {} threadsPerBlock {} hasIbPhase {}",
      totalBytes,
      segmentBytes,
      numBlocks,
      ctran::allreduce::hierring::kBlockSize,
      hasIbPhase);

  auto opCount = comm->ctran_->getOpCount();

  ctran::allreduce::hierring::KernArgs kernArgs{};
  FB_COMMCHECK(
      fused::fill_common_kern_args(
          kernArgs.common,
          sendbuff,
          recvbuff,
          phase2Buf,
          count,
          segmentElems,
          nNodes,
          pMin,
          nLocalRanks,
          localRank,
          numBlocks,
          datatype,
          redOp,
          comm));
  kernArgs.ring = ring;

  return fused::submit_fused_kernel(
      comm,
      stream,
      "AllReduceHierarchicalRing",
      opCount,
      numBlocks,
      ctran::allreduce::hierring::kBlockSize,
      &kernArgs,
      reinterpret_cast<const void*>(ctranKernelAllReduceHierarchicalRing));
#else
  CLOGF(ERR, "AllReduce cthierarchical_ring requires ENABLE_PIPES");
  return commInternalError;
#endif
}
