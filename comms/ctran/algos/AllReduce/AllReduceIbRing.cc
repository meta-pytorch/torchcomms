// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <chrono>
#include <optional>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceFused.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

// AllReduceIbRing.cuh exposes the host-safe `hierring::RingTopology` (used
// below regardless of ENABLE_PRIMS) outside its ENABLE_PRIMS guard; the device
// `KernArgs` / kernel entry inside the guard are used only on the PIPES path.
#include "comms/ctran/algos/AllReduce/AllReduceIbRing.cuh"

#if defined(ENABLE_PRIMS)
#include "comms/prims/transport/MultiPeerTransport.h"
#endif

namespace fused = ctran::allreduce::fused;

#if defined(ENABLE_PRIMS)
namespace {
// Validates that an IB ring neighbor (`peerRank`) resolves to a P2P_IBGDA
// transport. The kernel dereferences `transports[peerRank].p2p_ib.ibgda`
// (AllReduceIbRing.cu), so a non-IBGDA peer — e.g. a comm using the IBRC
// backend, where the same union holds `.ibrc` — would reinterpret the wrong
// union member and trap/hang at kernel time. The coarse
// NCCL_CTRAN_IBGDA_SENDRECV_ENABLE knob does not catch this. Mirrors
// AllReduceIbTree.cc's validateAllReduceTreeIbPeer (file-local, not reusable).
commResult_t validateHierRingIbPeer(
    const comms::prims::MultiPeerTransport& transport,
    int rank,
    int peerRank,
    const char* edgeName) {
  const auto type = transport.get_transport_type(peerRank);
  if (type == comms::prims::TransportType::P2P_IBGDA) {
    return commSuccess;
  }
  CLOGF(
      ERR,
      "AllReduce cthierarchical_ring requires P2P_IBGDA transport for {} edge "
      "rank {} -> peer {}; got {}",
      edgeName,
      rank,
      peerRank,
      comms::prims::transport_type_name(type));
  return commInvalidArgument;
}
} // namespace
#endif

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
    // Degenerate case: device-to-device memcpy short-circuit
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

  if (redOp != commSum) {
    CLOGF(ERR, "AllReduce cthierarchical_ring currently supports commSum only");
    return commInvalidArgument;
  }

  if (datatype != commFloat32 && datatype != commFloat16) {
    CLOGF(
        ERR,
        "AllReduce cthierarchical_ring currently supports commFloat32/commFloat16 only, got {}",
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

  const size_t rawSegmentElems = (count + pMin - 1) / pMin; // ceil(count/pMin)
  const size_t elemSz = commTypeSize(datatype);
  const size_t alignElems = 16 / elemSz;
  const size_t segmentElems =
      (rawSegmentElems + alignElems - 1) & ~(alignElems - 1); // round up to 16B
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

#if defined(ENABLE_PRIMS)
  const int nLocalRanks = statex->nLocalRanks();
  const size_t elementSize = commTypeSize(datatype);
  const size_t totalBytes = count * elementSize;
  const size_t segmentBytes = segmentElems * elementSize;
  const bool hasIbPhase = participatesInIB && nNodes > 1;

  // Size-tiered count that drives block tiling (i.e., bucket the
  // owner-segment size into the 4/8/16 tiers, then cap it and reduce).
  // Larger segments get more blocks (more tile parallelism).
  // Small segments get fewer (avoids too much per-block/WQE overhead).
  const int numBlocks =
      fused::compute_num_blocks_ring(segmentBytes, fused::get_num_block_cap());

  if (hasIbPhase) {
    // Validate the IB transport before launching: the kernel blindly
    // dereferences `transports[prev/next].p2p_ib.ibgda`, so a missing
    // MultiPeerTransport, lazy (unmaterialized) connect, or a non-IBGDA peer
    // would trap/hang on device. Fail fast on the host instead.
    if (!comm->multiPeerTransport_) {
      CLOGF(
          ERR,
          "AllReduce cthierarchical_ring requires MultiPeerTransport for inter-node IB transfers");
      return commInvalidArgument;
    }
    if (comm->multiPeerTransport_->is_lazy_mode()) {
      CLOGF(
          ERR,
          "AllReduce cthierarchical_ring requires eager IB connect (NCCL_CTRAN_IBGDA_LAZY_CONNECT=0); lazy mode is not supported");
      return commInvalidArgument;
    }
    commResult_t prevValid = validateHierRingIbPeer(
        *comm->multiPeerTransport_, rank, ring.prevRank, "prev");
    if (prevValid != commSuccess) {
      return prevValid;
    }
    commResult_t nextValid = validateHierRingIbPeer(
        *comm->multiPeerTransport_, rank, ring.nextRank, "next");
    if (nextValid != commSuccess) {
      return nextValid;
    }

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
  CLOGF(ERR, "AllReduce cthierarchical_ring requires ENABLE_PRIMS");
  return commInternalError;
#endif
}
