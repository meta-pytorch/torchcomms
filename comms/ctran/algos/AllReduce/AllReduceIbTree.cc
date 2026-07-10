// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <climits>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceFused.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/topo/CtranTreeBuilder.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PRIMS)
#include "comms/ctran/algos/AllReduce/AllReduceIbTree.cuh"
#include "comms/prims/transport/MultiPeerTransport.h"
#endif

namespace fused = ctran::allreduce::fused;

namespace {

template <typename NodeToRank>
void populateTreeTopology(
    ctran::allreduce::TreeTopology& dst,
    const ctran::algos::topo::TreeNeighbors& src,
    const NodeToRank& nodeToRank) {
  dst.parentRank = nodeToRank(src.parent);
  dst.numChildren = src.numChildren;
  dst.isRoot = src.isRoot();
  dst.isLeaf = src.isLeaf();
  for (int c = 0; c < src.numChildren; c++) {
    dst.childRanks[c] = nodeToRank(src.children[c]);
  }
}

/**
 * Return whether `comm` is a control communicator created by nccl-tests for
 * inter-suite synchronization. CTREE delegates these to the legacy AllReduce
 * implementation; the helper is local because new fused algorithms must not
 * fall back into legacy code.
 */
bool isNcclTestsSyncComm(CtranComm* comm) {
  const auto& commDesc = comm->statex_->commDesc();
  return commDesc == "nccl-tests-suite-global-sync-comm" ||
      commDesc == "nccl-tests-suite-sync-comm";
}

/** Return the tree fanout for this run; default to NCCL-like binary trees. */
int getAllReduceTreeFanOut() {
  constexpr int kDefaultFanOut = 2;
  const int fanOut = NCCL_CTRAN_TREE_FANOUT;
  if (fanOut >= 1 && fanOut <= ctran::algos::topo::kMaxTreeChildren) {
    return fanOut;
  }

  CLOGF(
      WARN,
      "Ignoring invalid NCCL_CTRAN_TREE_FANOUT={} for ctree fanout; using {}",
      fanOut,
      kDefaultFanOut);
  return kDefaultFanOut;
}

#if defined(ENABLE_PRIMS)
commResult_t validateAllReduceTreeIbPeer(
    const comms::prims::MultiPeerTransport& transport,
    int rank,
    int peerRank,
    const char* treeName,
    const char* edgeName) {
  const auto type = transport.get_transport_type(peerRank);
  if (type == comms::prims::TransportType::P2P_IBGDA ||
      type == comms::prims::TransportType::P2P_IBRC) {
    return commSuccess;
  }

  CLOGF(
      ERR,
      "AllReduce ctree requires an IB transport for {} {} edge "
      "rank {} -> peer {}; got {}",
      treeName,
      edgeName,
      rank,
      peerRank,
      comms::prims::transport_type_name(type));
  return commInvalidArgument;
}

commResult_t validateAllReduceTreeIbTopology(
    const comms::prims::MultiPeerTransport& transport,
    int rank,
    const ctran::allreduce::TreeTopology& tree,
    const char* treeName) {
  if (tree.parentRank >= 0) {
    commResult_t result = validateAllReduceTreeIbPeer(
        transport, rank, tree.parentRank, treeName, "parent");
    if (result != commSuccess) {
      return result;
    }
  }
  for (int c = 0; c < tree.numChildren; c++) {
    commResult_t result = validateAllReduceTreeIbPeer(
        transport, rank, tree.childRanks[c], treeName, "child");
    if (result != commSuccess) {
      return result;
    }
  }
  return commSuccess;
}
#endif

} // namespace

/**
 * Run CTRAN tree AllReduce using NVL reduce-scatter, dual IB trees, and NVL
 * all-gather.
 *
 * The tensor is split across `pMin` owner ranks per node. Each owner segment is
 * further partitioned into logical CUDA blocks so multiple blocks can process a
 * segment concurrently while the two tree lanes operate on disjoint halves of
 * each block-owned tile.
 */
commResult_t ctranAllReduceTree(
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
  (void)timeout;

  // Single rank AllReduce is the identity for every datatype and reduction op.
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

  if (isNcclTestsSyncComm(comm)) {
    CLOGF(
        DBG,
        "AllReduce ctree delegates nccl-tests sync comm {} to ctdirect",
        statex->commDesc());
    return ctranAllReduceDirect(
        sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
  }

  if (redOp != commSum) {
    CLOGF(ERR, "AllReduce ctree currently supports commSum only");
    return commInvalidArgument;
  }

  if (!fused::is_supported_fused_type(datatype)) {
    CLOGF(
        ERR,
        "AllReduce ctree unsupported datatype {}",
        commDataTypeToString(datatype));
    return commInvalidArgument;
  }

  if (nNodes > 1 && !NCCL_CTRAN_IBGDA_SENDRECV_ENABLE) {
    CLOGF(
        ERR,
        "AllReduce ctree requires NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1 for inter-node IB transfers");
    return commInvalidArgument;
  }
  if (!NCCL_CTRAN_USE_PIPES) {
    CLOGF(
        ERR,
        "AllReduce ctree requires NCCL_CTRAN_USE_PIPES=1 for Prims transports");
    return commInvalidArgument;
  }

  const int pMin = fused::compute_p_min(comm);

  const size_t segmentElems = (count + pMin - 1) / pMin;
  const bool participatesInIB = (localRank < pMin);
  const int fanOut = getAllReduceTreeFanOut();

  // Build dual trees (in node-space: 0..nNodes-1)
  ctran::allreduce::TreeTopology tree0{};
  ctran::allreduce::TreeTopology tree1{};

  if (participatesInIB && nNodes > 1) {
    int nodeRank = nodeId;
    auto [t0, t1] =
        ctran::algos::topo::buildDualKaryTree(nNodes, nodeRank, fanOut);

    // Map node-space neighbors to communicator ranks
    auto nodeToRank = [&](int node) -> int {
      if (node < 0) {
        return -1;
      }
      return statex->localRankToRank(localRank, node);
    };

    populateTreeTopology(tree0, t0, nodeToRank);
    populateTreeTopology(tree1, t1, nodeToRank);
  } else if (participatesInIB && nNodes == 1) {
    // Single node: no IB tree needed, Phase 2 is a no-op
    tree0.isRoot = true;
    tree0.isLeaf = true;
    tree1.isRoot = true;
    tree1.isLeaf = true;
  }

  CLOGF(
      DBG,
      "AllReduce ctree: rank {} localRank {} nNodes {} pMin {} "
      "segmentElems {} fanOut {} participatesInIB {} "
      "tree0[parent={} children={} root={} leaf={}] "
      "tree1[parent={} children={} root={} leaf={}]",
      rank,
      localRank,
      nNodes,
      pMin,
      segmentElems,
      fanOut,
      participatesInIB,
      tree0.parentRank,
      tree0.numChildren,
      tree0.isRoot,
      tree0.isLeaf,
      tree1.parentRank,
      tree1.numChildren,
      tree1.isRoot,
      tree1.isLeaf);

#if defined(ENABLE_PRIMS)
  const int nLocalRanks = statex->nLocalRanks();
  const size_t elementSize = commTypeSize(datatype);
  const size_t totalBytes = count * elementSize;
  const size_t segmentBytes = segmentElems * elementSize;
  const bool hasIbPhase = participatesInIB && nNodes > 1;
  const int numBlocks =
      fused::compute_num_blocks(totalBytes, fused::get_num_block_cap());
  if (hasIbPhase) {
    if (!comm->multiPeerTransport_) {
      CLOGF(
          ERR,
          "AllReduce ctree requires MultiPeerTransport for inter-node IB transfers");
      return commInvalidArgument;
    }
    commResult_t result = validateAllReduceTreeIbTopology(
        *comm->multiPeerTransport_, rank, tree0, "tree0");
    if (result != commSuccess) {
      return result;
    }
    result = validateAllReduceTreeIbTopology(
        *comm->multiPeerTransport_, rank, tree1, "tree1");
    if (result != commSuccess) {
      return result;
    }

    const int requiredIbGroups = numBlocks * ctran::allreduce::tree::kTreeLanes;
    if (requiredIbGroups > NCCL_CTRAN_IB_MAX_GROUPS) {
      CLOGF(
          ERR,
          "AllReduce ctree requires {} IBGDA send/recv groups, exceeding "
          "NCCL_CTRAN_IB_MAX_GROUPS={}",
          requiredIbGroups,
          NCCL_CTRAN_IB_MAX_GROUPS);
      return commInvalidArgument;
    }
  }

  void* phase2Buf = fused::compute_phase2_buf(
      recvbuff, localRank, segmentBytes, participatesInIB);

  CLOGF(
      DBG,
      "AllReduce ctree launch: totalBytes {} segmentBytes {} numBlocks {} "
      "threadsPerBlock {} hasIbPhase {}",
      totalBytes,
      segmentBytes,
      numBlocks,
      ctran::allreduce::tree::kBlockSize,
      hasIbPhase);

  auto opCount = comm->ctran_->getOpCount();

  ctran::allreduce::tree::KernArgs kernArgs{};
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
  kernArgs.tree0 = tree0;
  kernArgs.tree1 = tree1;
  kernArgs.ibSendRecvGroups = NCCL_CTRAN_IB_MAX_GROUPS;

  return fused::submit_fused_kernel(
      comm,
      stream,
      "AllReduceTree",
      opCount,
      numBlocks,
      ctran::allreduce::tree::kBlockSize,
      &kernArgs,
      reinterpret_cast<const void*>(ctranKernelAllReduceTree));
#else
  CLOGF(ERR, "AllReduce ctree requires ENABLE_PRIMS");
  return commInternalError;
#endif
}
