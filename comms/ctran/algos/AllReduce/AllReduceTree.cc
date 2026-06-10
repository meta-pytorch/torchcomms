// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <climits>
#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/AllReduce/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/topo/CtranTreeBuilder.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PIPES)
#include "comms/ctran/algos/AllReduce/AllReduceTree.cuh"
#include "comms/prims/transport/MultiPeerTransport.h"
#endif

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

/** Return whether the ctree device kernel supports `datatype`. */
bool isSupportedTreeType(commDataType_t datatype) {
  return datatype == commFloat32 || datatype == commFloat16;
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

#if defined(ENABLE_PIPES)
/** Return the topology-wide cap for CTREE logical CUDA blocks. */
int getAllReduceTreeNumBlockCap() {
  return std::max(1, NCCL_CTRAN_MAX_NBLOCKS);
}

/**
 * Select the number of independent data tiles for this launch.
 *
 * The algorithm is correct with one tile block. Larger values only expose more
 * tile parallelism. `NCCL_CTRAN_MAX_NBLOCKS` caps the default policy. The
 * automatic policy starts at the cap, then reduces the number of blocks while
 * the message has less than one block-threshold of work per block.
 */
int getAllReduceTreeNumBlocks(size_t totalBytes) {
  const int cap = getAllReduceTreeNumBlockCap();
  const size_t perBlockThresholdBytes =
      static_cast<size_t>(ctran::allreduce::tree::kBlockSize) * 64;

  int numBlocks = cap;
  while (numBlocks > 1 &&
         totalBytes < static_cast<size_t>(numBlocks) * perBlockThresholdBytes) {
    numBlocks--;
  }
  return numBlocks;
}

commResult_t validateAllReduceTreeIbPeer(
    const comms::prims::MultiPeerTransport& transport,
    int rank,
    int peerRank,
    const char* treeName,
    const char* edgeName) {
  const auto type = transport.get_transport_type(peerRank);
  if (type == comms::prims::TransportType::P2P_IBGDA) {
    return commSuccess;
  }

  CLOGF(
      ERR,
      "AllReduce ctree requires P2P_IBGDA transport for {} {} edge "
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

  if (redOp != commSum) {
    CLOGF(ERR, "AllReduce ctree currently supports commSum only");
    return commInvalidArgument;
  }

  if (!isSupportedTreeType(datatype)) {
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

  // Compute P_min: minimum nLocalRanks across all nodes
  int pMin = INT_MAX;
  for (int n = 0; n < nNodes; n++) {
    int someRankOnNode = statex->localRankToRank(0, n);
    pMin = std::min(pMin, statex->nLocalRanks(someRankOnNode));
  }

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

#if defined(ENABLE_PIPES)
  const int nLocalRanks = statex->nLocalRanks();
  const size_t elementSize = commTypeSize(datatype);
  const size_t totalBytes = count * elementSize;
  const size_t segmentBytes = segmentElems * elementSize;
  const bool hasIbPhase = participatesInIB && nNodes > 1;
  const int numBlocks = getAllReduceTreeNumBlocks(totalBytes);
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

    const int maxIbGroups = NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS;
    const int requiredIbGroups = numBlocks * ctran::allreduce::tree::kTreeLanes;
    if (requiredIbGroups > maxIbGroups) {
      CLOGF(
          ERR,
          "AllReduce ctree requires {} IBGDA send/recv groups, exceeding "
          "NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS={}",
          requiredIbGroups,
          maxIbGroups);
      return commInvalidArgument;
    }
  }

  // phase2Buf holds the locally-reduced segment from Phase 1 and serves as
  // the working buffer for Phase 2's reduce-up + broadcast-down. Non-owner
  // local ranks do not dereference phase2Buf, but keep it inside the user
  // buffer so the kernel never receives an out-of-range pointer in
  // heterogeneous topologies.
  void* phase2Buf = recvbuff;
  if (participatesInIB) {
    phase2Buf = static_cast<char*>(recvbuff) +
        static_cast<size_t>(localRank) * segmentBytes;
  }

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

  // Populate kernel args
  ctran::allreduce::tree::KernArgs kernArgs{};
  kernArgs.sendbuff = sendbuff;
  kernArgs.recvbuff = recvbuff;
  kernArgs.phase2Buf = phase2Buf;
  kernArgs.count = count;
  kernArgs.segmentElems = segmentElems;
  kernArgs.nNodes = nNodes;
  kernArgs.pMin = pMin;
  kernArgs.nLocalRanks = nLocalRanks;
  kernArgs.localRank = localRank;
  kernArgs.numBlocks = numBlocks;
  kernArgs.datatype = datatype;
  kernArgs.redOp = redOp;
  kernArgs.transports = comm->getMultiPeerTransportsPtr();
  if (kernArgs.transports == nullptr) {
    CLOGF(
        ERR,
        "AllReduce ctree: getMultiPeerTransportsPtr() returned null - "
        "Prims transport not initialized. Ensure ENABLE_PIPES is defined "
        "and multiPeerTransport is set up.");
    return commInternalError;
  }
  kernArgs.tree0 = tree0;
  kernArgs.tree1 = tree1;

  // Build local rank -> global rank mapping
  if (nLocalRanks > CTRAN_MAX_NVL_PEERS) {
    CLOGF(
        ERR,
        "AllReduce ctree: nLocalRanks {} exceeds CTRAN_MAX_NVL_PEERS {}",
        nLocalRanks,
        CTRAN_MAX_NVL_PEERS);
    return commInternalError;
  }
  for (int lr = 0; lr < nLocalRanks; lr++) {
    kernArgs.localRankToGlobalRank[lr] = statex->localRankToRank(lr);
  }

  KernelConfig config(
      KernelConfig::KernelType::ALLREDUCE, stream, "AllReduceTree", opCount);

  config.numBlocks = static_cast<unsigned int>(numBlocks);
  config.numThreads = ctran::allreduce::tree::kBlockSize;
  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.algoArgs = &kernArgs;

  // Device kernel drives all NVL and IB transport progress; no host-side GPE
  // opGroup is needed.
  std::vector<std::unique_ptr<struct OpElem>> opGroup;

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      nullptr,
      config,
      reinterpret_cast<void*>(ctranKernelAllReduceTree)));

  return commSuccess;
#else
  CLOGF(ERR, "AllReduce ctree requires ENABLE_PIPES");
  return commInternalError;
#endif
}
