// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <climits>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/AllReduce/Types.h"
#include "comms/ctran/algos/topo/CtranTreeBuilder.h"
#include "comms/utils/logger/LogUtils.h"

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

} // namespace

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

  // Single rank: just copy
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

  // Compute P_min: minimum nLocalRanks across all nodes
  int pMin = INT_MAX;
  for (int n = 0; n < nNodes; n++) {
    int someRankOnNode = statex->localRankToRank(0, n);
    pMin = std::min(pMin, statex->nLocalRanks(someRankOnNode));
  }

  const size_t segmentElems = (count + pMin - 1) / pMin;
  const bool participatesInIB = (localRank < pMin);
  const int fanOut = 2; // binary trees for V1

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
      "segmentElems {} participatesInIB {} "
      "tree0[parent={} children={} root={} leaf={}] "
      "tree1[parent={} children={} root={} leaf={}]",
      rank,
      localRank,
      nNodes,
      pMin,
      segmentElems,
      participatesInIB,
      tree0.parentRank,
      tree0.numChildren,
      tree0.isRoot,
      tree0.isLeaf,
      tree1.parentRank,
      tree1.numChildren,
      tree1.isRoot,
      tree1.isLeaf);

  // TODO: Populate kernel args and launch kernel (Diffs 4+5)
  CLOGF(ERR, "AllReduce ctree kernel not yet implemented");
  return commResult_t::commInternalError;
}
