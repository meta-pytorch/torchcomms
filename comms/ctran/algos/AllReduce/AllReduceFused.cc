// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/AllReduce/AllReduceFused.h"

#include <algorithm>
#include <climits>
#include <exception>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PRIMS)
#include "comms/prims/transport/MultiPeerTransport.h"
#endif

namespace ctran::allreduce::fused {

bool is_supported_fused_type(commDataType_t datatype) {
  return datatype == commFloat32 || datatype == commFloat16;
}

int compute_p_min(CtranComm* comm) {
  const auto* statex = comm->statex_.get();
  const int nNodes = statex->nNodes();
  int pMin = INT_MAX;
  for (int n = 0; n < nNodes; n++) {
    int someRankOnNode = statex->localRankToRank(0, n);
    pMin = std::min(pMin, statex->nLocalRanks(someRankOnNode));
  }
  return pMin;
}

int get_num_block_cap() {
  return std::max(1, NCCL_CTRAN_MAX_NBLOCKS);
}

int compute_num_blocks(size_t totalBytes, int cap) {
  const size_t perBlockThresholdBytes = common::kBlockSize * 64;

  int numBlocks = cap;
  while (numBlocks > 1 &&
         totalBytes < static_cast<size_t>(numBlocks) * perBlockThresholdBytes) {
    numBlocks--;
  }
  return numBlocks;
}

int compute_num_blocks_ring(size_t segmentBytes, int cap) {
  // Size-aware launch-geometry cap on the Phase-2 owner-segment size. The
  // hierarchical ring does 2(W-1) serialized IB steps per pipeline window, so
  // per-WQE/group overhead dominates at small/medium sizes and fewer blocks win
  // there; large segments use the full cap. This is a launch-geometry ceiling
  // on TOP of compute_num_blocks -- NOT a re-implementation of per-block
  // sizing: the per-block byte threshold and the reduce-to-fit loop live in
  // compute_num_blocks (single source of truth); this only lowers the ceiling
  // before delegating. Keyed on segmentBytes (not totalBytes) because Phase 2
  // operates per owner segment (segmentBytes ~= totalBytes/pMin in HYBRID). The
  // 4/8/16 tiers are a starting point at this stack position; they are
  // validated by the block-count sweep in the descendant diff D108428650.
  int tier;
  if (segmentBytes <= (1ull << 20)) { // <= 1 MB
    tier = 4;
  } else if (segmentBytes <= (32ull << 20)) { // <= 32 MB
    tier = 8;
  } else {
    tier = 16;
  }
  return compute_num_blocks(segmentBytes, std::min(cap, tier));
}

void* compute_phase2_buf(
    void* recvbuff,
    int localRank,
    size_t segmentBytes,
    bool participatesInIB) {
  if (participatesInIB) {
    return static_cast<char*>(recvbuff) +
        static_cast<size_t>(localRank) * segmentBytes;
  }
  return recvbuff;
}

#if defined(ENABLE_PRIMS)

commResult_t fill_common_kern_args(
    common::CommonKernArgs& args,
    const void* sendbuff,
    void* recvbuff,
    void* phase2Buf,
    size_t count,
    size_t segmentElems,
    int nNodes,
    int pMin,
    int nLocalRanks,
    int localRank,
    int numBlocks,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    std::optional<std::vector<int>> ibPeers) {
  args.sendbuff = sendbuff;
  args.recvbuff = recvbuff;
  args.phase2Buf = phase2Buf;
  args.count = count;
  args.segmentElems = segmentElems;
  args.nNodes = nNodes;
  args.pMin = pMin;
  args.nLocalRanks = nLocalRanks;
  args.localRank = localRank;
  args.numBlocks = numBlocks;
  args.datatype = datatype;
  args.redOp = redOp;
  // Transport array for the kernel. Eager mode: the no-peer getter is valid.
  // Lazy mode: the no-peer getter throws, so the caller MUST hand us an
  // explicit peer list to materialize -- an empty list is fine (ranks that use
  // no IB slots still get a valid pointer), but a missing (nullopt) list is a
  // caller that has not audited its lazy peers, so fail rather than hand the
  // kernel unmaterialized slots.
  const bool lazy =
      comm->multiPeerTransport_ && comm->multiPeerTransport_->is_lazy_mode();
  if (!lazy) {
    args.transports = comm->getMultiPeerTransportsPtr();
  } else if (!ibPeers.has_value()) {
    CLOGF(
        ERR,
        "AllReduce fused: lazy IB connect requires an explicit peer list, "
        "but the caller provided none");
    return commInvalidArgument;
  } else {
    try {
      args.transports = comm->getMultiPeerTransportsPtr(*ibPeers);
    } catch (const std::exception& e) {
      CLOGF(
          ERR,
          "AllReduce fused: lazy peer materialization failed: {}",
          e.what());
      return commInternalError;
    } catch (...) {
      CLOGF(
          ERR,
          "AllReduce fused: lazy peer materialization failed (non-standard exception)");
      return commInternalError;
    }
  }
  if (args.transports == nullptr) {
    CLOGF(
        ERR,
        "AllReduce fused: getMultiPeerTransportsPtr() returned null — "
        "Prims transport not initialized. Ensure ENABLE_PRIMS is defined "
        "and multiPeerTransport is set up.");
    return commInternalError;
  }

  if (nLocalRanks > CTRAN_MAX_NVL_PEERS) {
    CLOGF(
        ERR,
        "AllReduce fused: nLocalRanks {} exceeds CTRAN_MAX_NVL_PEERS {}",
        nLocalRanks,
        CTRAN_MAX_NVL_PEERS);
    return commInternalError;
  }
  const auto* statex = comm->statex_.get();
  for (int lr = 0; lr < nLocalRanks; lr++) {
    args.localRankToGlobalRank[lr] = statex->localRankToRank(lr);
  }

  return commSuccess;
}

commResult_t submit_fused_kernel(
    CtranComm* comm,
    cudaStream_t stream,
    const char* kernelName,
    uint64_t opCount,
    int numBlocks,
    int numThreads,
    void* algoArgs,
    const void* kernelFnPtr) {
  (void)kernelName;
  (void)opCount;

  int* flag = nullptr;
  void* devState = nullptr;
  void* kernelArgs[] = {&flag, &devState, algoArgs};

  FB_CUDACHECK(cudaLaunchKernel(
      kernelFnPtr,
      dim3(static_cast<unsigned int>(numBlocks), 1, 1),
      dim3(static_cast<unsigned int>(numThreads), 1, 1),
      kernelArgs,
      0,
      stream));
  comm->ctran_->updateOpCount();

  return commSuccess;
}

#endif // ENABLE_PRIMS

} // namespace ctran::allreduce::fused
