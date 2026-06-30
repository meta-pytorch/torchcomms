// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/AllReduce/AllReduceFused.h"

#include <algorithm>
#include <climits>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

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
    CtranComm* comm) {
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
  const size_t elemBytes = commTypeSize(datatype);
  args.blockTileBytes = common::compute_aligned_tile_parition_size(
                            segmentElems, elemBytes, numBlocks) *
      elemBytes;
  args.datatype = datatype;
  args.redOp = redOp;
  args.transports = comm->getMultiPeerTransportsPtr();
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
