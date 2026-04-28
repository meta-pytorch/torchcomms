// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Cudagraph-aware AllGather: when ctranAllGather() is called during CUDA graph
// capture with a ctgraph* algorithm, this module handles buffer registration
// and algorithm dispatch. The algo is either explicitly specified
// (ctgraph_pipeline, ctgraph_ring, ctgraph_rd) or auto-selected by
// selectCtgraphAlgo() based on topology and message size.
//
// Two registration strategies:
//
//   winPersistBuffReg (ctgraph_pipeline, nLocalRanks > 1):
//     Both local and remote addresses must be stable across replays.
//     Uses window: ctranWinRegister → allGatherWinInit → allGatherWinExec.
//
//   localPersistBuffReg (ctgraph_ring/rd, nLocalRanks == 1):
//     Only local registration persists; remote exchange happens at each replay
//     via IB isendCtrl/irecvCtrl inside the GPE host node.
//     Uses globalRegisterWithPtr so searchRegHandle hits the fast path.
//
// Cleanup: Resources are registered for deferred cleanup at capture time
// (not at graph destruction). This ensures cleanup runs during comm
// destruction on the main thread, regardless of when or whether the graph
// is destroyed. Graph replay is guaranteed to finish before comm destroy.

#include <folly/ScopeGuard.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

// winPersistBuffReg: register recvbuff as a window, exchange handles with
// all peers. Both local and remote addresses must be stable across replays.
commResult_t winPersistBuffReg(
    void* recvbuff,
    size_t recvBytes,
    CtranComm* comm,
    cudaStream_t stream,
    ctran::CtranWin** winOut,
    CtranPersistentRequest** requestOut) {
  FB_COMMCHECK(ctran::ctranWinRegister(recvbuff, recvBytes, comm, winOut));

  auto winGuard = folly::makeGuard([winOut]() { delete *winOut; });

  CtranPersistentRequest* request = nullptr;
  {
    meta::comms::StreamCaptureModeGuard captureGuard{
        cudaStreamCaptureModeRelaxed};
    FB_COMMCHECK(ctran::allGatherWinInit(*winOut, comm, stream, request));
  }

  winGuard.dismiss();
  *requestOut = request;
  return commSuccess;
}

// localPersistBuffReg: register recvbuff locally via globalRegisterWithPtr.
// Only local registration persists; remote exchange happens at each replay.
commResult_t localPersistBuffReg(void* recvbuff, size_t recvBytes) {
  meta::comms::StreamCaptureModeGuard captureGuard{
      cudaStreamCaptureModeRelaxed};
  FB_COMMCHECK(ctran::globalRegisterWithPtr(recvbuff, recvBytes, true, true));
  return commSuccess;
}

enum NCCL_ALLGATHER_ALGO selectCtgraphAlgo(
    size_t sendBytes,
    const ncclx::CommStateX* statex) {
  if (statex->nLocalRanks() > 1) {
    return NCCL_ALLGATHER_ALGO::ctgraph_pipeline;
  }
  return (sendBytes >= NCCL_CTGRAPH_ALLGATHER_RING_THRESHOLD)
      ? NCCL_ALLGATHER_ALGO::ctgraph_ring
      : NCCL_ALLGATHER_ALGO::ctgraph_rd;
}

} // namespace

commResult_t ctranAllGatherCudagraphAware(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLGATHER_ALGO algo) {
  const auto statex = comm->statex_.get();
  const int nRanks = statex->nRanks();
  const size_t recvBytes = sendcount * commTypeSize(datatype) * nRanks;

  // auto-select algo if not specified
  if (algo == NCCL_ALLGATHER_ALGO::ctgraph) {
    algo = selectCtgraphAlgo(sendcount * commTypeSize(datatype), statex);
  }

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllGather cudagraph-aware: algo {} "
      "sendcount {} recvBytes {} commHash {:x} commDesc {} nRanks {} nLocalRanks {} nNodes {}",
      allGatherAlgoName(algo),
      sendcount,
      recvBytes,
      statex->commHash(),
      statex->commDesc(),
      nRanks,
      statex->nLocalRanks(),
      statex->nNodes());

  // Buffer registration + cleanup guard
  ctran::CtranWin* win = nullptr;
  CtranPersistentRequest* request = nullptr;

  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
      FB_COMMCHECK(
          winPersistBuffReg(recvbuff, recvBytes, comm, stream, &win, &request));
      break;
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
    case NCCL_ALLGATHER_ALGO::ctgraph_rd:
      FB_COMMCHECK(localPersistBuffReg(recvbuff, recvBytes));
      break;
    default:
      FB_ERRORRETURN(
          commInvalidArgument,
          "Unexpected algo {} in ctranAllGatherCudagraphAware",
          allGatherAlgoName(algo));
  }

  // Cleanup lambda: shared between error guard and deferred cleanup.
  std::function<void()> cleanup;
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
      cleanup = [request, win]() {
        if (request) {
          ctran::allGatherWinDestroy(request);
          delete request;
        }
        if (win) {
          win->free(true /* skipBarrier */);
          delete win;
        }
      };
      break;
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
    case NCCL_ALLGATHER_ALGO::ctgraph_rd:
      cleanup = [recvbuff, recvBytes]() {
        ctran::globalDeregisterWithPtr(recvbuff, recvBytes);
      };
      break;
    default:
      break;
  }

  auto cleanupGuard = folly::makeGuard([&cleanup]() {
    if (cleanup) {
      cleanup();
    }
  });

  // Execute (captured into graph)
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline: {
      auto savedPAlgo = NCCL_ALLGATHER_P_ALGO;
      NCCL_ALLGATHER_P_ALGO = NCCL_ALLGATHER_P_ALGO::ctpipeline;
      auto pAlgoGuard = folly::makeGuard(
          [savedPAlgo]() { NCCL_ALLGATHER_P_ALGO = savedPAlgo; });
      FB_COMMCHECK(
          ctran::allGatherWinExec(sendbuff, sendcount, datatype, request));
      break;
    }
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
      FB_COMMCHECK(ctranAllGatherRing(
          sendbuff, recvbuff, sendcount, datatype, comm, stream));
      break;
    case NCCL_ALLGATHER_ALGO::ctgraph_rd:
      FB_COMMCHECK(ctranAllGatherRd(
          sendbuff, recvbuff, sendcount, datatype, comm, stream));
      break;
    default:
      break;
  }

  // Success — transfer ownership to deferred cleanup
  cleanupGuard.dismiss();
  comm->cudagraphDeferredCleanup.add(std::move(cleanup));

  return commSuccess;
}
