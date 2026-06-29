// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Cudagraph-aware AllGather: when ctranAllGather() is called during CUDA graph
// capture with a ctgraph* algorithm, this module handles buffer registration
// and algorithm dispatch. The algo is either explicitly specified
// (ctgraph_pipeline, ctgraph_rdpipeline, ctgraph_ring, ctgraph_rd) or
// auto-selected by selectCtgraphAlgo() based on topology and message size.
//
// Two registration strategies:
//
//   winPersistBuffReg (ctgraph_pipeline/rdpipeline, nLocalRanks > 1):
//     Local NVL peer addresses must be stable across replays because the
//     captured CE broadcasts use them. Rail peers exchange IB addr/rkey once
//     during allGatherWinInit.
//     Uses local-NVL window registration → allGatherWinInit → AlgoImpl
//     dispatch.
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
#include <memory>
#include <utility>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranCommSplit.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/utils/MathUtils.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

// winPersistBuffReg: register recvbuff as a window on the caller-provided
// local-NVL clique comm (nvlComm), persisting only the NVL peers' addresses.
// The cross-node rail IB rkeys are NOT exchanged here: allGatherWinInit leaves
// them UNSET (deferRail) and they are filled lazily on the first graph replay
// by ensureRailKeysExchanged() in the exec path. At replay all ranks are in
// lockstep, so that in-line allGatherCtrl is quick and never gated by a
// straggler's capture-time cuMemImport. Registering on the local-NVL comm
// preserves normal whole-communicator window registration semantics within
// that domain.
commResult_t winPersistBuffReg(
    void* recvbuff,
    size_t recvBytes,
    CtranComm* comm,
    CtranComm* nvlComm,
    cudaStream_t stream,
    ctran::CtranWin** nvlWinOut,
    CtranPersistentRequest** requestOut) {
  FB_COMMCHECK(
      ctran::ctranWinRegister(recvbuff, recvBytes, nvlComm, nvlWinOut));

  auto winGuard = folly::makeGuard([nvlWinOut]() {
    if (*nvlWinOut != nullptr) {
      (*nvlWinOut)->free(true /* skipBarrier */);
      delete *nvlWinOut;
      *nvlWinOut = nullptr;
    }
  });

  CtranPersistentRequest* request = nullptr;
  {
    meta::comms::StreamCaptureModeGuard captureGuard{
        cudaStreamCaptureModeRelaxed};
    FB_COMMCHECK(
        ctran::allGatherWinInit(
            *nvlWinOut, comm, stream, request, /*deferRail=*/true));
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
  const bool largeMessage = sendBytes >= NCCL_CTGRAPH_ALLGATHER_RING_THRESHOLD;
  if (statex->nLocalRanks() > 1) {
    return (!largeMessage && ctran::utils::isPowerOfTwo(statex->nNodes()))
        ? NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline
        : NCCL_ALLGATHER_ALGO::ctgraph_pipeline;
  }
  return (!largeMessage && ctran::utils::isPowerOfTwo(statex->nRanks()))
      ? NCCL_ALLGATHER_ALGO::ctgraph_rd
      : NCCL_ALLGATHER_ALGO::ctgraph_ring;
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

  // Each branch handles registration, cleanup guard, execute, and deferred
  // cleanup transfer in one block so the full lifecycle is visible together.
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
    case NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline: {
      // Split a local-NVL clique comm to register the recvbuff window on, so
      // only the NVL peers' addresses are persisted; the cross-node rail rkeys
      // are exchanged in allGatherWinInit. nvlComm must outlive all graph
      // replays, so it is moved into the deferred-cleanup closure below.
      std::shared_ptr<CtranComm> nvlComm;
      FB_COMMCHECK(ctranCommSplitLocalNvl(comm, &nvlComm));

      ctran::CtranWin* nvlWin = nullptr;
      CtranPersistentRequest* request = nullptr;
      FB_COMMCHECK(winPersistBuffReg(
          recvbuff, recvBytes, comm, nvlComm.get(), stream, &nvlWin, &request));
      FB_CHECKABORT(
          nvlWin != nullptr, "winPersistBuffReg succeeded but nvlWin is null");
      FB_CHECKABORT(
          request != nullptr,
          "winPersistBuffReg succeeded but request is null");

      auto cleanup = [request, nvlWin, nvlComm = std::move(nvlComm)]() mutable {
        ctran::allGatherWinDestroy(request);
        delete request;
        nvlWin->free(true /* skipBarrier */);
        delete nvlWin;
        nvlComm.reset();
      };
      auto cleanupGuard = folly::makeGuard(cleanup);

      auto* pAlgo =
          reinterpret_cast<ctran::allgatherp::AlgoImpl*>(request->algo);
      if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline) {
        FB_COMMCHECK(pAlgo->execStreamedRecursiveDoubling(
            sendbuff, sendcount, datatype));
      } else {
        FB_COMMCHECK(pAlgo->execPipeline(sendbuff, sendcount, datatype));
      }

      cleanupGuard.dismiss();
      comm->cudagraphDeferredCleanup.add(std::move(cleanup));
      break;
    }
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
    case NCCL_ALLGATHER_ALGO::ctgraph_rd: {
      FB_COMMCHECK(localPersistBuffReg(recvbuff, recvBytes));

      auto cleanup = [recvbuff, recvBytes]() {
        ctran::globalDeregisterWithPtr(recvbuff, recvBytes);
      };
      auto cleanupGuard = folly::makeGuard(cleanup);

      if (algo == NCCL_ALLGATHER_ALGO::ctgraph_ring) {
        FB_COMMCHECK(ctranAllGatherRing(
            sendbuff, recvbuff, sendcount, datatype, comm, stream));
      } else {
        FB_COMMCHECK(ctranAllGatherStreamedRd(
            sendbuff, recvbuff, sendcount, datatype, comm, stream));
      }

      cleanupGuard.dismiss();
      comm->cudagraphDeferredCleanup.add(std::move(cleanup));
      break;
    }
    default:
      FB_ERRORRETURN(
          commInvalidArgument,
          "Unexpected algo {} in ctranAllGatherCudagraphAware",
          allGatherAlgoName(algo));
  }

  return commSuccess;
}
