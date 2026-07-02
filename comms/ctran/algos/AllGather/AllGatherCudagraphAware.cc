// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Cudagraph-aware AllGather: dispatches a ctgraph* AllGather captured into a
// CUDA graph. The algo is explicitly specified or auto-selected by
// selectCtgraphAlgo() from topology and message size. Two registration paths:
//
//   ctgraph_pipeline/rdpipeline (nLocalRanks > 1): the intra-node NVL broadcast
//     addresses must be stable across replays, so the recvbuff is registered as
//     a local-NVL window (createAllGatherPWithWindow); the cross-node rail
//     rkeys are exchanged in the exec path.
//
//   ctgraph_ring/rd (nLocalRanks == 1): only the local registration persists
//     (localPersistBuffReg); remote peers are exchanged per replay in the GPE
//     host node.
//
// Resources are handed to deferred cleanup, which runs at comm destruction
// (after all replays) on the main thread.

#include <folly/ScopeGuard.h>
#include <memory>
#include <utility>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/AllGatherPWithWin.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/MathUtils.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

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

  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
    case NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline: {
      // Build the windowed persistent algo, run it, then hand the
      // request/window/nvlComm to the deferred cleanup -- they must outlive all
      // graph replays.
      CtranPersistentRequest* request = nullptr;
      FB_COMMCHECK(
          ctran::createAllGatherPWithWindow(
              comm, recvbuff, recvBytes, stream, &request));
      // No localPersistBuffReg here (cf. the ring/rd branch below): registering
      // recvbuff as a window in createAllGatherPWithWindow is also its local
      // registration, which the algo uses (via pArgs->recvHdl) for the rail
      // puts.
      auto cleanup = [request]() {
        ctran::destroyAllGatherPWithWindow(request);
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

      // Exec succeeded; hand ownership to the deferred cleanup (runs at comm
      // destroy, after all graph replays).
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
