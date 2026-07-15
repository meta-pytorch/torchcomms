// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Cudagraph-aware AllGather: dispatches a ctgraph* AllGather captured into a
// CUDA graph. The algo is explicitly specified or auto-selected by
// selectCtgraphAlgo() from topology and message size. Two registration paths:
//
//   ctgraph_pipeline/rdpipeline (nLocalRanks > 1): the intra-node NVL broadcast
//     addresses must be stable across replays, so the recvbuff is registered
//     via a scoped regcache handle owned by the persistent AllGatherP request,
//     and the local NVL IPC imports are self-managed by the request (released
//     at graph destroy); cross-node rail rkeys are exchanged in the exec path.
//     Teardown is triggerd by the captured graph destory callback that releases
//     resource with refcount and defers CUDA release to later user thread call.
//
//   ctgraph_ring/rd (nLocalRanks == 1): the recvbuff is registered via a scoped
//     regcache handle held for the captured graph's lifetime and released by a
//     dedicated graph-destroy callback (same model as the pipeline branch,
//     minus the IPC imports); remote peers are exchanged per replay
//     in the GPE host node.

#include <folly/ScopeGuard.h>
#include <memory>
#include <utility>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/CommUtils.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/utils/MathUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::allgatherp::createPersistentRequest;
using ctran::allgatherp::destroyPersistentRequest;
using ctran::utils::cudagraph::registerGraphDestroyCallback;

namespace {

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
      INIT,
      "CTRAN-AGP: AllGather cudagraph-aware: algo {} "
      "sendcount {} recvbuff {} recvBytes {} commHash {:x} commDesc {} nRanks {} nLocalRanks {} nNodes {}",
      allGatherAlgoName(algo),
      sendcount,
      recvbuff,
      recvBytes,
      statex->commHash(),
      statex->commDesc(),
      nRanks,
      statex->nLocalRanks(),
      statex->nNodes());

  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
    case NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline: {
      // FIXME: ctgraph_pipeline/rdpipeline should handle nLocalRanks == 1 too,
      // so consolidate with ctgraph_ring|ctgraph_rd
      if (statex->nLocalRanks() <= 1) {
        FB_ERRORRETURN(
            commInvalidUsage,
            "ctgraph_pipeline/rdpipeline requires nLocalRanks > 1; got {}",
            statex->nLocalRanks());
      }

      CtranPersistentRequest* request = nullptr;
      FB_COMMCHECK(createPersistentRequest(
          comm,
          stream,
          recvbuff,
          sendcount * nRanks,
          datatype,
          &request,
          // Need wait for IPC exchange before capture CE copy
          /*waitForInit=*/true));
      auto cleanupGuard = folly::makeGuard([request, comm]() {
        // Unregister the cleanup token before deleting the request: the
        // token's closure captures the raw request pointer, so leaving it in
        // the comm registry would let a later drainPersistentCleanups() run it
        // on freed memory (use-after-free).
        comm->unregisterPersistentCleanup(request->cleanup_);
        FB_COMMCHECKIGNORE(destroyPersistentRequest(request));
        delete request;
      });

      // Hand teardown to the captured graph (SW-only) BEFORE capturing the
      // collective ops: if op capture fails, the graph tears down the request
      // via this callback instead of releasing its NVL imports / registration
      // while the captured graph still references them.
      //
      // Release the pooled pipeSync + scoped registration through the shared
      // cleanup token (runs at most once). If the comm was destroyed first
      // (comm-then-graph ordering), the comm drain already ran the token and
      // this no-ops -- the token's shared_ptr lives in the request, which the
      // comm drain does NOT delete, so it is safe to touch here. Never
      // unregister from the comm registry in this callback: it may run after
      // the comm is gone. The spent token therefore lingers in the comm's
      // persistentCleanups_ registry until CtranComm::destroy() drains it;
      // this is bounded by the number of distinct graph captures sharing (and
      // destroyed during the lifetime of) this comm -- typically O(1) in
      // steady-state training (capture once, replay many) -- so it is an
      // acceptable bounded retention, not an unbounded leak. The graph owns the
      // request object (never returned to the user), so this callback deletes
      // it after releasing resources.
      auto destroyCb = [](void* p) {
        auto* req = static_cast<CtranPersistentRequest*>(p);
        DCHECK(req->cleanup_ != nullptr);
        req->cleanup_->run();
        delete req;
      };
      FB_COMMCHECK(registerGraphDestroyCallback(stream, request, destroyCb));
      cleanupGuard.dismiss();

      auto* pAlgo =
          reinterpret_cast<ctran::allgatherp::AlgoImpl*>(request->algo);
      if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline) {
        FB_COMMCHECK(pAlgo->execStreamedRecursiveDoubling(
            sendbuff, sendcount, datatype));
      } else {
        FB_COMMCHECK(pAlgo->execPipeline(sendbuff, sendcount, datatype));
      }
      break;
    }
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
    case NCCL_ALLGATHER_ALGO::ctgraph_rd: {
      // Scoped local registration of recvbuff, owned for the captured graph's
      // lifetime. Require segment already be allocator-cached (CCA hook).
      // FIXME: ctgraph_ring|ctgraph_rd can be consolidated with
      // ctgraph_{*}pipeline too.
      ctran::ScopedRegHdl localRecvReg;
      auto regCache = ctran::RegCache::getInstance();
      ctran::CHECK_VALID_REGCACHE(regCache);
      FB_COMMCHECK(regCache->acquireScopedRegister(
          recvbuff,
          recvBytes,
          comm->statex_->cudaDev(),
          comm->ctran_->mapper->getBackends(),
          localRecvReg));

      // Heap-own the scoped ref so a graph-destroy callback can release it.
      auto* heldReg = new ctran::ScopedRegHdl(std::move(localRecvReg));
      auto cleanupGuard = folly::makeGuard([heldReg]() { delete heldReg; });

      // SW-only release, safe in the CUDA user-object destroy context. Actual
      // deregistration is deferred to next regcache access.
      auto destroyCb = [](void* p) {
        delete static_cast<ctran::ScopedRegHdl*>(p);
      };
      FB_COMMCHECK(registerGraphDestroyCallback(stream, heldReg, destroyCb));
      cleanupGuard.dismiss();

      if (algo == NCCL_ALLGATHER_ALGO::ctgraph_ring) {
        FB_COMMCHECK(ctranAllGatherRing(
            sendbuff, recvbuff, sendcount, datatype, comm, stream));
      } else {
        FB_COMMCHECK(ctranAllGatherStreamedRd(
            sendbuff, recvbuff, sendcount, datatype, comm, stream));
      }
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
