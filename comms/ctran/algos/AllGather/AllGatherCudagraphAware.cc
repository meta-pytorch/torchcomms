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
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/utils/MathUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::allgatherp::createPersistentRequest;
using ctran::allgatherp::destroyPersistentRequest;
using ctran::utils::cudagraph::registerGraphDestroyCallback;

namespace {

// FIXME: consolidate with eager path AllgatherP
commResult_t exchangeIpcMemHdl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  auto* op = opGroup.front().get();
  auto* pArgs = reinterpret_cast<ctran::allgatherp::PersistArgs*>(
      op->allgatherp_init.pArgs);
  CtranComm* comm = op->comm_;
  auto* mapper = comm->ctran_->mapper.get();
  const auto* const statex = comm->statex_.get();
  const int myRank = statex->rank();
  const int nLocalRanks = statex->nLocalRanks();

  std::unordered_map<std::string_view, double> timers;

  {
    CtranMapperTimer timer;
    FB_COMMCHECK(mapper->intraAllGatherCtrl(
        pArgs->recvbuff,
        pArgs->recvHdl,
        pArgs->remoteRecvBuffs,
        pArgs->remoteAccessKeys,
        pArgs->remoteIpcRegHdls_));
    timers["ipcExchange"] = timer.durationUs();
  }

  {
    CtranMapperTimer timer;
    FB_COMMCHECK(mapper->intraBarrier());
    timers["intraBarrier"] = timer.durationUs();
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-AGP: Rank {} graph init local exchange comm {} recvbuff {} recvHdl {} nLocalRanks {} commHash {:x}: "
      "initExchange breakdown [ipcExchange {} us, intraBarrier {} us]",
      myRank,
      (void*)comm,
      pArgs->recvbuff,
      pArgs->recvHdl,
      nLocalRanks,
      statex->commHash(),
      timers["ipcExchange"],
      timers["intraBarrier"]);
  for (int i = 0; i < nLocalRanks; ++i) {
    int peerRank = statex->localRankToRank(i);
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-AGP     Peer {}: addr {} ipcImport {}",
        i,
        pArgs->remoteRecvBuffs[peerRank],
        myRank == peerRank ? "(local)"
                           : pArgs->remoteIpcRegHdls_.at(i).toString());
  }

  pArgs->initState = ctran::allgatherp::InitState::kInitialized;
  return commSuccess;
}

// Acquire a graph-lifetime scoped local registration for recvbuff via the
// regcache CCA-cached path. Requires the segment to already be allocator-cached
// (CCA hook); otherwise acquireScopedRegister returns commInvalidUsage and this
// fails cleanly.
commResult_t setupScopedRegister(
    CtranComm* comm,
    void* const recvbuff,
    const size_t recvBytes,
    ctran::ScopedRegHdl& outRegHdl) {
  auto regCache = ctran::RegCache::getInstance();
  ctran::CHECK_VALID_REGCACHE(regCache);
  return regCache->acquireScopedRegister(
      recvbuff,
      recvBytes,
      comm->statex_->cudaDev(),
      comm->ctran_->mapper->getBackends(),
      outRegHdl);
}

commResult_t setupPersistentRequest(
    CtranComm* comm,
    CtranPersistentRequest* const request,
    cudaStream_t stream,
    void* const recvbuff,
    const size_t recvBytes,
    const size_t maxRecvCount,
    const commDataType_t datatype) {
  auto* algo = reinterpret_cast<ctran::allgatherp::AlgoImpl*>(request->algo);
  auto& pArgs = algo->pArgs;
  pArgs.recvbuff = recvbuff;
  pArgs.maxRecvCount = maxRecvCount;
  pArgs.datatype = datatype;

  // Acquire the local recv registration through the scoped regcache API. The
  // recvbuff's segment must already be allocator-cached (CCA hook); otherwise
  // acquireScopedRegister returns commInvalidUsage and fails cleanly. Scoped
  // handles are owned by persistent request and released at graph destroy.
  double scopedRegisterUs = 0;
  {
    ctran::ScopedRegHdl localRecvReg;
    CtranMapperTimer timer;
    FB_COMMCHECK(setupScopedRegister(comm, recvbuff, recvBytes, localRecvReg));
    pArgs.recvHdl = localRecvReg.get();
    pArgs.recvRegHdl_ =
        std::make_unique<ctran::ScopedRegHdl>(std::move(localRecvReg));
    scopedRegisterUs = timer.durationUs();
  }

  const int nRanks = comm->statex_->nRanks();
  const int nLocalRanks = comm->statex_->nLocalRanks();
  pArgs.remoteRecvBuffs.assign(nRanks, nullptr);
  pArgs.remoteAccessKeys.assign(nRanks, CtranMapperRemoteAccessKey{});

  const auto* const statex = comm->statex_.get();
  const int myRank = statex->rank();

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP_INIT,
      stream,
      "CtranAllGatherPGraphInit",
      comm->ctran_->getOpCount());

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLGATHERP_INIT, stream, comm, config.opCount);
  op->allgatherp_init.pArgs = &pArgs;
  opGroup.push_back(std::move(op));

  pArgs.initState = ctran::allgatherp::InitState::kSubmitted;
  auto submitGuard = folly::makeGuard([&pArgs] {
    pArgs.initState = ctran::allgatherp::InitState::kUninitialized;
  });

  double initExchangeUs = 0;
  {
    CtranMapperTimer timer;
    FB_COMMCHECK(comm->ctran_->gpe->submitHost(
        std::move(opGroup),
        exchangeIpcMemHdl,
        config,
        /* cpuFlag */ nullptr));
    submitGuard.dismiss();

    while (pArgs.initState.load() !=
           ctran::allgatherp::InitState::kInitialized) {
      FB_COMMCHECK(comm->getAsyncResult());
    }
    initExchangeUs = timer.durationUs();
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-AGP: Rank {} setupPersistentRequest comm {} recvbuff {} recvHdl {} nLocalRanks {} commHash {:x}: "
      "scopedRegister {} us, initExchange {} us",
      myRank,
      (void*)comm,
      recvbuff,
      pArgs.recvHdl,
      nLocalRanks,
      statex->commHash(),
      scopedRegisterUs,
      initExchangeUs);

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
      FB_COMMCHECK(createPersistentRequest(comm, stream, &request));
      auto cleanupGuard = folly::makeGuard([request]() {
        FB_COMMCHECKIGNORE(destroyPersistentRequest(request));
        delete request;
      });

      FB_COMMCHECK(setupPersistentRequest(
          comm,
          request,
          stream,
          recvbuff,
          recvBytes,
          sendcount * nRanks,
          datatype));

      // Hand teardown to the captured graph (SW-only) BEFORE capturing the
      // collective ops: if op capture fails, the graph tears down the request
      // via this callback instead of releasing its NVL imports / registration
      // while the captured graph still references them.
      auto destroyCb = [](void* p) {
        auto* req = static_cast<CtranPersistentRequest*>(p);
        FB_COMMCHECKIGNORE(destroyPersistentRequest(req));
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
      ctran::ScopedRegHdl localRecvReg;
      FB_COMMCHECK(
          setupScopedRegister(comm, recvbuff, recvBytes, localRecvReg));
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
