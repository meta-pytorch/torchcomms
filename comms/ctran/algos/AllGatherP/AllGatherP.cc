// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/CommUtils.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/regcache/RegCache.h"

#include <folly/ScopeGuard.h>

#include <memory>

using ctran::algos::GpeKernelSync;
using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::createPersistentRequest;
using ctran::allgatherp::destroyPersistentRequest;

#define CHECK_VALID_PREQ(pReq)                                     \
  do {                                                             \
    if (pReq->type != CtranPersistentRequest::Type::ALLGATHER_P) { \
      FB_ERRORRETURN(                                              \
          commInvalidArgument,                                     \
          "Unexpected PersistentRequest type {} called into {}",   \
          pReq->type,                                              \
          __func__);                                               \
    }                                                              \
  } while (0)

namespace ctran::allgatherp {
const std::string algoInitName = "CtranAllGatherPInit";

commResult_t exchangeMemHdl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->allgatherp_init.pArgs);
  CtranComm* comm = opGroup.front()->comm_;

  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  CtranAlgoLogger logger(algoInitName, op->opCount, comm);

  // Async (GPE-thread) invocation of the shared IPC-handle exchange. Imports
  // only the intra-node NVL IPC handles (self-managed by AGP, released at
  // eager/graph destroy) and marks initState kInitialized. remoteRecvBuffs /
  // remoteAccessKeys are indexed by global rank; only local NVL-peer slots are
  // filled (self and inter-node IB peers stay UNSET). Leaves ibKeysExchanged
  // false so the first exec performs the real inter-node IB rkey exchange
  // (matching the graph path); subsequent execs only re-sync.
  FB_COMMCHECK(exchangeIpcReg(comm, *pArgs));

  if (NCCL_CTRAN_ENABLE_TRACE_LOG) {
    for (int i = 0; i < nRanks; i++) {
      CLOGF_TRACE(
          INIT,
          "    remoteRecvBuffs[{}]: {}, remoteAccessKey: {}",
          i,
          (void*)pArgs->remoteRecvBuffs[i],
          pArgs->remoteAccessKeys[i].toString());
    }
  }

  return commSuccess;
}

commResult_t AlgoImpl::initResources() {
  std::vector<GpeKernelSync*> gpeKernelSyncs;
  FB_COMMCHECK(comm_->ctran_->gpe->allocGpeKernelSyncs(
      1 /* count */, 1 /* nworkers */, gpeKernelSyncs));
  if (gpeKernelSyncs.empty()) {
    FB_ERRORRETURN(commInternalError, "allocGpeKernelSyncs returned no syncs");
  }

  resource_.pipeSync = gpeKernelSyncs.front();
  return commSuccess;
}

commResult_t createPersistentRequest(
    CtranComm* comm,
    cudaStream_t stream,
    void* recvbuff,
    size_t maxRecvCount,
    commDataType_t datatype,
    CtranPersistentRequest** out,
    bool waitForInit) {
  if (out == nullptr) {
    return commInvalidArgument;
  }
  *out = nullptr;

  // Acquire the local recv registration through the scoped regcache API FIRST,
  // before allocating any pooled resource (initResources allocates a
  // GpeKernelSync into pArgs.pipeSync). The recvbuff's segment must already be
  // allocator-cached (CCA hook); otherwise acquireScopedRegister returns
  // commInvalidUsage and we return immediately with nothing allocated (no
  // leak). Doing the scoped registration before initResources avoids leaking a
  // pooled GpeKernelSync when it fails, since the local AlgoImpl below would
  // only run its (empty) destructor, not AlgoImpl::destroy().
  const size_t recvBytes = maxRecvCount * commTypeSize(datatype);
  ctran::ScopedRegHdl localRecvReg;
  CtranMapperTimer scopedRegisterTimer;
  auto regCache = ctran::RegCache::getInstance();
  ctran::CHECK_VALID_REGCACHE(regCache);
  FB_COMMCHECK(regCache->acquireScopedRegister(
      recvbuff,
      recvBytes,
      comm->statex_->cudaDev(),
      comm->ctran_->mapper->getBackends(),
      localRecvReg));
  const double scopedRegisterUs = scopedRegisterTimer.durationUs();

  auto algo = std::make_unique<AlgoImpl>(comm, stream);
  FB_COMMCHECK(algo->initResources());

  // The unique_ptr locals auto-clean the AlgoImpl/request on any failure below.
  // The scoped handle is owned by the persistent request and released at
  // destroy.
  algo->pArgs.recvHdl = localRecvReg.get();
  algo->pArgs.recvRegHdl_ =
      std::make_unique<ctran::ScopedRegHdl>(std::move(localRecvReg));
  algo->pArgs.recvbuff = recvbuff;
  algo->pArgs.maxRecvCount = maxRecvCount;
  algo->pArgs.datatype = datatype;

  auto request = std::make_unique<CtranPersistentRequest>(
      CtranPersistentRequest::Type::ALLGATHER_P, comm, stream);
  request->algo = algo.release();
  // Once request->algo is set, the AlgoImpl (and its pooled GpeKernelSync) is
  // only released via destroyPersistentRequest, not the unique_ptr destructor.
  // Guard the pooled resource against early returns below and dismiss right
  // before releasing ownership to *out.
  auto reqGuard = folly::makeGuard([&request] {
    FB_COMMCHECKIGNORE(destroyPersistentRequest(request.get()));
  });

  auto* algoPtr = reinterpret_cast<AlgoImpl*>(request->algo);

  double ipcExchangeUs = 0.0;
  {
    CtranMapperTimer ipcExchangeTimer;

    // Dummy placeholder for existing submitHost API, no actual kernel launch
    // FIXME: submitHost() should not require it
    KernelConfig config(
        KernelConfig::KernelType::ALLGATHERP_INIT,
        stream,
        "CtranAllGatherPInit",
        comm->ctran_->getOpCount());

    std::vector<std::unique_ptr<OpElem>> opGroup;
    auto op = std::make_unique<OpElem>(
        OpElem::opType::ALLGATHERP_INIT, stream, comm, config.opCount);
    op->allgatherp_init.pArgs = &algoPtr->pArgs;
    opGroup.push_back(std::move(op));

    // Publish kSubmitted BEFORE handing work to the GPE thread. exchangeMemHdl
    // sets kInitialized on the GPE thread; setting kSubmitted after submit
    // could clobber that back to kSubmitted and deadlock waitInit()/destroy().
    // If submit fails, reset to kUninitialized so a later destroy() does not
    // wait on an init that never ran.
    algoPtr->pArgs.initState = InitState::kSubmitted;
    auto submitGuard = folly::makeGuard(
        [algoPtr] { algoPtr->pArgs.initState = InitState::kUninitialized; });
    FB_COMMCHECK(comm->ctran_->gpe->submitHost(
        std::move(opGroup), exchangeMemHdl, config, /* cpuFlag */ nullptr));
    submitGuard.dismiss();

    // Eager init returns async; graph capture waits synchronously so the
    // captured collective ops see fully-populated pArgs.
    if (waitForInit) {
      FB_COMMCHECK(algoPtr->waitInit());
    }
    ipcExchangeUs = ipcExchangeTimer.durationUs();
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-AGP: Rank {} createPersistentRequest ({}): comm {} recvbuff {} recvHdl {} nLocalRanks {} commHash {:x}: scopedRegister {} us, ipcExchange {} us",
      comm->statex_->rank(),
      waitForInit ? "graph" : "eager",
      (void*)comm,
      algoPtr->pArgs.recvbuff,
      algoPtr->pArgs.recvHdl,
      comm->statex_->nLocalRanks(),
      comm->statex_->commHash(),
      scopedRegisterUs,
      ipcExchangeUs);

  reqGuard.dismiss();
  *out = request.release();

  // Route teardown through the one-shot cleanup token: it releases the
  // pooled pipeSync + scoped registration (via destroyPersistentRequest),
  // running at most once regardless of which path (eager free, graph-destroy
  // callback, comm drain before terminate()) fires first. It does NOT delete
  // the request object -- that stays with the object's owner. The closure
  // captures the RAW request pointer (NOT a shared_ptr) so the token does not
  // co-own the request -- no ownership cycle. The token is co-owned by the
  // request (cleanup_), the comm registry, and (for graph) the graph
  // user-object.
  auto cleanup = std::make_shared<PersistentCleanup>([request = *out]() {
    FB_COMMCHECKIGNORE(destroyPersistentRequest(request));
  });
  (*out)->cleanup_ = cleanup;
  comm->registerPersistentCleanup(cleanup);
  return commSuccess;
}

commResult_t destroyPersistentRequest(CtranPersistentRequest* const request) {
  if (request == nullptr) {
    return commSuccess;
  }
  auto res = commSuccess;
  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  if (algo != nullptr) {
    // destroy() is best-effort; delete unconditionally so a partial failure
    // does not leak the AlgoImpl or leave request->algo dangling.
    res = algo->destroy();
    delete algo;
    request->algo = nullptr;
  }
  return res;
}

AlgoImpl::~AlgoImpl() = default;

commResult_t AlgoImpl::destroy() {
  // Async init populates pArgs on the GPE thread; wait for it only while that
  // async init is still in flight (kSubmitted). Capture the result rather than
  // early-returning, so the cleanup below always runs -- otherwise a failed
  // async init would leak the pooled GpeKernelSync slot (never reset()).
  auto res = commSuccess;
  if (pArgs.initState.load() == InitState::kSubmitted) {
    res = waitInit();
  }
  if (resource_.pipeSync != nullptr) {
    resource_.pipeSync->reset();
    resource_.pipeSync = nullptr;
  }
  // Release the scoped NVL IPC imports and the scoped local recv registration.
  // Both are deferred/SW-only (no CUDA), so safe from the graph-destroy
  // callback. Later explicit cleanupInvalidImports() or access to regCache
  // would do actual release.
  pArgs.remoteIpcRegHdls_.clear();
  pArgs.recvRegHdl_.reset();
  return res;
}
} // namespace ctran::allgatherp

namespace ctran {
bool allGatherPSupport(CtranComm* comm) {
  if (!ctranInitialized(comm)) {
    return false;
  }

  const auto statex = comm->statex_.get();
  auto mapper = comm->ctran_->mapper.get();
  const auto myRank = statex->rank();
  // Check if all remote peers are supported by ctran
  for (auto rank = 0; rank < statex->nRanks(); rank++) {
    if (mapper->getBackend(rank) == CtranMapperBackend::UNSET &&
        rank != myRank) {
      return false;
    }
  }

  return true;
}

commResult_t allGatherPInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request) {
  // createPersistentRequest owns the scoped local recv registration, the
  // read-only pArgs fields (shared with the graph path), and the async IPC
  // handle exchange submitted on the GPE thread; it fails cleanly with
  // commInvalidUsage if recvbuff is not allocator-cached. waitForInit=false:
  // eager init returns without waiting; the first exec waits via waitInit().
  FB_COMMCHECK(createPersistentRequest(
      comm,
      stream,
      recvbuff,
      maxRecvCount,
      datatype,
      &request,
      /*waitForInit=*/false));
  auto requestGuard = folly::makeGuard([&request, comm] {
    comm->unregisterPersistentCleanup(request->cleanup_);
    FB_COMMCHECKIGNORE(destroyPersistentRequest(request));
    delete request;
    request = nullptr;
  });

  requestGuard.dismiss();
  return commSuccess;
}

commResult_t allGatherPExec(
    const void* sendbuff,
    const size_t count,
    commDataType_t datatype,
    CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  auto algo = reinterpret_cast<AlgoImpl*>(request->algo);
  const auto nRanks = request->comm_->statex_->nRanks();
  if (count * nRanks * commTypeSize(datatype) >
      algo->pArgs.maxRecvCount * commTypeSize(algo->pArgs.datatype)) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "AllGatherP invalid sendbuff count {} * nRanks {} * sizeof datatype {} exceeds maxRecvCount {} * sizeof datatype {}.",
        count,
        nRanks,
        datatype,
        algo->pArgs.maxRecvCount,
        algo->pArgs.datatype);
  }

  switch (NCCL_ALLGATHER_P_ALGO) {
    case NCCL_ALLGATHER_P_ALGO::ctdirect:
      return algo->execDirect(sendbuff, count, datatype);
    case NCCL_ALLGATHER_P_ALGO::ctpipeline:
      return algo->execPipeline(sendbuff, count, datatype);
    case NCCL_ALLGATHER_P_ALGO::ctsrdpipeline:
      return algo->execStreamedRecursiveDoubling(sendbuff, count, datatype);
    default:
      return ErrorStackTraceUtil::log(commInternalError);
  }
}

commResult_t allGatherPDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  // Route the resource release through the shared cleanup token so it runs at
  // most once across {eager free, graph-destroy callback, comm drain before
  // terminate()}. If the comm was already destroyed (comm-then-free ordering),
  // the token was already run by the comm drain and this no-ops -- no hang, no
  // double release. The request object itself is still owned/deleted by the
  // caller (unchanged contract); the token releases only the pooled pipeSync +
  // scoped registration. Keep the token alive across run(), then drop the
  // comm's registry entry (comm is alive on this eager path).
  auto token = request->cleanup_;
  auto* comm = request->comm_;
  DCHECK(token != nullptr);
  token->run();
  comm->unregisterPersistentCleanup(token);

  // The cleanup token released this request's remote NVL IPC imports
  // deferred (SW-only, so the shared teardown is also safe from the
  // graph-destroy callback). This is a user thread, so drain the parked
  // cuMemUnmaps now rather than leaving them for the next IpcRegCache entry /
  // comm teardown.
  auto ipcRegCache = ctran::IpcRegCache::getInstance();
  ctran::CHECK_VALID_IPC_REGCACHE(ipcRegCache);
  ipcRegCache->cleanupInvalidImports();

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "allGatherPDestroy: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);

  return commSuccess;
}
} // namespace ctran
