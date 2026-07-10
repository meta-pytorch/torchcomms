// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/regcache/IpcRegCache.h"

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
  auto mapper = comm->ctran_->mapper.get();

  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  CtranAlgoLogger logger(algoInitName, op->opCount, comm);

  pArgs->remoteAccessKeys.resize(nRanks, CtranMapperRemoteAccessKey());
  pArgs->remoteRecvBuffs.resize(nRanks, nullptr);
  // Imports are self-managed by AGP (adopted into pArgs.remoteIpcRegHdls_ and
  // released at eager/graph destroy), so recordExport=false keeps the
  // export-notify path from double-releasing them. allGatherCtrl sizes
  // remoteIpcRegHdls_ to nLocalRanks and fills the imported NVL peers' handles
  // (indexed by local rank; self/non-imported slots stay empty).
  FB_COMMCHECK(mapper->allGatherCtrl(
      pArgs->recvbuff,
      pArgs->recvHdl,
      pArgs->remoteRecvBuffs,
      pArgs->remoteAccessKeys,
      CtranMapperBackend::UNSET,
      /*recordExport=*/false,
      &pArgs->remoteIpcRegHdls_));

  // Ensure all ranks have finished remote importing before return
  FB_COMMCHECK(mapper->barrier());

  // The full-comm exchange above already imported the rail peer rkeys, so the
  // gpeFn's per-replay handshake skips the rkey exchange and only re-syncs.
  pArgs->ibKeysExchanged = true;

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

  // Mark the remote registration as initialized, so that consequent execution
  // can schedule CE based NVL copy
  pArgs->initState = InitState::kInitialized;
  return commSuccess;
}

extern __global__ void ncclKernelAllGatherPInit(
    int* flag,
    CtranAlgoDeviceState* devState);

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
    CtranPersistentRequest** out) {
  if (out == nullptr) {
    return commInvalidArgument;
  }
  *out = nullptr;

  auto algo = std::make_unique<AlgoImpl>(comm, stream);
  FB_COMMCHECK(algo->initResources());

  auto request = std::make_unique<CtranPersistentRequest>(
      CtranPersistentRequest::Type::ALLGATHER_P, comm, stream);
  request->algo = algo.release();
  *out = request.release();
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

commResult_t AlgoImpl::initialize() {
  auto opCount = comm_->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      algoInitName,
      nullptr,
      pArgs.recvbuff,
      pArgs.maxRecvCount,
      pArgs.datatype,
      -1,
      comm_,
      stream_);

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP_INIT,
      stream_,
      algoInitName,
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = comm_->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLGATHERP_INIT, stream_, comm_, opCount);
  op->allgatherp_init.pArgs = &pArgs;
  opGroup.push_back(std::move(op));

  // Publish kSubmitted BEFORE handing work to the GPE thread. exchangeMemHdl
  // sets kInitialized on the GPE thread; setting kSubmitted after submit could
  // clobber that back to kSubmitted and deadlock waitInit()/destroy(). If
  // submit fails, reset to kUninitialized so a later destroy() does not wait on
  // an init that never ran.
  pArgs.initState = InitState::kSubmitted;
  auto submitGuard =
      folly::makeGuard([this] { pArgs.initState = InitState::kUninitialized; });
  FB_COMMCHECK(comm_->ctran_->gpe->submit(
      std::move(opGroup),
      exchangeMemHdl,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherPInit)));
  submitGuard.dismiss();

  return commSuccess;
};

AlgoImpl::~AlgoImpl() = default;

commResult_t AlgoImpl::destroy() {
  // Async init populates pArgs on the GPE thread; wait for it only while that
  // async init is still in flight (kSubmitted).
  if (pArgs.initState.load() == InitState::kSubmitted) {
    FB_COMMCHECK(waitInit());
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
  return commSuccess;
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
  auto mapper = comm->ctran_->mapper.get();
  const auto maxRecvSize = maxRecvCount * commTypeSize(datatype);
  void* recvHdl;
  bool localRegRecv;

  FB_COMMCHECK(
      mapper->searchRegHandle(recvbuff, maxRecvSize, &recvHdl, &localRegRecv));
  if (localRegRecv) {
    FB_COMMCHECK(mapper->deregDynamic(recvHdl));
    CLOGF(
        ERR,
        "recvbuff is not registered. Pointer: {} length: {}",
        (void*)recvbuff,
        maxRecvSize);
    return commInternalError;
  }

  FB_COMMCHECK(createPersistentRequest(comm, stream, &request));
  auto requestGuard = folly::makeGuard([&request] {
    FB_COMMCHECKIGNORE(destroyPersistentRequest(request));
    delete request;
    request = nullptr;
  });

  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  // Set up persistent arguments
  algo->pArgs.recvHdl = recvHdl;
  algo->pArgs.recvbuff = recvbuff;
  algo->pArgs.maxRecvCount = maxRecvCount;
  algo->pArgs.datatype = datatype;

  // Initialize algo internal resource and schedule handle exchange on GPE
  // thread
  FB_COMMCHECK(algo->initialize());

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
    case NCCL_ALLGATHER_P_ALGO::ctrdpipeline:
      return algo->execRecursiveDoubling(sendbuff, count, datatype);
    case NCCL_ALLGATHER_P_ALGO::ctsrdpipeline:
      return algo->execStreamedRecursiveDoubling(sendbuff, count, datatype);
    default:
      return ErrorStackTraceUtil::log(commInternalError);
  }
}

commResult_t allGatherPDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  // No need to dereg handles now since user should call explicit
  // commDeregister before buffer is freed
  FB_COMMCHECK(destroyPersistentRequest(request));

  // destroyPersistentRequest released this request's remote NVL IPC imports
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
