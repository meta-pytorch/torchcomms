// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/utils/logger/LogUtils.h"

#include <folly/ScopeGuard.h>

using ctran::alltoallp::AlgoImpl;

#define CHECK_VALID_PREQ(pReq)                                    \
  do {                                                            \
    if (pReq->type != CtranPersistentRequest::Type::ALLTOALL_P) { \
      FB_ERRORRETURN(                                             \
          commInvalidArgument,                                    \
          "Unexpected PersistentRequest type {} called into {}",  \
          pReq->type,                                             \
          __func__);                                              \
    }                                                             \
  } while (0)

namespace ctran {
commResult_t AllToAllPInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request) {
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  SetCudaDevRAII setCudaDev(statex->cudaDev());

  std::string skip_ctrl_msg;
  hints.get("ncclx_alltoallp_skip_ctrl_msg_exchange", skip_ctrl_msg);
  // createPersistentRequest owns the scoped local recv registration, the
  // read-only pArgs fields (shared with the graph path), and the async IPC
  // handle exchange submitted on the GPE thread; it fails cleanly with
  // commInvalidUsage if recvbuff is not allocator-cached. waitForInit=false:
  // eager init returns without waiting; the first exec waits via waitInit().
  FB_COMMCHECK(
      ctran::alltoallp::createPersistentRequest(
          comm,
          stream,
          recvbuff,
          maxRecvCount,
          datatype,
          &request,
          /*waitForInit=*/false,
          /*skipCtrlMsg=*/skip_ctrl_msg == "true"));
  auto requestGuard = folly::makeGuard([&request, comm] {
    // Unregister the cleanup token before deleting the request: the token's
    // closure captures the raw request pointer, so leaving it in the comm
    // registry would let a later drainPersistentCleanups() run it on freed
    // memory (use-after-free).
    comm->unregisterPersistentCleanup(request->cleanup_);
    FB_COMMCHECKIGNORE(ctran::alltoallp::destroyPersistentRequest(request));
    delete request;
    request = nullptr;
  });

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllToAllPInit: rank {} initialized request {}, recvbuff {}, comm {} commHash {:x} commDesc {} [nranks={}, localRanks={}] stream={}",
      statex->rank(),
      (void*)request,
      (void*)recvbuff,
      (void*)comm,
      statex->commHash(),
      statex->commDesc(),
      nRanks,
      statex->nLocalRanks(),
      (void*)stream);

  requestGuard.dismiss();
  return commSuccess;
}

// active buffer that user is using for this run
commResult_t AllToAllPExec(
    const void* sendbuff,
    const size_t count,
    CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);
  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  CLOGF_TRACE(
      COLL,
      "AllToAllPExec: rank {} started executing request {}, sendbuff {} recvbuff {} count {} ",
      request->comm_->statex_->rank(),
      (void*)request,
      sendbuff,
      (void*)algo->pArgs.recvbuff,
      count);
  return algo->exec(sendbuff, count);
}

commResult_t AllToAllPDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  // Route the resource release through the shared cleanup token so it runs at
  // most once across {eager free, graph-destroy callback, comm drain before
  // terminate()}. The request object itself is still owned/deleted by the
  // caller (unchanged contract); the token releases only the scoped
  // registration.
  // A request reaching AllToAllPDestroy is always Type::ALLTOALL_P, created by
  // createPersistentRequest, which always sets cleanup_ (other request types,
  // e.g. dedup, have their own destroy path and never reach here).
  auto token = request->cleanup_;
  auto* comm = request->comm_;
  DCHECK(token != nullptr);
  token->run();
  comm->unregisterPersistentCleanup(token);

  // The cleanup token released this request's remote NVL IPC imports deferred
  // (SW-only). This is a user thread, so drain the parked cuMemUnmaps now.
  auto ipcRegCache = ctran::IpcRegCache::getInstance();
  ctran::CHECK_VALID_IPC_REGCACHE(ipcRegCache);
  ipcRegCache->cleanupInvalidImports();

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "AllToAllPDestroy: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);

  return commSuccess;
}

bool AllToAllPSupport(CtranComm* comm) {
  if (!ctranInitialized(comm)) {
    return false;
  }
  const auto statex = comm->statex_.get();

  for (int rank = 0; rank < statex->nRanks(); rank++) {
    if (comm->ctran_->mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
      return false;
    }
  }

  return true;
}
} // namespace ctran
