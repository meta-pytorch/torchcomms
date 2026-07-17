// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Cudagraph-aware AllToAll: dispatches a ctgraph AllToAll captured into a
// CUDA graph via the persistent window-based AllToAllP algorithm. The recvbuff
// is registered via a scoped regcache handle owned by the persistent
// AllToAllP request, and the intra-node NVL IPC imports are self-managed by
// the request (released at graph destroy). Teardown is triggered by the
// captured graph destroy callback that releases resources with refcount and
// defers CUDA release to a later user-thread call.

#include <folly/ScopeGuard.h>
#include <memory>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::alltoallp::createPersistentRequest;
using ctran::alltoallp::destroyPersistentRequest;
using ctran::utils::cudagraph::registerGraphDestroyCallback;

commResult_t ctranAllToAllCudagraphAware(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLTOALL_ALGO algo) {
  const auto statex = comm->statex_.get();
  const int nRanks = statex->nRanks();

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-A2AP: AllToAll cudagraph-aware: algo {} "
      "count {} recvbuff {} commHash {:x} commDesc {} nRanks {} nLocalRanks {} nNodes {}",
      allToAllAlgoName(algo),
      count,
      recvbuff,
      statex->commHash(),
      statex->commDesc(),
      nRanks,
      statex->nLocalRanks(),
      statex->nNodes());

  // AllToAll recvbuff is full-size (count per peer across all ranks).
  CtranPersistentRequest* request = nullptr;
  FB_COMMCHECK(createPersistentRequest(
      comm,
      stream,
      recvbuff,
      /*maxRecvCount=*/count * nRanks,
      datatype,
      &request,
      // Need wait for IPC exchange before capture CE copy
      /*waitForInit=*/true,
      // skipCtrlMsg is a user-controlled hint, not something the cudagraph path
      // forces on; leave it off so exec uses the standard per-exec ctrl-message
      // IB exchange.
      /*skipCtrlMsg=*/false));
  auto cleanupGuard = folly::makeGuard([request, comm]() {
    comm->unregisterPersistentCleanup(request->cleanup_);
    FB_COMMCHECKIGNORE(destroyPersistentRequest(request));
    delete request;
  });

  // Hand teardown to the captured graph BEFORE capturing the ops, so a capture
  // failure tears the request down here rather than leaving NVL imports /
  // registration the graph still references. The callback runs the shared
  // cleanup token (call_once: no-ops if the comm drain already ran it) and
  // deletes the graph-owned request. It must NOT touch the comm registry here
  // (may run after the comm is gone); the spent token just lingers until
  // CtranComm::destroy() drains it -- bounded, O(1) for capture-once/replay.
  auto destroyCb = [](void* p) {
    auto* req = static_cast<CtranPersistentRequest*>(p);
    DCHECK(req->cleanup_ != nullptr);
    req->cleanup_->run();
    delete req;
  };
  FB_COMMCHECK(registerGraphDestroyCallback(stream, request, destroyCb));
  cleanupGuard.dismiss();

  auto* pAlgo = reinterpret_cast<ctran::alltoallp::AlgoImpl*>(request->algo);
  FB_COMMCHECK(pAlgo->exec(sendbuff, count));

  return commSuccess;
}
