// Copyright (c) Meta Platforms, Inc. and affiliates.

// Windowed-capture build/teardown for the ctgraph AllGather path. The
// persistent state that survives every graph replay is split two ways: the
// CtranWin window owns the NVL peer addresses (baked into the captured
// CopyEngine broadcasts), and the persistent request (AlgoImpl) owns the
// pipeSync flag the captured kernels poll. The cross-node rail rkeys are
// exchanged by each algo with only its own rail peers during its per-step sync.

#include <folly/ScopeGuard.h>
#include <memory>
#include <utility>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/CtranCommSplit.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/AllGatherPWithWin.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/CudaRAII.h"

using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::PersistArgs;

#define CHECK_VALID_PREQ(pReq)                                         \
  do {                                                                 \
    if (!(pReq)) {                                                     \
      FB_ERRORRETURN(                                                  \
          commInvalidArgument,                                         \
          "Null PersistentRequest passed to {}",                       \
          __func__);                                                   \
    }                                                                  \
    if (pReq->type != CtranPersistentRequest::Type::ALLGATHER_P_WIN) { \
      FB_ERRORRETURN(                                                  \
          commInvalidArgument,                                         \
          "Unexpected PersistentRequest type {} called into {}",       \
          pReq->type,                                                  \
          __func__);                                                   \
    }                                                                  \
  } while (0)

namespace ctran {

// Internal helpers for the windowed-capture lifecycle. They are file-local: the
// only entry points are createAllGatherPWithWindow /
// destroyAllGatherPWithWindow.
namespace {

// Point the persistent algo at the local-NVL window: copy the window's NVL
// remote addresses into pArgs, keyed by parent-comm rank. Pure host work (no
// mapper exchange), so it runs on the calling thread.
commResult_t attachWindow(CtranPersistentRequest* request, CtranWin* win) {
  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  auto& pArgs = algo->pArgs;
  pArgs.recvbuff = win->winDataPtr;
  pArgs.recvHdl = win->dataRegHdl;

  // The window is always registered on the split local-NVL subcomm, so its comm
  // and parentRanks are present and map window rank -> parent-comm rank.
  FB_CHECKABORT(win->comm != nullptr, "windowed AllGatherP: win->comm is null");
  const auto& winToCommRank = win->comm->parentRanks();
  FB_CHECKABORT(
      winToCommRank.size() == win->remWinInfo.size(),
      "window remWinInfo size {} does not match parentRanks size {}",
      win->remWinInfo.size(),
      winToCommRank.size());

  const auto nRanks = request->comm_->statex_->nRanks();
  pArgs.remoteRecvBuffs.assign(nRanks, nullptr);
  pArgs.remoteAccessKeys.assign(nRanks, CtranMapperRemoteAccessKey{});
  for (size_t w = 0; w < winToCommRank.size(); ++w) {
    pArgs.remoteRecvBuffs[winToCommRank[w]] = win->remWinInfo[w].dataAddr;
    pArgs.remoteAccessKeys[winToCommRank[w]] = win->remWinInfo[w].dataRkey;
  }
  pArgs.initialized.store(true);
  return commSuccess;
}

} // namespace

// Build a persistent windowed AllGatherP: create the algo, split a local-NVL
// subcomm, register the recvbuff as a window on it, and attach its NVL
// addresses. On success `*out` is a persistent request whose AlgoImpl owns the
// window and subcomm (nvlWin / nvlComm), which must outlive every replay --
// tear down with destroyAllGatherPWithWindow at comm destroy. The pipeSync
// alloc runs under the relaxed capture-mode guard.
commResult_t createAllGatherPWithWindow(
    CtranComm* comm,
    void* recvbuff,
    size_t recvBytes,
    cudaStream_t stream,
    CtranPersistentRequest** out) {
  // The window keeps the intra-node NVL broadcast addresses stable across
  // replays; a single local rank has no NVL peers, so it must use
  // ctgraph_ring/rd instead.
  if (comm->statex_->nLocalRanks() <= 1) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "createAllGatherPWithWindow requires nLocalRanks > 1; got {}",
        comm->statex_->nLocalRanks());
  }

  // Create the persistent algo (allocates its pipeSync). No remote exchange
  // here: attachWindow fills the NVL addresses and each algo exchanges its rail
  // peer rkeys with its own peers at exec time.
  CtranPersistentRequest* request = nullptr;
  {
    meta::comms::StreamCaptureModeGuard captureGuard{
        cudaStreamCaptureModeRelaxed};
    auto algo = std::make_unique<AlgoImpl>(comm, stream);
    algo->pArgs.initialized.store(false);
    FB_COMMCHECK(algo->initResources());
    request = new CtranPersistentRequest(
        CtranPersistentRequest::Type::ALLGATHER_P_WIN, comm, stream);
    request->algo = algo.release();
  }
  auto requestGuard =
      folly::makeGuard([&request]() { destroyAllGatherPWithWindow(request); });

  std::shared_ptr<CtranComm> nvlComm;
  FB_COMMCHECK(ctranCommSplitLocalNvl(comm, &nvlComm));

  CtranWin* nvlWin = nullptr;
  FB_COMMCHECK(ctranWinRegister(recvbuff, recvBytes, nvlComm.get(), &nvlWin));
  FB_CHECKABORT(nvlWin != nullptr, "ctranWinRegister returned null window");
  auto winGuard = folly::makeGuard([&nvlWin]() {
    nvlWin->free(true /* skipBarrier */);
    delete nvlWin;
  });

  FB_COMMCHECK(attachWindow(request, nvlWin));

  // attachWindow was the last fallible step; commit, then hand the window and
  // subcomm to the algo, which owns them from here (freed in
  // AlgoImpl::destroy).
  requestGuard.dismiss();
  winGuard.dismiss();
  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  algo->nvlWin = nvlWin;
  algo->nvlComm = std::move(nvlComm);
  *out = request;
  return commSuccess;
}

commResult_t destroyAllGatherPWithWindow(CtranPersistentRequest* request) {
  if (request == nullptr) {
    return commSuccess;
  }
  CHECK_VALID_PREQ(request);
  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  if (algo != nullptr) {
    // AlgoImpl::destroy frees the pipeSync, the window, and the subcomm.
    FB_COMMCHECK(algo->destroy());
    delete algo;
  }
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "destroyAllGatherPWithWindow: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);
  delete request;
  return commSuccess;
}

} // namespace ctran
