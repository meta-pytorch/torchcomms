// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <functional>
#include <memory>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/algos/AllToAll/HostTypes.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/PersistentCleanup.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

using ctran::alltoallp::AlgoImpl;
using ctran::alltoallp::destroyPersistentRequest;
using ctran::alltoallp::InitState;

namespace {

// Build a persistent AllToAllP request that BORROWS the window's registration
// + NVL/IPC state instead of running AllToAllP's own IPC exchange. The window
// owns and frees that state; this request's teardown only releases the
// AllToAllP-owned resources.
commResult_t createPersistentRequestFromWindow(
    CtranComm* comm,
    cudaStream_t stream,
    ctran::CtranWin* win,
    void* recvbuff,
    const size_t count,
    const commDataType_t datatype,
    const size_t offset,
    CtranPersistentRequest** out) {
  *out = nullptr;
  const auto nRanks = comm->statex_->nRanks();
  const size_t maxRecvCount = count * nRanks;

  // AllToAllP's AlgoImpl has no pooled resource init; construct it and fill
  // pArgs directly from the window.
  auto algo = std::make_unique<AlgoImpl>(comm, stream);

  auto& pArgs = algo->pArgs;
  pArgs.recvbuff = recvbuff;
  pArgs.maxRecvCount = maxRecvCount;
  pArgs.datatype = datatype;
  // Borrow the window's local data registration; leave recvRegHdl null so
  // teardown does not release the window-owned registration.
  pArgs.recvHdl = win->dataRegHdl;
  pArgs.skipCtrlMsg = false;

  // Borrow peer recv buffers + access keys from the symmetric window.
  // remoteIpcRegHdls stays empty (the NVL imports are owned by the window).
  pArgs.remoteRecvBuffs.assign(nRanks, nullptr);
  pArgs.remoteAccessKeys.clear();
  pArgs.remoteAccessKeys.reserve(nRanks);
  for (int r = 0; r < nRanks; r++) {
    const auto& info = win->remWinInfo[r];
    if (info.dataAddr != nullptr) {
      // Peer r's window buffer base at the same window offset. exec adds this
      // rank's own recv-slot offset internally (see ctranAllToAllPIbImpl).
      pArgs.remoteRecvBuffs[r] = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(info.dataAddr) + offset);
    }
    pArgs.remoteAccessKeys.push_back(info.dataRkey);
  }

  // NVL/IPC already exchanged by the window.
  pArgs.initState = InitState::kInitialized;
  // An ipc_only window has not exchanged inter-node IB rkeys yet, so let the
  // first exec do it (as AllToAllP does). A full-exchange window already
  // carries IB rkeys in remWinInfo, so skip that first-exec exchange.
  pArgs.ibKeysExchanged = !win->isIpcOnly();

  auto request = std::make_unique<CtranPersistentRequest>(
      CtranPersistentRequest::Type::ALLTOALL_P, comm, stream);
  request->algo = algo.release();
  // Return request ownership to the window.
  *out = request.release();

  // Route teardown through a one-shot cleanup token (see PersistentCleanup.h):
  // it releases only AllToAllP-owned resources; the borrowed window state is
  // released later by the window's free().
  auto cleanup = std::make_shared<PersistentCleanup>([request = *out]() {
    FB_COMMCHECKIGNORE(destroyPersistentRequest(request));
  });
  (*out)->cleanup_ = cleanup;
  // The token can fire from TWO sites: (a) the window's free() -- the primary
  // path, which runs the token, unregisters it, and deletes the request; and
  // (b) the comm's drainPersistentCleanups() at comm destroy -- a safety net if
  // the window outlives its free(). The token is idempotent (call_once), so
  // there is no double cleanup.
  comm->registerPersistentCleanup(cleanup);
  return commSuccess;
}

} // namespace

bool checkCtranAllToAllCtwinSupport(
    CtranComm* comm,
    const void* recvbuff,
    const size_t recvBytes,
    const enum NCCL_ALLTOALL_ALGO algo,
    ctran::CtranWin** winOut) {
  if (winOut != nullptr) {
    *winOut = nullptr;
  }
  // Dormant until a caller passes a non-empty recvbuff.
  if (recvbuff == nullptr || recvBytes == 0) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "AllToAll {} unsupported: needs a non-empty recvbuff (recvbuff={}, recvBytes={}).",
        allToAllAlgoName(algo),
        recvbuff,
        recvBytes);
    return false;
  }
  // Need a symmetric AllToAllP window covering the full recv range (exec
  // computes each peer's recvbuf as its window buffer at the same offset).
  auto* win = comm->findWindowForBuffer(recvbuff, recvBytes);
  if (win == nullptr || !win->isSymmetric() || !ctran::AllToAllPSupport(comm)) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "AllToAll {} unsupported: recvbuff {} ({} bytes) is not covered by a symmetric AllToAllP window.",
        allToAllAlgoName(algo),
        recvbuff,
        recvBytes);
    return false;
  }
  if (winOut != nullptr) {
    *winOut = win;
  }
  return true;
}

commResult_t ctranAllToAllCtwin(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLTOALL_ALGO algo) {
  // The persistent request is WINDOW-owned (cached in persistentReqs_, freed at
  // window free()).
  // LIFETIME CONTRACT: the window (and its cached request) MUST outlive any
  // CUDA graph that captured a ctwin alltoall over it.
  const auto nRanks = comm->statex_->nRanks();
  const size_t typeSize = commTypeSize(datatype);
  const size_t recvBytes = count * nRanks * typeSize;

  ctran::CtranWin* win = nullptr;
  if (!checkCtranAllToAllCtwinSupport(comm, recvbuff, recvBytes, algo, &win)) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "AllToAll {}: unsupported for recvbuff {} (len {} bytes) -- needs a "
        "symmetric AllToAllP window.",
        allToAllAlgoName(algo),
        recvbuff,
        recvBytes);
  }

  const size_t offset = reinterpret_cast<uintptr_t>(recvbuff) -
      reinterpret_cast<uintptr_t>(win->winDataPtr);

  // Log the resolved window id so a rank-divergent window pick (all ranks must
  // resolve a buffer to the same window) is debuggable.
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllToAll ctwin: rank {} resolved recvbuff {} ({} bytes, offset {}) to window id {}",
      comm->statex_->rank(),
      recvbuff,
      recvBytes,
      offset,
      win->id());

  // Reuse (or lazily build) the persistent request cached on the window, keyed
  // by <offset, len, stream> so a request is reused only for sequential
  // collectives with the same stream.
  commResult_t createRes = commSuccess;
  auto* request = win->getOrCreatePersistentRequest(
      offset, recvBytes, stream, [&]() -> CtranPersistentRequest* {
        CtranPersistentRequest* req = nullptr;
        createRes = createPersistentRequestFromWindow(
            comm, stream, win, recvbuff, count, datatype, offset, &req);
        return createRes == commSuccess ? req : nullptr;
      });
  FB_COMMCHECK(createRes);
  if (request == nullptr) {
    FB_ERRORRETURN(
        commInternalError,
        "AllToAll ctwin: failed to obtain a persistent request for recvbuff {}.",
        recvbuff);
  }

  return ctran::AllToAllPExec(sendbuff, count, request);
}
