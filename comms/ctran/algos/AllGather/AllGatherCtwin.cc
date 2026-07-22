// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <functional>
#include <memory>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/PersistentCleanup.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/MathUtils.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::destroyPersistentRequest;
using ctran::allgatherp::InitState;

namespace {

// Auto-select the persistent AGP variant for the nLocalRanks>1 case by topology
// (mirrors ctgraph's selectCtgraphAlgo). Called only for nLocalRanks>1; the
// nLocalRanks==1 case routes to the dedicated ring/ctsrd instead (see
// ctranAllGatherCtwin). ctwin cannot use the NCCL_ALLGATHER_P_ALGO cvar because
// multiple comms with different topologies may enable it.
enum NCCL_ALLGATHER_P_ALGO chooseAgpAlgo(
    size_t sendBytes,
    const ncclx::CommStateX* statex) {
  const bool largeMessage = sendBytes >= NCCL_CTGRAPH_ALLGATHER_RING_THRESHOLD;
  return (!largeMessage && ctran::utils::isPowerOfTwo(statex->nNodes()))
      ? NCCL_ALLGATHER_P_ALGO::ctsrdpipeline
      : NCCL_ALLGATHER_P_ALGO::ctpipeline;
}

// Build a persistent AllGatherP request that BORROWS the window's registration
// + NVL/IPC state instead of running AGP's own IPC exchange. The window owns
// and frees that state; this request's teardown only releases the AGP pooled
// pipeSync.
commResult_t createPersistentRequestFromWindow(
    CtranComm* comm,
    cudaStream_t stream,
    ctran::CtranWin* win,
    void* recvbuff,
    const size_t sendcount,
    const commDataType_t datatype,
    const size_t offset,
    const enum NCCL_ALLGATHER_P_ALGO agpVariant,
    CtranPersistentRequest** out) {
  *out = nullptr;
  const auto nRanks = comm->statex_->nRanks();
  const size_t maxRecvCount = sendcount * nRanks;

  // initResources() allocates the pooled GpeKernelSync (pipeSync) used by exec.
  auto algo = std::make_unique<AlgoImpl>(comm, stream);
  FB_COMMCHECK(algo->initResources());

  auto& pArgs = algo->pArgs;
  pArgs.recvbuff = recvbuff;
  pArgs.maxRecvCount = maxRecvCount;
  pArgs.datatype = datatype;
  // Borrow the window's local data registration; leave recvRegHdl_ null.
  pArgs.recvHdl = win->dataRegHdl;
  pArgs.recvRegHdl_ = nullptr;

  // Borrow peer recv buffers + access keys from symmetric window.
  // remoteIpcRegHdls_ stays empty.
  pArgs.remoteRecvBuffs.assign(nRanks, nullptr);
  pArgs.remoteAccessKeys.clear();
  pArgs.remoteAccessKeys.reserve(nRanks);
  for (int r = 0; r < nRanks; r++) {
    const auto& info = win->remWinInfo[r];
    if (info.dataAddr != nullptr) {
      pArgs.remoteRecvBuffs[r] = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(info.dataAddr) + offset);
    }
    pArgs.remoteAccessKeys.push_back(info.dataRkey);
  }

  // NVL/IPC already exchanged
  pArgs.initState = InitState::kInitialized;
  // An ipc_only window has not exchanged inter-node IB rkeys yet, so let the
  // first exec do it (as AGP does). A full-exchange window already carries IB
  // rkeys in remWinInfo, so skip that first-exec exchange.
  pArgs.ibKeysExchanged = !win->isIpcOnly();

  pArgs.algo = agpVariant;

  // NVL CE-multicast: if the window's data registration carries a multicast
  // overlay (set up in CtranWin::exchange when win_register_multicast is on and
  // the NVL group supports it), cache recvbuff's multicast write base so
  // nvlCeBcast fans out via a single NVSwitch write instead of N-1 per-peer
  // unicast copies. std::nullopt (unicast) when there is no overlay.
  pArgs.mcWrite =
      comm->ctran_->mapper->multicastWriteBase(win->dataRegHdl, recvbuff);

  auto request = std::make_unique<CtranPersistentRequest>(
      CtranPersistentRequest::Type::ALLGATHER_P, comm, stream);
  request->algo = algo.release();
  // Return request ownership to window.
  *out = request.release();

  // Route teardown through a one-shot cleanup token (see PersistentCleanup.h):
  // it only resets AGP-owned resources (pooled pipeSync, empty
  // remoteIpcRegHdls_, null recvRegHdl_); the borrowed window state is released
  // later by the window's free().
  auto cleanup = std::make_shared<PersistentCleanup>([request = *out]() {
    FB_COMMCHECKIGNORE(destroyPersistentRequest(request));
  });
  (*out)->cleanup_ = cleanup;
  // The token can fire from TWO sites: (a) the window's free() -- the primary
  // path, which runs the token, unregisters it, and deletes the request; and
  // (b) the comm's drainPersistentCleanups() at comm destroy -- a safety net if
  // the window outlives its free(), so the pooled pipeSync is returned before
  // CtranGpe::terminate()'s pool drain. The token is idempotent (call_once), so
  // there is no double cleanup.
  comm->registerPersistentCleanup(cleanup);
  return commSuccess;
}

} // namespace

bool checkCtranAllGatherCtwinSupport(
    CtranComm* comm,
    const void* recvbuff,
    const size_t recvBytes,
    const enum NCCL_ALLGATHER_ALGO algo,
    ctran::CtranWin** winOut) {
  if (winOut != nullptr) {
    *winOut = nullptr;
  }
  // Dormant until a caller passes a non-empty recvbuff.
  if (recvbuff == nullptr || recvBytes == 0) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "AllGather {} unsupported: needs a non-empty recvbuff (recvbuff={}, recvBytes={}).",
        allGatherAlgoName(algo),
        recvbuff,
        recvBytes);
    return false;
  }
  // Need a symmetric AllGatherP window covering the full recv range (exec
  // computes each peer's recvbuf as its window buffer at the same offset).
  auto* win = comm->findWindowForBuffer(recvbuff, recvBytes);
  if (win == nullptr || !win->isSymmetric() || !win->allGatherPSupported()) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "AllGather {} unsupported: recvbuff {} ({} bytes) is not covered by a symmetric AllGatherP window.",
        allGatherAlgoName(algo),
        recvbuff,
        recvBytes);
    return false;
  }
  // Forced dedicated variants need nLocalRanks==1; forced recursive-doubling
  // variants need power-of-2 topology. Plain `ctwin` auto-selects a valid
  // variant for any topology.
  const auto* statex = comm->statex_.get();
  if ((algo == NCCL_ALLGATHER_ALGO::ctwin_ring ||
       algo == NCCL_ALLGATHER_ALGO::ctwin_srd) &&
      statex->nLocalRanks() != 1) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "AllGather {} unsupported: forced dedicated ring/streamed-RD requires nLocalRanks==1, got {}.",
        allGatherAlgoName(algo),
        statex->nLocalRanks());
    return false;
  }
  if (algo == NCCL_ALLGATHER_ALGO::ctwin_srd &&
      !ctran::utils::isPowerOfTwo(statex->nRanks())) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "AllGather {} unsupported: forced streamed-RD requires power-of-2 nRanks, got {}.",
        allGatherAlgoName(algo),
        statex->nRanks());
    return false;
  }
  if (algo == NCCL_ALLGATHER_ALGO::ctwin_rdpipeline &&
      !ctran::utils::isPowerOfTwo(statex->nNodes())) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "AllGather {} unsupported: forced rdpipeline requires power-of-2 nNodes, got {}.",
        allGatherAlgoName(algo),
        statex->nNodes());
    return false;
  }
  if (winOut != nullptr) {
    *winOut = win;
  }
  return true;
}

commResult_t ctranAllGatherCtwin(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLGATHER_ALGO algo) {
  // The persistent request is WINDOW-owned (cached in persistentReqs_, freed at
  // window free()).
  // LIFETIME CONTRACT: the window (and its cached request / pooled pipeSync)
  // MUST outlive any CUDA graph that captured a ctwin allgather over it.
  const auto nRanks = comm->statex_->nRanks();
  const size_t typeSize = commTypeSize(datatype);
  const size_t recvBytes = sendcount * nRanks * typeSize;

  ctran::CtranWin* win = nullptr;
  if (!checkCtranAllGatherCtwinSupport(comm, recvbuff, recvBytes, algo, &win)) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "AllGather {}: unsupported for recvbuff {} (len {} bytes) -- needs a "
        "symmetric AllGatherP window and a compatible topology.",
        allGatherAlgoName(algo),
        recvbuff,
        recvBytes);
  }

  // Route by algo. Plain `ctwin` auto-selects (mirroring ctgraph): at
  // nLocalRanks==1 the dedicated ring/streamed-RD (better optimized than AGP's
  // degenerate nLocalRanks==1 ring; the recvbuf's window registration keeps
  // them capture-safe); at nLocalRanks>1 the persistent AGP path (reuses the
  // window's NVL/IPC state). ctwin_* force a specific algo. Caveat: the
  // dedicated path does a per-replay ctrl/rkey exchange, so the ctgraph
  // capture/replay ordering constraint applies.
  const auto* statex = comm->statex_.get();
  const size_t sendBytes = sendcount * typeSize;
  const bool largeMessage = sendBytes >= NCCL_CTGRAPH_ALLGATHER_RING_THRESHOLD;
  const bool autoNLocal1 =
      algo == NCCL_ALLGATHER_ALGO::ctwin && statex->nLocalRanks() == 1;

  if (algo == NCCL_ALLGATHER_ALGO::ctwin_srd ||
      (autoNLocal1 && !largeMessage &&
       ctran::utils::isPowerOfTwo(statex->nRanks()))) {
    return ctranAllGatherStreamedRd(
        sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }
  if (algo == NCCL_ALLGATHER_ALGO::ctwin_ring || autoNLocal1) {
    // ctwin_ring forces the dedicated ring at any topology; auto @
    // nLocalRanks==1 falls here for large or non-power-of-2 (ring is the
    // general fallback).
    return ctranAllGatherRing(
        sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }

  // Persistent AGP path: ctwin auto at nLocalRanks>1, or forced ctwin_pipeline
  // / ctwin_rdpipeline.
  enum NCCL_ALLGATHER_P_ALGO agpVariant;
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctwin_pipeline:
      agpVariant = NCCL_ALLGATHER_P_ALGO::ctpipeline;
      break;
    case NCCL_ALLGATHER_ALGO::ctwin_rdpipeline:
      agpVariant = NCCL_ALLGATHER_P_ALGO::ctsrdpipeline;
      break;
    default: // ctwin auto, nLocalRanks>1
      agpVariant = chooseAgpAlgo(sendBytes, statex);
      break;
  }

  const size_t offset = reinterpret_cast<uintptr_t>(recvbuff) -
      reinterpret_cast<uintptr_t>(win->winDataPtr);

  // Log the resolved window id so a rank-divergent window pick (all ranks must
  // resolve a buffer to the same window) is debuggable.
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllGather ctwin: rank {} resolved recvbuff {} ({} bytes, offset {}) to window id {}",
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
            comm,
            stream,
            win,
            recvbuff,
            sendcount,
            datatype,
            offset,
            agpVariant,
            &req);
        return createRes == commSuccess ? req : nullptr;
      });
  FB_COMMCHECK(createRes);
  if (request == nullptr) {
    FB_ERRORRETURN(
        commInternalError,
        "AllGather ctwin: failed to obtain a persistent request for recvbuff {}.",
        recvbuff);
  }

  return ctran::allGatherPExec(sendbuff, sendcount, datatype, request);
}
