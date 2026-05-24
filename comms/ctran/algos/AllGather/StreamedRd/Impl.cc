// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <deque>

#include <folly/ScopeGuard.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllGather/StreamedRd/Plan.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/utils/DevUtils.cuh"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

using ctran::algos::IPersistPlan;
using ctran::algos::PersistPlanKey;
using ctran::allgather::ctsrd::createPersistPlan;
using ctran::allgather::ctsrd::PersistPlan;
using ctran::allgather::ctsrd::Plan;

const auto myAlgo = NCCL_ALLGATHER_ALGO::ctsrd;

// Resolve the NCCL_CTRAN_ALLGATHER_CTSRD_FWD_PEERS cvar to a concrete forward
// depth for the given nSteps. -1 (default) and any value >= nSteps are clamped
// to nSteps, which is the maximally-streamed configuration (every received
// chunk is forwarded immediately to all future-step peers).
int resolveFwdPeers(int nSteps) {
  const int v = NCCL_CTRAN_ALLGATHER_CTSRD_FWD_PEERS;
  if (v < 0 || v >= nSteps) {
    return nSteps;
  }
  return v;
}

struct PutQElem {
  int step;
  int chunkOffset;
};

struct AlgoContext {
  CtranMapper* mapper;
  void* recvbuff;
  void* memHdl;
  const size_t sendSize;
  const Plan& recvPlan;
  const Plan& sendPlan;

  // Per-step ctrl exchange state
  std::vector<void*> remoteRecvBuffs;
  std::vector<CtranMapperRemoteAccessKey> remoteAccessKeys;
  std::vector<CtranMapperRequest> irecvReqs;
  std::vector<CtranMapperRequest> isendReqs;
  // Per-chunk notification, indexed by chunk offset in recvbuff
  std::vector<CtranMapperNotify> notifyVec;

  std::deque<PutQElem> putQ;
  // One tracked request per peer (= per step), indexed by step.
  // Only the last iput to each peer uses a tracked request; earlier iputs
  // pass nullptr (fire-and-forget, safe due to in-order completion guarantee).
  std::vector<CtranMapperRequest> iputReqs;

  // Per-step deferred puts flushed at step start. Always sized to nSteps.
  // For fwdPeers >= nSteps this stays empty and postDeferredPuts is a no-op.
  std::vector<std::vector<PutQElem>> deferredPuts;
  // Per-step count of received notifications
  std::vector<int> numReceived;

  // Per-step count of enqueued puts, for plan order verification.
  std::vector<int> putCount;

  void enqueuePut(int step, int chunkOffset) {
    const auto nth = putCount.at(step)++;
    const auto expChunkOffset = sendPlan.chunk(step, nth);
    FB_CHECKABORT(
        expChunkOffset == chunkOffset,
        "enqueuePut: step {} put #{} expected chunk {} got {}",
        step,
        nth,
        expChunkOffset,
        chunkOffset);
    putQ.push_back({step, chunkOffset});
  }

  AlgoContext(
      CtranMapper* mapper,
      void* recvbuff,
      size_t sendSize,
      size_t nRanks,
      const Plan& recvPlan,
      const Plan& sendPlan)
      : mapper(mapper),
        recvbuff(recvbuff),
        memHdl(nullptr),
        sendSize(sendSize),
        recvPlan(recvPlan),
        sendPlan(sendPlan),
        remoteRecvBuffs(recvPlan.nSteps()),
        remoteAccessKeys(recvPlan.nSteps()),
        irecvReqs(recvPlan.nSteps()),
        isendReqs(recvPlan.nSteps()),
        notifyVec(nRanks),
        iputReqs(recvPlan.nSteps()),
        deferredPuts(recvPlan.nSteps()),
        numReceived(recvPlan.nSteps(), 0),
        putCount(recvPlan.nSteps(), 0) {}
};

// Enqueue or defer a chunk received at step `recvStep` based on fwdPeers.
// Chunks with target step in [recvStep+1, recvStep+fwdPeers] are enqueued
// immediately; chunks with target step in [recvStep+fwdPeers+1, nSteps-1]
// are pushed onto deferredPuts and flushed at the start of the target step.
inline void
fwdChunk(AlgoContext& ctx, int recvStep, int chunkOffset, int fwdPeers) {
  const auto nSteps = ctx.recvPlan.nSteps();
  const int immEnd = std::min(recvStep + 1 + fwdPeers, nSteps);
  for (int fwd = recvStep + 1; fwd < immEnd; fwd++) {
    ctx.enqueuePut(fwd, chunkOffset);
  }
  for (int fwd = immEnd; fwd < nSteps; fwd++) {
    ctx.deferredPuts.at(fwd).push_back({fwd, chunkOffset});
  }
}

// Receive chunks for the current step and post forward puts for future steps.
inline commResult_t
progressRecvFwd(AlgoContext& ctx, int step, int expNumRecv, int fwdPeers) {
  while (ctx.numReceived.at(step) < expNumRecv) {
    const auto cid = ctx.numReceived.at(step);
    const auto chunkOffset = ctx.recvPlan.chunk(step, cid);
    auto rcvd = false;
    FB_COMMCHECK(
        ctx.mapper->checkNotify(&ctx.notifyVec.at(chunkOffset), &rcvd));
    if (!rcvd) {
      break;
    }
    ctx.numReceived.at(step)++;
    fwdChunk(ctx, step, chunkOffset, fwdPeers);
  }
  return commSuccess;
}

// Wait for all tracked iput requests (one per peer/step).
inline commResult_t waitAllPuts(AlgoContext& ctx) {
  const auto nSteps = ctx.recvPlan.nSteps();
  for (int step = 0; step < nSteps; step++) {
    FB_COMMCHECK(ctx.mapper->waitRequest(&ctx.iputReqs.at(step)));
  }
  return commSuccess;
}

// Issue ready puts from the queue. Only the last iput per peer is tracked.
inline commResult_t progressPuts(AlgoContext& ctx) {
  while (!ctx.putQ.empty()) {
    const auto& elem = ctx.putQ.front();
    CtranMapperRequest* req = nullptr;
    if (ctx.sendPlan.isLastChunk(elem.step, elem.chunkOffset)) {
      req = &ctx.iputReqs.at(elem.step);
    }
    FB_COMMCHECK(ctx.mapper->iput(
        reinterpret_cast<char*>(ctx.recvbuff) + elem.chunkOffset * ctx.sendSize,
        reinterpret_cast<char*>(ctx.remoteRecvBuffs.at(elem.step)) +
            elem.chunkOffset * ctx.sendSize,
        ctx.sendSize,
        ctx.recvPlan.peer(elem.step),
        CtranMapperConfig{
            .memHdl_ = ctx.memHdl,
            .remoteAccessKey_ = ctx.remoteAccessKeys.at(elem.step),
            .notify_ = true},
        req));
    ctx.putQ.pop_front();
  }
  return commSuccess;
}

// Exchange ctrl messages and initialize per-chunk notifications for all steps.
inline commResult_t exchangeCtrl(AlgoContext& ctx) {
  const auto nSteps = ctx.recvPlan.nSteps();
  for (int step = 0; step < nSteps; step++) {
    const auto peer = ctx.recvPlan.peer(step);
    FB_COMMCHECK(ctx.mapper->irecvCtrl(
        &ctx.remoteRecvBuffs.at(step),
        &ctx.remoteAccessKeys.at(step),
        peer,
        &ctx.irecvReqs.at(step)));
    FB_COMMCHECK(ctx.mapper->isendCtrl(
        ctx.recvbuff, ctx.memHdl, peer, &ctx.isendReqs.at(step)));
    for (const auto chunkOffset : ctx.recvPlan.chunks(step)) {
      FB_COMMCHECK(ctx.mapper->initNotify(
          peer, ctx.memHdl, &ctx.notifyVec.at(chunkOffset)));
    }
  }
  for (int step = 0; step < nSteps; step++) {
    FB_COMMCHECK(ctx.mapper->waitRequest(&ctx.irecvReqs.at(step)));
  }
  return commSuccess;
}

// Flush pending fwds deferred from earlier steps into putQ.
inline void postDeferredPuts(AlgoContext& ctx, int step) {
  for (auto& fwd : ctx.deferredPuts.at(step)) {
    ctx.enqueuePut(fwd.step, fwd.chunkOffset);
  }
}

inline commResult_t waitCtrl(AlgoContext& ctx) {
  const auto nSteps = ctx.recvPlan.nSteps();
  for (int step = 0; step < nSteps; step++) {
    FB_COMMCHECK(ctx.mapper->waitRequest(&ctx.isendReqs.at(step)));
  }
  return commSuccess;
}

commResult_t impl(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  auto* op = opGroup.front().get();
  const auto sendSize =
      op->allgather.sendcount * commTypeSize(op->allgather.datatype);
  auto* comm = opGroup.front()->comm_;
  const auto& statex = comm->statex_;
  const auto rank = statex->rank();
  const auto nRanks = statex->nRanks();
  auto* mapper = comm->ctran_->mapper.get();

  CtranAlgoLogger logger(allGatherAlgoName(myAlgo), op->opCount, comm);

  auto* profiler = comm->ctran_->profiler.get();
  if (profiler) {
    profiler->initForEachColl(
        op->opCount, NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT);
  }

  CTRAN_PROFILER_IF(profiler, {
    auto& algoContext = profiler->algoContext;
    algoContext.algorithmName = allGatherAlgoName(myAlgo);
    algoContext.sendContext.messageSizes = std::to_string(sendSize);
    algoContext.recvContext.messageSizes = std::to_string(sendSize * nRanks);
  });

  auto profileGuard = folly::makeGuard([&]() {
    CTRAN_PROFILER_IF(profiler, { profiler->reportToScuba(); });
    mapper->reportProfiling();
  });

  CtranMapperContext mapperCtx(
      allGatherAlgoName(myAlgo), sendSize, sendSize * nRanks);
  mapper->setContext(std::move(mapperCtx));

  auto* persistPlan = static_cast<const PersistPlan*>(
      comm->ctran_->algo->getOrCreatePersistPlan(
          PersistPlanKey::kAllgatherCtsrd, [&]() {
            return std::make_unique<PersistPlan>(createPersistPlan(
                rank, nRanks, resolveFwdPeers(ctran::utils::log2i(nRanks))));
          }));
  const auto& recvPlan = persistPlan->recvPlan();
  const auto& sendPlan = persistPlan->sendPlan();
  // Short-circuit for single rank; GPE should not have been invoked in this
  // case, single-rank out-of-place is handled by user thread via cudaMemcpy
  if (recvPlan.nSteps() == 0) {
    return commSuccess;
  }
  const auto nSteps = recvPlan.nSteps();
  const auto fwdPeers = recvPlan.fwdPeers();
  AlgoContext ctx(
      mapper,
      reinterpret_cast<void*>(op->allgather.recvbuff),
      sendSize,
      nRanks,
      recvPlan,
      sendPlan);

  // Register recv buffer
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));
  auto localMemReg = false;
  FB_COMMCHECK(mapper->searchRegHandle(
      ctx.recvbuff, nRanks * sendSize, &ctx.memHdl, &localMemReg));
  auto memRegGuard = folly::makeGuard([&]() {
    if (localMemReg) {
      mapper->deregDynamic(ctx.memHdl);
    }
  });
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL));
  FB_COMMCHECK(exchangeCtrl(ctx));
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL));

  // Data transfer: step-ordered event loop
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA));

  // Pre-enqueue own chunk for EVERY step at startup, regardless of fwdPeers.
  // Own has no network dependency — we want it on the wire as early as
  // possible so the receiver gets our data immediately.
  for (int step = 0; step < nSteps; step++) {
    ctx.enqueuePut(step, rank);
  }

  for (int step = 0; step < nSteps; step++) {
    postDeferredPuts(ctx, step);
    const int expNumRecv = static_cast<int>(ctx.recvPlan.chunks(step).size());
    while (ctx.numReceived.at(step) < expNumRecv) {
      FB_COMMCHECK(progressRecvFwd(ctx, step, expNumRecv, fwdPeers));
      FB_COMMCHECK(progressPuts(ctx));
    }
  }

  FB_COMMCHECK(waitAllPuts(ctx));

  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  FB_COMMCHECK(waitCtrl(ctx));

  return commSuccess;
}

} // namespace

commResult_t ctranAllGatherStreamedRd(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  auto* algo = comm->ctran_->algo.get();
  // Get or create the persist plan to drive the algorithm. It is persisted for
  // the lifetime of the communicator since it is defined purely based on nRanks
  // and the (immutable) fwdPeers cvar.
  algo->getOrCreatePersistPlan(PersistPlanKey::kAllgatherCtsrd, [&]() {
    const auto rank = comm->statex_->rank();
    const auto nRanks = comm->statex_->nRanks();
    return std::make_unique<PersistPlan>(createPersistPlan(
        rank, nRanks, resolveFwdPeers(ctran::utils::log2i(nRanks))));
  });

  CTRAN_COLL_INFO(
      allGatherAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      -1,
      comm,
      stream);
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHER,
      stream,
      allGatherAlgoName(myAlgo),
      comm->ctran_->getOpCount());
  void* extraCopyBuff = nullptr;
  FB_COMMCHECK(prepareAllGatherArgs(
      opGroup,
      config,
      &extraCopyBuff,
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      comm,
      stream));
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      impl,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherCtranGpeStub)));
  if (extraCopyBuff != nullptr) {
    FB_CUDACHECK(cudaMemcpyAsync(
        recvbuff,
        extraCopyBuff,
        sendcount * commTypeSize(datatype) * comm->statex_->nRanks(),
        cudaMemcpyDefault,
        stream));
  }
  return commSuccess;
}
