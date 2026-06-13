// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <memory>
#include <vector>

#include "comms/ctran/algos/AllGather/StreamedRd/Plan.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::allgather::ctsrd::common {

// Resolve the NCCL_CTRAN_ALLGATHER_CTSRD_FWD_PEERS cvar to a concrete forward
// depth for the given nSteps. -1 (default) and any value >= nSteps are clamped
// to nSteps, which is the maximally-streamed configuration (every received
// chunk is forwarded immediately to all future-step peers).
inline int resolveFwdPeers(int nSteps) {
  const int fwdPeers = NCCL_CTRAN_ALLGATHER_CTSRD_FWD_PEERS;
  if (fwdPeers < 0 || fwdPeers >= nSteps) {
    return nSteps;
  }
  return fwdPeers;
}

struct PutQElem {
  int step;
  int chunk;
  CtranMapperRequest* flushReq;
};

} // namespace ctran::allgather::ctsrd::common

namespace ctran::allgather::ctsrd {

struct AlgoContext {
  CtranMapper* mapper;
  void* recvbuff;
  void* memHdl;
  const size_t sendSize;
  const Plan& recvPlan;
  const Plan& sendPlan;

  // Per-step ctrl exchange state.
  std::vector<void*> remoteRecvBuffs;
  std::vector<CtranMapperRemoteAccessKey> remoteAccessKeys;
  std::vector<CtranMapperRequest> irecvReqs;
  std::vector<CtranMapperRequest> isendReqs;
  // Per-chunk notification, indexed by chunk offset in recvbuff.
  std::vector<CtranMapperNotify> notifyVec;

  std::deque<common::PutQElem> putQ;
  // One tracked request per peer (= per step), indexed by step.
  // Only the last iput to each peer uses a tracked request; earlier iputs
  // pass nullptr (fire-and-forget, safe due to in-order completion guarantee).
  std::vector<CtranMapperRequest> iputReqs;
  // Per-step deferred puts flushed at step start.
  std::vector<std::vector<common::PutQElem>> deferredPuts;
  // Per-step iflush requests referenced by putQ/deferredPuts. progressPuts()
  // waits for the relevant flush before issuing a dependent iput.
  std::vector<std::vector<std::unique_ptr<CtranMapperRequest>>> recvFlushReqs;
  // Per-step count of received notifications.
  std::vector<int> numReceived;

  // Per-step count of enqueued puts, for plan order verification.
  std::vector<int> putCount;

  AlgoContext(
      CtranMapper* mapper,
      void* recvbuff,
      void* memHdl,
      size_t sendSize,
      size_t notifyVecSize,
      const Plan& recvPlan,
      const Plan& sendPlan)
      : mapper(mapper),
        recvbuff(recvbuff),
        memHdl(memHdl),
        sendSize(sendSize),
        recvPlan(recvPlan),
        sendPlan(sendPlan),
        remoteRecvBuffs(recvPlan.nSteps()),
        remoteAccessKeys(recvPlan.nSteps()),
        irecvReqs(recvPlan.nSteps()),
        isendReqs(recvPlan.nSteps()),
        notifyVec(notifyVecSize),
        iputReqs(recvPlan.nSteps()),
        deferredPuts(recvPlan.nSteps()),
        recvFlushReqs(recvPlan.nSteps()),
        numReceived(recvPlan.nSteps(), 0),
        putCount(recvPlan.nSteps(), 0) {}

  AlgoContext(
      CtranMapper* mapper,
      void* recvbuff,
      size_t sendSize,
      size_t notifyVecSize,
      const Plan& recvPlan,
      const Plan& sendPlan)
      : AlgoContext(
            mapper,
            recvbuff,
            nullptr,
            sendSize,
            notifyVecSize,
            recvPlan,
            sendPlan) {}
};

} // namespace ctran::allgather::ctsrd

namespace ctran::allgather::ctsrd::common {

// Exchange ctrl messages and initialize per-chunk notifications for all steps.
template <typename AlgoContext>
inline commResult_t exchangeCtrl(AlgoContext& ctx) {
  const auto nSteps = ctx.recvPlan.nSteps();
  for (int step = 0; step < nSteps; step++) {
    const auto peer = ctx.peer(step);
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

template <typename AlgoContext>
inline commResult_t waitCtrl(AlgoContext& ctx) {
  const auto nSteps = ctx.recvPlan.nSteps();
  for (int step = 0; step < nSteps; step++) {
    FB_COMMCHECK(ctx.mapper->waitRequest(&ctx.isendReqs.at(step)));
  }
  return commSuccess;
}

template <typename AlgoContext>
inline commResult_t
postRecvFlush(AlgoContext& ctx, int step, CtranMapperRequest** flushReq) {
  *flushReq = nullptr;
  // Preserve the pre-flush scheduling path on platforms where local flush is a
  // no-op; do not add a completed request that gates the put queue.
  if (!ctx.mapper->isLocalFlushEnabled()) {
    return commSuccess;
  }

  CtranMapperRequest* req = nullptr;
  FB_COMMCHECK(ctx.mapper->iflush(ctx.recvbuff, ctx.memHdl, &req));
  if (req != nullptr) {
    auto reqOwner = std::unique_ptr<CtranMapperRequest>(req);
    *flushReq = reqOwner.get();
    ctx.recvFlushReqs.at(step).push_back(std::move(reqOwner));
  }
  return commSuccess;
}

// Receive chunks for the current step, post one iflush for the ready batch, and
// queue future-step forwards that will poll that flush before issuing iput.
template <typename AlgoContext>
inline commResult_t progressRecvFwd(AlgoContext& ctx, int step) {
  const auto nSteps = ctx.recvPlan.nSteps();
  const auto expNumRecv = static_cast<int>(ctx.recvPlan.chunks(step).size());
  std::vector<int> readyChunks;
  readyChunks.reserve(static_cast<std::size_t>(expNumRecv));
  while (ctx.numReceived.at(step) < expNumRecv) {
    const auto recvIdx = ctx.numReceived.at(step);
    const auto chunk = ctx.recvPlan.chunk(step, recvIdx);
    auto received = false;
    FB_COMMCHECK(ctx.mapper->checkNotify(&ctx.notifyVec.at(chunk), &received));
    if (!received) {
      break;
    }
    ctx.numReceived.at(step)++;
    readyChunks.push_back(chunk);
  }

  if (readyChunks.empty()) {
    return commSuccess;
  }

  CtranMapperRequest* flushReq = nullptr;
  FB_COMMCHECK(postRecvFlush(ctx, step, &flushReq));

  for (const auto chunk : readyChunks) {
    const int immEnd = std::min(step + 1 + ctx.recvPlan.fwdPeers(), nSteps);
    for (int fwd = step + 1; fwd < immEnd; fwd++) {
      ctx.enqueuePut(fwd, chunk, flushReq);
    }
    for (int fwd = immEnd; fwd < nSteps; fwd++) {
      ctx.deferredPuts.at(fwd).push_back({fwd, chunk, flushReq});
    }
  }
  return commSuccess;
}

template <typename AlgoContext>
inline commResult_t
issuePut(AlgoContext& ctx, const PutQElem& elem, CtranMapperRequest* req) {
  const auto peer = ctx.peer(elem.step);
  const auto offset = ctx.chunkByteOffset(elem.chunk);
  FB_COMMCHECK(ctx.mapper->iput(
      reinterpret_cast<char*>(ctx.recvbuff) + offset,
      reinterpret_cast<char*>(ctx.remoteRecvBuffs.at(elem.step)) + offset,
      ctx.sendSize,
      peer,
      CtranMapperConfig{
          .memHdl_ = ctx.memHdl,
          .remoteAccessKey_ = ctx.remoteAccessKeys.at(elem.step),
          .notify_ = true},
      req));
  return commSuccess;
}

// Issue ready puts from the queue. Only the last iput per peer is tracked.
template <typename AlgoContext>
inline commResult_t progressPuts(AlgoContext& ctx) {
  while (!ctx.putQ.empty()) {
    const auto elem = ctx.putQ.front();
    if (elem.flushReq != nullptr) {
      bool flushComplete = false;
      FB_COMMCHECK(ctx.mapper->testRequest(elem.flushReq, &flushComplete));
      if (!flushComplete) {
        return commSuccess;
      }
    }

    CtranMapperRequest* req = nullptr;
    if (ctx.sendPlan.isLastChunk(elem.step, elem.chunk)) {
      req = &ctx.iputReqs.at(elem.step);
    }
    FB_COMMCHECK(issuePut(ctx, elem, req));
    ctx.putQ.pop_front();
  }
  return commSuccess;
}

// Flush pending fwds deferred from earlier steps into putQ.
template <typename AlgoContext>
inline void postDeferredPuts(AlgoContext& ctx, int step) {
  for (const auto& fwd : ctx.deferredPuts.at(step)) {
    ctx.enqueuePut(fwd.step, fwd.chunk, fwd.flushReq);
  }
}

template <typename AlgoContext>
inline commResult_t waitStepFlushes(AlgoContext& ctx, int step) {
  for (const auto& req : ctx.recvFlushReqs.at(step)) {
    FB_COMMCHECK(ctx.mapper->waitRequest(req.get()));
  }
  return commSuccess;
}

template <typename AlgoContext>
inline commResult_t waitAllFlushes(AlgoContext& ctx) {
  const auto nSteps = ctx.recvPlan.nSteps();
  for (int step = 0; step < nSteps; step++) {
    FB_COMMCHECK(waitStepFlushes(ctx, step));
  }
  return commSuccess;
}

// Wait for all tracked iput requests (one per peer/step).
template <typename AlgoContext>
inline commResult_t waitAllPuts(AlgoContext& ctx) {
  const auto nSteps = ctx.recvPlan.nSteps();
  for (int step = 0; step < nSteps; step++) {
    FB_COMMCHECK(ctx.mapper->waitRequest(&ctx.iputReqs.at(step)));
  }
  return commSuccess;
}

template <typename AlgoContext>
inline commResult_t progressSteps(AlgoContext& ctx, int localChunk) {
  const auto nSteps = ctx.recvPlan.nSteps();
  for (int step = 0; step < nSteps; step++) {
    ctx.enqueuePut(step, localChunk);
  }

  for (int step = 0; step < nSteps; step++) {
    postDeferredPuts(ctx, step);
    const auto expNumRecv = static_cast<int>(ctx.recvPlan.chunks(step).size());
    while (ctx.numReceived.at(step) < expNumRecv) {
      FB_COMMCHECK(progressRecvFwd(ctx, step));
      FB_COMMCHECK(progressPuts(ctx));
    }
    FB_COMMCHECK(ctx.onStepComplete(step));
  }

  // Flush-gated puts may still be queued after all recv steps are done.
  while (!ctx.putQ.empty()) {
    FB_COMMCHECK(progressPuts(ctx));
  }
  // Make all received NIC writes GPU-visible before subsequent GPU work.
  FB_COMMCHECK(waitAllFlushes(ctx));
  FB_COMMCHECK(waitAllPuts(ctx));
  return commSuccess;
}

template <typename AlgoContext>
inline commResult_t runExchangeAndProgress(
    AlgoContext& ctx,
    int localChunk,
    ctran::Profiler* profiler) {
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL));
  FB_COMMCHECK(exchangeCtrl(ctx));
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL));

  // Data transfer: step-ordered event loop.
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA));
  FB_COMMCHECK(progressSteps(ctx, localChunk));
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  FB_COMMCHECK(waitCtrl(ctx));
  return commSuccess;
}

} // namespace ctran::allgather::ctsrd::common
