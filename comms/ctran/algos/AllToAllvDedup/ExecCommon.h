// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_fp16.h>
#include <cstddef>
#include <vector>

#include "comms/ctran/algos/AllToAllvDedup/AlgoImpl.h"
#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/gpe/CtranGpe.h"

namespace ctran::alltoallvdedup {

extern thread_local uint64_t thOpCount;
extern thread_local int thMyRank;

using namespace utils;

#define DEBUG_LOG(fmt, ...)      \
  CLOGF_SUBSYS(                  \
      INFO,                      \
      COLL,                      \
      "rank {} opCount {} " fmt, \
      thMyRank,                  \
      thOpCount,                 \
      ##__VA_ARGS__);

constexpr int kLastTransAckMagic = 991;

inline int lastTransAckValue(const int rank) {
  return kLastTransAckMagic + rank;
}

// Shared structures
struct ExecCtx {
  PersistArgs pArgs;
  ResourceRef* resource;
  PersistConfig* config;
  ncclx::CommStateX* commStatex;
  CtranMapper* mapper;
  utils::TraceRecord* ts;
  uint64_t opCount;
};

struct ProgressState {
  struct StepState {
    int ready{0};
    int posted{0};
    int done{0};
  };

  // track op pre sendTrans, i.e., sendCopy in exec() and fwdRed in combine()
  std::vector<StepState> preSendTrans;
  // cross-node send and recv
  std::vector<StepState> sendTrans;
  std::vector<StepState> recvTrans;
  // track op post recvTrans, i.e., fwd in exec() and sendRed in combine()
  std::vector<StepState> postRecvTrans;
  // track op on intra forward; only for profiling
  std::vector<StepState> intraFwd;
  // track op on receiver rank; only for profiling
  std::vector<StepState> recvCopy;
  // track the completion of remote recvTrans; sender side waits for it before
  // return, so it ensures all remote chunks are available for next opCount
  std::vector<int> lastTransAck;

  std::vector<int> numSendSteps;
  std::vector<int> numRecvSteps;

  int numPendingSendNodes{0};
  int numPendingRecvNodes{0};

  // nNodes, computed in prepare at kTmpNumSendIdxH
  int* numSendBlocks;

  // sender (sender in exec() and forwarder in combine()) transport objects
  std::vector<std::vector<CtranMapperRequest>> sendTransReqs;
  std::vector<std::vector<CtranMapperRequest>> remChkRSyncReqs;

  // receiver (forwarder in combine(), sender in exec()) transport objects
  std::vector<CtranMapperNotify> recvTransNotifies;
  std::vector<std::vector<CtranMapperRequest>> remChkSSyncReqs;

  ProgressState(int nNodes, int nLocalRanks);
};

enum class ProfileEvent {
  kSendCopy,
  kSendRed,
  kFwd,
  kFwdRed,
  kIntraFwd,
  kIntraFwdRed,
  kRecvTrans,
  kRecvCopy,
  kRecvRed,
  kSendTrans,
};

inline std::tuple<const std::string_view, int> getProfileEventLabelSeq(
    const ProfileEvent& event) {
  switch (event) {
    case ProfileEvent::kSendCopy:
      return std::tuple("sendCopy", 100);
    case ProfileEvent::kSendRed:
      return std::tuple("sendRed", 100);
    case ProfileEvent::kSendTrans:
      return std::tuple("sendTrans", 200);
    case ProfileEvent::kIntraFwd:
      return std::tuple("infraFwd", 300);
    case ProfileEvent::kIntraFwdRed:
      return std::tuple("infraFwdRed", 300);
    case ProfileEvent::kRecvTrans:
      return std::tuple("recvTrans", 300);
    case ProfileEvent::kFwd:
      return std::tuple("fwd", 400);
    case ProfileEvent::kFwdRed:
      return std::tuple("fwdRed", 400);
    case ProfileEvent::kRecvCopy:
      return std::tuple("recvCopy", 500);
    case ProfileEvent::kRecvRed:
      return std::tuple("recvRed", 500);
  }
  return std::tuple("unknown", 0);
}

inline void profileStart(
    ExecCtx& ctx,
    const int node,
    const int step,
    const ProfileEvent& event) {
  const auto [label, seq] = getProfileEventLabelSeq(event);
  const std::map<std::string, std::string> metaData = {
      {"peerNode", std::to_string(node)}, {"step", std::to_string(step)}};
  ctx.ts->startInterval(
      fmt::format("{}_{}", label, node),
      step,
      seq + node * NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS +
          (step % NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS),
      metaData);
}

inline void profileEnd(
    ExecCtx& ctx,
    const int node,
    const int step,
    const ProfileEvent& event) {
  const auto [label, seq] = getProfileEventLabelSeq(event);
  ctx.ts->endInterval(fmt::format("{}_{}", label, node), step);
}

inline void profilePoint(
    ExecCtx& ctx,
    const int node,
    const int step,
    const ProfileEvent& event) {
  const auto [label, seq] = getProfileEventLabelSeq(event);
  const std::map<std::string, std::string> metaData = {
      {"peerNode", std::to_string(node)}, {"step", std::to_string(step)}};

  ctx.ts->addPoint(
      fmt::format("{}_{}", label, node),
      step,
      seq + node * NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS +
          (step % NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS),
      metaData);
}

// Helper functions to emplace a request from the container with container space
// check. We need preallocate sufficient space to ensure the memory is not
// changed while used by backend for tracking completion.
inline CtranMapperRequest* getReservedReq(
    std::vector<CtranMapperRequest>& vec,
    const int reserved) {
  auto& req = vec.emplace_back();
  FB_CHECKTHROW(
      vec.size() <= reserved,
      "Emplace {}-th request exceeds reserved {}. It is likely a bug",
      vec.size() - 1,
      reserved);
  return &req;
}

inline void issuePreSendTrans(
    ExecCtx& ctx,
    const int node,
    const int step,
    ProgressState& state,
    const bool isExec) {
  auto& algoRes = ctx.resource;
  auto& preState = state.preSendTrans[node];

  const auto event = isExec ? ProfileEvent::kSendCopy : ProfileEvent::kFwdRed;
  profileStart(ctx, node, step, event);

  auto* gkSync = algoRes->kSync.sendGKSyncs + node;
  gkSync->post(step);
  preState.posted++;
  DEBUG_LOG(
      "Rank {} PreSendTrans posted for node {} step {}, sync {}; update posted={}/{}",
      ctx.commStatex->rank(),
      node,
      step,
      (void*)gkSync,
      preState.posted,
      state.numSendSteps[node]);
}

inline void checkPreSendTransDone(
    ExecCtx& ctx,
    const int node,
    ProgressState& state,
    const bool isExec) {
  auto& algoRes = ctx.resource;

  auto& preState = state.preSendTrans[node];
  auto& transState = state.sendTrans[node];
  auto* gkSync = algoRes->kSync.sendGKSyncs + node;
  for (int step = preState.done; step < preState.posted; step++) {
    if (gkSync->isComplete(step)) {
      preState.done++;

      const auto event =
          isExec ? ProfileEvent::kSendCopy : ProfileEvent::kFwdRed;
      profileEnd(ctx, node, step, event);

      transState.ready++;
      DEBUG_LOG(
          "Rank {} PreSendTrans done for node {} step {}; update done={}, sendTrans.ready={}/{}",
          ctx.commStatex->rank(),
          node,
          step,
          preState.done,
          transState.ready,
          state.numSendSteps[node]);
    }
  }
}

const int kProfileEnd = INT_MAX;
inline void
profileIntraFwd(ExecCtx& ctx, ProgressState& state, const bool isExec) {
  auto& algoRes = ctx.resource;
  const auto myNode = ctx.commStatex->node();
  auto* gkSync = algoRes->kSync.intraFwdGKSyncs;
  auto& intraFwdState = state.intraFwd[0];

  // TODO: we don't yet track per step per local rank time for intraFwd, maybe
  // add?
  const auto startVal = intraFwdState.posted * 2;
  if (gkSync->isComplete(startVal)) {
    profileStart(ctx, myNode, 0, ProfileEvent::kIntraFwd);

    DEBUG_LOG("profileIntraFwd start for step {}", intraFwdState.posted);
    intraFwdState.posted++;
  }

  const auto completeVal = intraFwdState.done * 2 + 1;
  if (gkSync->isComplete(completeVal)) {
    profileEnd(ctx, myNode, 0, ProfileEvent::kIntraFwd);
    DEBUG_LOG("profileIntraFwd end for step {}", intraFwdState.done);
    intraFwdState.done++;
  }
}

inline void profileRecvCopy(
    ExecCtx& ctx,
    ProgressState& state,
    const bool isExec,
    const bool isIntra = false) {
  const auto event = isExec ? ProfileEvent::kRecvCopy : ProfileEvent::kRecvRed;
  auto& algoRes = ctx.resource;
  const auto nLocalRanks = ctx.commStatex->nLocalRanks();

  for (int r = 0; r < nLocalRanks; r++) {
    auto& rCopyState = state.recvCopy[r];
    auto* gkSync = isIntra ? algoRes->kSync.intraRecvCopyGKSyncs + r
                           : algoRes->kSync.recvCopyGKSyncs + r;
    const auto startVal = rCopyState.posted * 2;
    auto step = rCopyState.posted;
    if (gkSync->isComplete(startVal)) {
      profileStart(ctx, r, step, event);

      DEBUG_LOG(
          "{} start for step {}",
          isIntra ? "profileIntraRecvCopy" : "profileRecvCopy",
          step);
      rCopyState.posted++;
    }

    const auto completeVal = rCopyState.done * 2 + 1;
    step = rCopyState.done;
    if (gkSync->isComplete(completeVal)) {
      profileEnd(ctx, r, step, event);
      DEBUG_LOG(
          "{} end for step {}",
          isIntra ? "profileIntraRecvCopy" : "profileRecvCopy",
          step);
      rCopyState.done++;
    }
  }
}

inline void
profileIntraRecvCopy(ExecCtx& ctx, ProgressState& state, const bool isExec) {
  profileRecvCopy(ctx, state, isExec, true);
}

inline commResult_t
prePostSendTransAck(ExecCtx& ctx, ProgressState& state, int node) {
  const auto numSteps = state.numSendSteps[node];
  // skip first NUM_CHUNKS as they are known to be ready
  const auto numRecvs = numSteps > NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS
      ? numSteps - NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS
      : 0;

  auto& statex = ctx.commStatex;
  auto& mapper = ctx.mapper;
  const int myLocalRank = statex->localRank();
  const int railPeerRank = statex->localRankToRank(myLocalRank, node);

  // post recvCtrl for per-step ack
  for (auto step = 0; step < numRecvs; step++) {
    auto* req = getReservedReq(state.remChkRSyncReqs[node], numSteps);
    FB_COMMCHECK(mapper->irecvCtrl(railPeerRank, req));
  }

  // post last ack to ensure all remote chunks are released before moving to
  // next opCount
  auto* req = getReservedReq(state.remChkRSyncReqs[node], numSteps);
  FB_COMMCHECK(mapper->irecvCtrlMsg(
      &state.lastTransAck[node], sizeof(int), railPeerRank, req));
  return commSuccess;
}

inline commResult_t checkSendTransReady(
    ExecCtx& ctx,
    ProgressState& state,
    int node,
    int step,
    bool& ready) {
  if (step < NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS) {
    ready = true;
    return commSuccess; // first step is always ready
  }

  auto& mapper = ctx.mapper;
  const auto reqId = step - NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS;
  auto& req = state.remChkRSyncReqs[node].at(reqId);

  bool done_ = false;
  FB_COMMCHECK(mapper->testRequest(&req, &done_));
  ready = done_;
  return commSuccess;
}

inline commResult_t
issueSendTrans(ExecCtx& ctx, ProgressState& state, int node, int step) {
  auto& statex = ctx.commStatex;
  auto& mapper = ctx.mapper;

  profileStart(ctx, node, step, ProfileEvent::kSendTrans);

  const int myLocalRank = statex->localRank();
  int railPeerRank = statex->localRankToRank(myLocalRank, node);
  const int myNode = statex->node();
  const auto numSteps = state.numSendSteps[node];

  auto& tmpSendBuff = ctx.resource->getBuf(ResourceBufName::kTmpSendBuff);
  auto& tmpRemBuffs = ctx.resource->getRemBufs(ResourceBufName::kTmpFwdBuff);

  void* tmpSendChunk = getTmpChunkPtr(*ctx.config, tmpSendBuff.ptr, step, node);
  void* tmpRemChunk =
      getTmpChunkPtr(*ctx.config, tmpRemBuffs[node].ptr, step, myNode);

  auto* req = getReservedReq(state.sendTransReqs[node], numSteps);

  // each RDMA chunk is always with CHUNK_SIZE except the last one
  auto putLen = NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE;
  if (step == numSteps - 1) {
    const auto numBlocks = state.numSendBlocks[node];
    const auto actualNumBlocks = numBlocks - ctx.pArgs.maxNumStepBlks * step;
    putLen = getTmpChunkActualLen(ctx.pArgs, actualNumBlocks);
  }
  FB_COMMCHECK(mapper->iput(
      tmpSendChunk,
      tmpRemChunk,
      putLen,
      railPeerRank,
      CtranMapperConfig{
          .memHdl_ = tmpSendBuff.regHdl,
          .remoteAccessKey_ = tmpRemBuffs[node].rkey,
          .notify_ = true /*notify*/},
      req));

  state.sendTrans[node].posted++;

  DEBUG_LOG(
      "Rank {} posted SendTrans to node {} (peerRank {}) step {}, tmpSendChunk={} tmpRemBuffs {} tmpRemChunk={}, req={}, updated sendTrans.posted={}",
      ctx.commStatex->rank(),
      node,
      railPeerRank,
      step,
      tmpSendChunk,
      tmpRemBuffs[node].ptr,
      tmpRemChunk,
      (void*)req,
      state.sendTrans[node].posted);
  return commSuccess;
}

inline commResult_t
checkSendTransDone(ExecCtx& ctx, ProgressState& state, int node) {
  auto& preState = state.preSendTrans[node];
  auto& transState = state.sendTrans[node];
  auto& reqs = state.sendTransReqs[node];
  const auto numSteps = state.numSendSteps[node];
  for (int step = transState.done; step < transState.posted; step++) {
    auto& req = reqs.at(step);
    bool done = false;
    FB_COMMCHECK(ctx.mapper->testRequest(&req, &done));
    if (!done) {
      // if current step is not done, future step will also in progress
      break;
    }

    profileEnd(ctx, node, step, ProfileEvent::kSendTrans);
    transState.done++;
    if (numSteps == transState.done) {
      // Already finished the last sendStep, mark this node is done and enqueue
      // receive of last trans ack from remote rank to indicate all remote
      // chunks are free
      state.numPendingSendNodes--;
    } else {
      // A sendTrans completion gives free chunk for the next preSendTrans
      preState.ready++;
    }
    DEBUG_LOG(
        "SendTrans done for node {} step {}; req {} update done={}, preSendTrans.ready={}/{}, numPendingSendNodes {}",
        node,
        step,
        (void*)&req,
        transState.done,
        preState.ready,
        numSteps,
        state.numPendingSendNodes);
  }
  return commSuccess;
}

inline void issuePostRecvTrans(
    ExecCtx& ctx,
    const int node,
    const int step,
    ProgressState& state,
    const bool isExec) {
  auto& algoRes = ctx.resource;
  auto& postState = state.postRecvTrans[node];

  const auto event = isExec ? ProfileEvent::kFwd : ProfileEvent::kSendRed;
  profileStart(ctx, node, step, event);

  auto* gkSync = algoRes->kSync.recvGKSyncs + node;
  gkSync->post(step);
  postState.posted++;
  DEBUG_LOG(
      "PostRTrans posted for node {} step {}; update posted={}/{}",
      node,
      step,
      postState.posted,
      state.numRecvSteps[node]);
}

inline commResult_t
postLastTransAck(ExecCtx& ctx, ProgressState& state, const int node) {
  const auto& statex = ctx.commStatex;
  auto& mapper = ctx.mapper;

  const auto numSteps = state.numRecvSteps[node];
  const auto myLocalRank = statex->localRank();
  const auto railPeerRank = statex->localRankToRank(myLocalRank, node);
  auto* req = getReservedReq(state.remChkSSyncReqs[node], numSteps);

  // For last sync, use special value for correctness validation
  int lastReadyVal = lastTransAckValue(statex->rank());
  return mapper->isendCtrlMsg(&lastReadyVal, sizeof(int), railPeerRank, req);
}

inline commResult_t postRecvChunkReady(
    ExecCtx& ctx,
    ProgressState& state,
    const int node,
    const int step) {
  const auto& statex = ctx.commStatex;
  auto& mapper = ctx.mapper;

  const auto numSteps = state.numRecvSteps[node];
  // skip the last NUM_CHUNKS, since sendTrans will be done by that
  if (numSteps - step <= NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS) {
    return commSuccess;
  }
  const auto myLocalRank = statex->localRank();
  const auto railPeerRank = statex->localRankToRank(myLocalRank, node);
  auto* req = getReservedReq(state.remChkSSyncReqs[node], numSteps);
  return mapper->isendCtrl(railPeerRank, req);
}

inline commResult_t checkPostRecvTransDone(
    ExecCtx& ctx,
    ProgressState& state,
    const int node,
    const bool isExec) {
  auto& algoRes = ctx.resource;

  auto& postState = state.postRecvTrans[node];
  auto* gkSync = algoRes->kSync.recvGKSyncs + node;

  const auto numSteps = state.numRecvSteps[node];
  for (auto step = postState.done; step < postState.posted; step++) {
    if (!gkSync->isComplete(step)) {
      break;
    }

    const auto event = isExec ? ProfileEvent::kFwd : ProfileEvent::kSendRed;
    profileEnd(ctx, node, step, event);

    postState.done++;
    if (postState.done == numSteps) {
      // Already finished the last recvStep, mark this node is done and post
      // last trans ack to send rank indicating all local chunks are free to use
      // for next opCount
      state.numPendingRecvNodes--;
      FB_COMMCHECK(postLastTransAck(ctx, state, node));
    } else {
      // sendCtrl to sender as the fwdChunk is ready for next sendTrans
      FB_COMMCHECK(postRecvChunkReady(ctx, state, node, step));
    }
    DEBUG_LOG(
        "PostRTrans done for node {} step {}; update done={}/{}, numPendingRecvNodes {}",
        node,
        step,
        postState.done,
        numSteps,
        state.numPendingRecvNodes);
  }
  return commSuccess;
}

inline commResult_t
checkRecvTransDone(ExecCtx& ctx, ProgressState& state, int node) {
  auto& mapper = ctx.mapper;

  auto& transState = state.recvTrans[node];
  auto& postState = state.postRecvTrans[node];
  const auto numSteps = state.numRecvSteps[node];
  auto& notify = state.recvTransNotifies[node];
  for (auto step = transState.done; step < numSteps; step++) {
    bool done = false;
    FB_COMMCHECK(mapper->checkNotify(&notify, &done));
    if (done) {
      profilePoint(ctx, node, step, ProfileEvent::kRecvTrans);
      transState.done++;
      // A recvTrans completion enables consequent postRecv by kernel
      postState.ready++;
      DEBUG_LOG(
          "RecvTrans done for node {}; update done={}, pRecvState.ready={}/{}",
          node,
          transState.done,
          postState.ready,
          numSteps);
    } else {
      // return to main loop if hit an incomplete receive; remaining receives
      // will be checked next time calling into this function
      break;
    }
  }
  return commSuccess;
}

inline commResult_t
progressXNodeRecv(ExecCtx& ctx, ProgressState& state, bool isExec) {
  if (state.numPendingRecvNodes == 0) {
    return commSuccess;
  }

  const auto& statex = ctx.commStatex;
  const int myNode = statex->node();
  const int nNodes = statex->nNodes();

  for (int n = 0; n < nNodes; n++) {
    auto& postState = state.postRecvTrans[n];
    auto& transState = state.recvTrans[n];
    const auto numStep = state.numRecvSteps[n];
    // Skip if no more pending sendBlocks to the peer node
    if (numStep == postState.done || n == myNode) {
      continue;
    }

    if (transState.done < numStep) {
      FB_COMMCHECK(checkRecvTransDone(ctx, state, n));
    }

    // postRecvTrans is fwd in exec() or sendRed in combine());
    // issue if ready
    if (postState.ready > postState.posted) {
      const auto posted = postState.posted;
      for (auto step = posted; step < postState.ready; step++) {
        issuePostRecvTrans(ctx, n, step, state, isExec);
      }
    }

    if (postState.posted > postState.done) {
      FB_COMMCHECK(checkPostRecvTransDone(ctx, state, n, isExec));
    }
  }
  return commSuccess;
}

inline commResult_t
progressXNodeSend(ExecCtx& ctx, ProgressState& state, bool isExec) {
  if (state.numPendingSendNodes == 0) {
    return commSuccess;
  }

  const auto& statex = ctx.commStatex;
  const auto myNode = statex->node();
  const auto nNodes = statex->nNodes();

  for (auto n = 0; n < nNodes; n++) {
    auto& preState = state.preSendTrans[n];
    auto& transState = state.sendTrans[n];
    const auto numStep = state.numSendSteps[n];
    // Skip if no more pending sendBlocks to the peer node
    if (numStep == 0 || n == myNode) {
      continue;
    }

    // preSend is sendCopy in exec() or fwdRed in combine();
    // issue if ready
    if (preState.ready > preState.posted) {
      int posted = preState.posted;
      for (int step = posted; step < preState.ready; step++) {
        issuePreSendTrans(ctx, n, step, state, isExec);
      }
    }

    if (preState.posted > preState.done) {
      checkPreSendTransDone(ctx, n, state, isExec);
    }

    // issue sendTrans if ready
    if (transState.ready > transState.posted) {
      int posted = transState.posted;
      for (int step = posted; step < transState.ready; step++) {
        bool ready = false;
        FB_COMMCHECK(checkSendTransReady(ctx, state, n, step, ready));
        if (!ready) {
          break; // skip all remaining steps if remote chunk is not ready
        }
        FB_COMMCHECK(issueSendTrans(ctx, state, n, step));
      }
    }

    FB_COMMCHECK(checkSendTransDone(ctx, state, n));
  }
  return commSuccess;
}

inline commResult_t
waitSyncComplete(ExecCtx& ctx, ProgressState& state, const bool isExec) {
  auto& mapper = ctx.mapper;
  const auto& statex = ctx.commStatex;

  for (auto node = 0; node < state.remChkRSyncReqs.size(); node++) {
    auto& nodeSyncs = state.remChkRSyncReqs[node];
    for (auto& req : nodeSyncs) {
      FB_COMMCHECK(mapper->waitRequest(&req));
    }

    // For any node with send transmission, we expect the last received sync is
    // lastTransAck, and sanity check the ack value
    if (nodeSyncs.size()) {
      const auto myLocalRank = statex->localRank();
      const auto railPeerRank = statex->localRankToRank(myLocalRank, node);
      FB_CHECKTHROW(
          state.lastTransAck[node] == lastTransAckValue(railPeerRank),
          "Unexpected lastTransAck value {} from rail peer rank {} on node {}",
          state.lastTransAck[node],
          railPeerRank,
          node);
    }
  }

  for (auto& nodeSyncs : state.remChkSSyncReqs) {
    for (auto& req : nodeSyncs) {
      FB_COMMCHECK(mapper->waitRequest(&req));
    }
  }
  return commSuccess;
}

inline commResult_t
updateProgressXNodeSendState(ExecCtx& ctx, ProgressState& state, bool isExec) {
  const auto& statex = ctx.commStatex;
  const auto& pArgs = ctx.pArgs;
  const int nNodes = statex->nNodes();
  const int myNode = statex->node();
  const auto maxNumStepBlks = pArgs.maxNumStepBlks;

  // tracks number of sendTrans blocks, copied from device
  // metadata in prepare kernel
  int* numSendBlocks = nullptr;
  if (isExec) {
    GET_RESOURCE_REGBUFPTR(ctx.resource, kTmpNumSendIdxH, numSendBlocks);
  } else {
    GET_RESOURCE_REGBUFPTR(ctx.resource, kTmpNumFwdIdxH, numSendBlocks);
  }

  state.numSendBlocks = numSendBlocks;
  state.numPendingSendNodes = 0;
  for (int n = 0; n < nNodes; n++) {
    const auto numBlocks = numSendBlocks[n];

    // Skip empty peer node or self, since kernel will handle intra-node
    // forward
    if (numBlocks == 0 || n == myNode) {
      continue;
    }

    // Update total number of steps per remote node; data is always split
    // as full chunks to saturate network
    const auto numSteps = (numBlocks + maxNumStepBlks - 1) / maxNumStepBlks;
    state.numSendSteps[n] = numSteps;
    state.numPendingSendNodes++;
    DEBUG_LOG(
        "Rank {} updated numSendSteps[{}] = {}, numPendingSendNodes {}, based on numBlocks {} maxNumStepBlks {}",
        statex->rank(),
        n,
        state.numSendSteps[n],
        state.numPendingSendNodes,
        numBlocks,
        maxNumStepBlks);

    // First NUM_CHUNKS preSend are ready to post, since tmpSendBuff is not
    // yet used.
    state.preSendTrans[n].ready =
        std::min(state.numSendSteps[n], NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS);

    // preserve space for request containers so they won't be resized
    state.sendTransReqs[n].reserve(numSteps);
    state.remChkRSyncReqs[n].reserve(numSteps);

    // pre-post all recvCtrl for sendTransAck
    prePostSendTransAck(ctx, state, n);
  }

  return commSuccess;
}

inline commResult_t
updateProgressXNodeRecvState(ExecCtx& ctx, ProgressState& state, bool isExec) {
  const auto& statex = ctx.commStatex;
  auto& mapper = ctx.mapper;
  const auto& pArgs = ctx.pArgs;

  const int nNodes = statex->nNodes();
  const int myLocalRank = statex->localRank();
  const int myNode = statex->node();
  const int myRank = statex->rank();
  const auto maxNumStepBlks = pArgs.maxNumStepBlks;

  // tracks number of recvTrans blocks, copied from device
  // metadata in prepare kernel
  int* numRecvBlocks = nullptr;
  if (isExec) {
    GET_RESOURCE_REGBUFPTR(ctx.resource, kTmpNumFwdIdxH, numRecvBlocks);
  } else {
    GET_RESOURCE_REGBUFPTR(ctx.resource, kTmpNumSendIdxH, numRecvBlocks);
  }

  // Update ProgressState to track progress completion
  state.numPendingRecvNodes = 0;
  for (int n = 0; n < nNodes; n++) {
    const auto numBlocks = numRecvBlocks[n];

    // intra-node recvFwd is fully handled by kernel
    if (n == myNode || numBlocks == 0) {
      continue;
    }

    // Update total number of steps per remote node; data is always split
    // as full chunks to saturate network
    const auto numSteps = (numBlocks + maxNumStepBlks - 1) / maxNumStepBlks;
    state.numRecvSteps[n] = numSteps;
    state.numPendingRecvNodes++;
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "Rank {} updated numRecvSteps[{}] = {}, numPendingRecvNodes {}, based on numBlocks {} maxNumStepBlks {}",
        myRank,
        n,
        state.numRecvSteps[n],
        state.numPendingRecvNodes,
        numBlocks,
        maxNumStepBlks);

    // preserve space for request containers so they won't be resized
    state.remChkSSyncReqs[n].reserve(numSteps);
  }

  auto& tmpFwdBuff = ctx.resource->getBuf(ResourceBufName::kTmpFwdBuff);
  for (int n = 0; n < nNodes; n++) {
    const auto peerRank = statex->localRankToRank(myLocalRank, n);
    FB_COMMCHECK(mapper->initNotify(
        peerRank, tmpFwdBuff.regHdl, &state.recvTransNotifies[n]));
  }
  return commSuccess;
}

inline void setCommonTraceMetadata(
    utils::TraceRecord* ts,
    struct OpElem* op,
    const bool isExec) {
  auto comm = op->comm_;
  auto statex = comm->statex_.get();
  const auto myRank = statex->rank();

  const auto config =
      reinterpret_cast<PersistConfig*>(op->alltoallv_dedup_exec.algoConfig);

  ts->addMetadata("opCount", std::to_string(thOpCount));
  ts->addMetadata("rank", std::to_string(myRank));
  ts->addMetadata("localRank", std::to_string(statex->localRank()));
  ts->addMetadata("numRanks", std::to_string(statex->nRanks()));
  ts->addMetadata("numNodes", std::to_string(statex->nNodes()));
  ts->addMetadata("numLocalRanks", std::to_string(statex->nLocalRanks()));

  ts->addMetadata("numSendGroups", std::to_string(config->numSendGroups));
  ts->addMetadata("numSendWorkers", std::to_string(config->numSendWorkers));
  ts->addMetadata("numFwdWorkers", std::to_string(config->numFwdWorkers));
  ts->addMetadata("numRecvGroups", std::to_string(config->numRecvGroups));
  ts->addMetadata("numRecvWorkers", std::to_string(config->numRecvWorkers));
  ts->addMetadata(
      "numIntraFwdWorkers", std::to_string(config->numIntraFwdWorkers));
  ts->addMetadata(
      "numIntraRecvWorkers", std::to_string(config->numIntraRecvWorkers));
}

inline void setupExecKernelArgs(
    const PersistArgs& pArgs,
    const PersistConfig& config,
    const ExecArgs& execArgs,
    ResourceImpl* resource,
    const uint64_t opCount,
    ExecKernArgs& kernArgs) {
  kernArgs.pArgs = pArgs;
  kernArgs.execArgs = execArgs;
  kernArgs.opCount = opCount;
  kernArgs.config = config;

  // Resource arguments
  resource->assignToKernArgs(kernArgs);
}

inline void setupGpeOp(
    const PersistArgs& pArgs,
    const ExecArgs& execArgs,
    const uint64_t opCount,
    ResourceRef& resRef,
    PersistConfig& config,
    CtranComm* comm,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    utils::TraceLogger* ctran_trace_logger) {
  std::unique_ptr<struct OpElem> op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::ALLTOALLV_DEDUP, comm, opCount));

  op->alltoallv_dedup_exec.pArgs =
      const_cast<void*>(reinterpret_cast<const void*>(&pArgs));
  op->alltoallv_dedup_exec.algoResource = &resRef;
  op->alltoallv_dedup_exec.algoConfig = &config;
  op->alltoallv_dedup_exec.ctran_trace_logger =
      reinterpret_cast<void*>(ctran_trace_logger);

  opGroup.push_back(std::move(op));
}

inline void setupKernelConfig(
    const ICtran* ctran,
    const PersistConfig& pConfig,
    KernelConfig& config) {
  config.numThreads = pConfig.numThreads;
  config.numBlocks = pConfig.numSendGroups * pConfig.numSendWorkers +
      pConfig.numFwdWorkers + pConfig.numRecvGroups * pConfig.numRecvWorkers +
      pConfig.numIntraFwdWorkers + pConfig.numIntraRecvWorkers;
  config.args.devState_d = ctran->algo->getDevState();
}
} // namespace ctran::alltoallvdedup
