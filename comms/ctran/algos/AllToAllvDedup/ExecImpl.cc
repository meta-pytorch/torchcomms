// Copyright (c) Meta Platforms, Inc. and affiliates.

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

// Forward declaration of kernel at global scope
template <typename T>
extern __global__ void ncclKernelAllToAllvDedup(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::alltoallvdedup::ExecKernArgs args);

namespace ctran::alltoallvdedup {
// updated when GPE thread starts a new collective, only for logging purpose
thread_local uint64_t thOpCount = -1;

using namespace utils;
// #define DEBUG_LOG(fmt, ...) \
//   CLOGF_SUBSYS(INFO, COLL, "opCount {} " fmt, thOpCount, ##__VA_ARGS__)
#define DEBUG_LOG(fmt, ...) \
  do {                      \
  } while (0);

namespace {
void* alltoallv_dedup_dpKerns[commNumTypes] = {
    (void*)ncclKernelAllToAllvDedup<int8_t>,
    (void*)ncclKernelAllToAllvDedup<uint8_t>,
    (void*)ncclKernelAllToAllvDedup<int32_t>,
    (void*)ncclKernelAllToAllvDedup<uint32_t>,
    (void*)ncclKernelAllToAllvDedup<int64_t>,
    (void*)ncclKernelAllToAllvDedup<uint64_t>,
    (void*)ncclKernelAllToAllvDedup<half>,
    (void*)ncclKernelAllToAllvDedup<float>,
    (void*)ncclKernelAllToAllvDedup<double>,
#if defined(__CUDA_BF16_TYPES_EXIST__)
    (void*)ncclKernelAllToAllvDedup<__nv_bfloat16>,
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    (void*)ncclKernelAllToAllvDedup<__nv_fp8_e4m3>,
    (void*)ncclKernelAllToAllvDedup<__nv_fp8_e5m2>,
#endif
};

struct ExecCtx {
  ExecArgs args; // copied from opElem
  PersistArgs pArgs;
  ResourceRef* resource;
  PersistConfig* config;
  ncclx::CommStateX* commStatex;
  CtranMapper* mapper;
  utils::TraceRecord* ts;
  size_t opCount;
};

struct ProgressState {
  struct StepState {
    int ready{0};
    int posted{0};
    int done{0};
  };
  // tracks the state of copies from sendBuff to tmpSendBuffs
  std::vector<StepState> sendCopy;
  // tracks the state of issued puts on the sender side
  std::vector<StepState> sendTrans;
  // tracks the state of copies from tmpRecvBuffs to recvBuff
  std::vector<StepState> recvCopy;
  std::vector<int> numSendSteps;
  int numPendingSendNodes{0};

  // put requests for each node each step
  std::vector<std::vector<std::unique_ptr<CtranMapperRequest>>> sendTransReqs;
  std::vector<std::unordered_map<int, std::unique_ptr<CtranMapperRequest>>>
      fwdChunkRSyncReqs;

  // tracks the state of issued puts on the receiver side
  std::vector<StepState> recvTrans;
  // tracks the state of blocks forwarded to peers
  std::vector<StepState> recvFwd;
  std::vector<int> numFwdSteps;

  // Unlike request, 1 notify can be reused for multiple steps; thus 1 notify
  // per node
  std::vector<CtranMapperNotify> recvTransNotifies;
  std::vector<std::unordered_map<int, std::unique_ptr<CtranMapperRequest>>>
      fwdChunkSSyncReqs;
  int numPendingRecvFwdNodes{0};

  ProgressState(int nNodes, int nLocalRanks) {
    sendCopy.resize(nNodes, StepState());
    sendTrans.resize(nNodes, StepState());
    recvCopy.resize(nLocalRanks, StepState());
    numSendSteps.resize(nNodes, 0);
    sendTransReqs.resize(nNodes);
    fwdChunkRSyncReqs.resize(nNodes);
    recvTrans.resize(nNodes, StepState());
    recvFwd.resize(nNodes, StepState());
    numFwdSteps.resize(nNodes, 0);
    recvTransNotifies.resize(nNodes, CtranMapperNotify());
    fwdChunkSSyncReqs.resize(nNodes);
  }
};

inline void
syncPostSendCopy(ExecCtx& ctx, int node, int step, ProgressState& state) {
  auto& algoRes = ctx.resource;

  auto& copyState = state.sendCopy[node];
  algoRes->sendCopyGKSyncs[node]->post(step);
  copyState.posted++;
  DEBUG_LOG(
      "Rank {} sendCopy posted for node {} step {}, sync {}; update sendCopy.posted={}/{}",
      ctx.commStatex->rank(),
      node,
      step,
      algoRes->sendCopyGKSyncs[node],
      copyState.done,
      state.numSendSteps[node]);
}

inline void pollRecvCopyProgress(ExecCtx& ctx, ProgressState& state) {
  for (int n = 0; n < ctx.commStatex->nLocalRanks(); n++) {
    auto& copyState = state.recvCopy[n];
    auto& algoRes = ctx.resource;
    auto& sync = algoRes->recvCopyGKSyncs[n];
    auto step = copyState.done / 2;
    while (sync->isComplete(copyState.done)) {
      if (copyState.done % 2 == 0) {
        ctx.ts->startInterval(
            "recvCopy " + fmt::to_string(n),
            copyState.done / 2,
            500 + n * NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS +
                (step % NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS));
      } else {
        ctx.ts->endInterval(
            "recvCopy " + fmt::to_string(n), copyState.done / 2);
      }
      copyState.done++;
    }
  }
}

inline void pollSendCopyProgress(ExecCtx& ctx, int node, ProgressState& state) {
  auto& algoRes = ctx.resource;

  auto& copyState = state.sendCopy[node];
  auto& transState = state.sendTrans[node];
  auto& sync = algoRes->sendCopyGKSyncs[node];
  for (int step = copyState.done; step < copyState.posted; step++) {
    if (sync->isComplete(step)) {
      copyState.done++;
      ctx.ts->endInterval("sendCopy", step);
      transState.ready++;
      DEBUG_LOG(
          "Rank {} sendCopy done for node {} step {}; update sendCopy.done={}, sendTrans.ready={}/{}",
          ctx.commStatex->rank(),
          node,
          step,
          copyState.done,
          transState.ready,
          state.numSendSteps[node]);
    }
  }
}

inline commResult_t checkSendTransReady(
    ExecCtx& ctx,
    ProgressState& state,
    int node,
    int step,
    bool& ready) {
  CtranMapperRequest* req = nullptr;
  if (step < NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS) {
    ready = true;
    return commSuccess; // first step is always ready
  }

  auto& statex = ctx.commStatex;
  auto& mapper = ctx.mapper;

  // Once used all remote chunks, need to sync with remote peer to confirm
  // the chunk is ready for reuse
  auto it = state.fwdChunkRSyncReqs[node].find(step);
  // If the step's recvCtrl is not yet posted, post it now. If sendCtrl has
  // already arrived, the request will complete immediately.
  if (it == state.fwdChunkRSyncReqs[node].end()) {
    const int myLocalRank = statex->localRank();
    int railPeerRank = statex->localRankToRank(myLocalRank, node);
    FB_COMMCHECK(mapper->irecvCtrl(railPeerRank, &req));
    state.fwdChunkRSyncReqs[node][step] =
        std::unique_ptr<CtranMapperRequest>(req);
  } else {
    req = it->second.get();
  }

  bool done_ = false;
  FB_COMMCHECK(mapper->testRequest(req, &done_));
  ready = done_;
  return commSuccess;
}

inline commResult_t
postSendTrans(ExecCtx& ctx, ProgressState& state, int node, int step) {
  auto& statex = ctx.commStatex;
  auto& mapper = ctx.mapper;
  std::map<std::string, std::string> metaData = {
      {"peerNode", std::to_string(node)},
      {"step", std::to_string(step)},
      {"tmpChunkSize", std::to_string(NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE)}};
  ctx.ts->startInterval(
      "sendTrans",
      step,
      200 + node * NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS +
          (step % NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS),
      metaData);

  const int myLocalRank = statex->localRank();
  int railPeerRank = statex->localRankToRank(myLocalRank, node);
  const int myNode = statex->node();

  auto& tmpSendBuff = ctx.resource->getBuf(ResourceBufName::kTmpSendBuff);
  auto& tmpRemFwdBuffs = ctx.resource->getRemBufs(ResourceBufName::kTmpFwdBuff);

  void* tmpSendChunk = getTmpChunkPtr(*ctx.config, tmpSendBuff.ptr, step, node);

  void* remTmpFwdBuff =
      getTmpChunkPtr(*ctx.config, tmpRemFwdBuffs[node].ptr, step, myNode);
  CtranMapperRequest* req = nullptr;
  FB_COMMCHECK(mapper->iput(
      tmpSendChunk,
      remTmpFwdBuff,
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, // FIXME: get actual size from
                                             // sendCpy kernel
      railPeerRank,
      CtranMapperConfig{
          .memHdl_ = tmpSendBuff.regHdl,
          .remoteAccessKey_ = tmpRemFwdBuffs[node].rkey,
          .notify_ = true /*notify*/},
      &req));

  state.sendTransReqs[node].push_back(std::unique_ptr<CtranMapperRequest>(req));
  state.sendTrans[node].posted++;

  DEBUG_LOG(
      "Rank {} posted sendTrans to node {} (peerRank {}) step {}, tmpSendChunk={} tmpRemFwdBuffs {} remTmpFwdBuff={}, req={}, updated sendTrans.posted={}",
      ctx.commStatex->rank(),
      node,
      railPeerRank,
      step,
      tmpSendChunk,
      algoRes->tmpRemFwdBuffs[node]->ptr,
      remTmpFwdBuff,
      (void*)req,
      state.sendTrans[node].posted);
  return commSuccess;
}

inline commResult_t
pollSendTransProgress(ExecCtx& ctx, ProgressState& state, int node) {
  auto& transState = state.sendTrans[node];
  auto& copyState = state.sendCopy[node];
  auto& reqs = state.sendTransReqs[node];
  int totalNSteps = state.numSendSteps[node];
  for (int step = transState.done; step < transState.posted; step++) {
    auto req = reqs[step].get();
    FB_CHECKABORT(
        req != nullptr,
        "sendTrans req not found for node {} step {}",
        node,
        step);
    bool done = false;
    FB_COMMCHECK(ctx.mapper->testRequest(req, &done));
    if (!done) {
      // if current step is not done, future step requests will also not be
      // done
      break;
    } else {
      ctx.ts->endInterval("sendTrans", step);

      transState.done++;
      if (totalNSteps == transState.done) {
        // Already finished the last sendStep, mark this node is done
        state.numPendingSendNodes--;
      } else {
        // A sendTrans completion gives free chunk for the next sendCopy
        copyState.ready++;
      }
      DEBUG_LOG(
          "sendTrans done for node {} step {}; update sendTrans.done={}, sendCopy.ready={}/{}, numPendingSendNodes {}",
          node,
          step,
          transState.done,
          copyState.ready,
          totalNSteps,
          state.numPendingSendNodes);
    }
  }
  return commSuccess;
}

inline void
syncPostFwd(ExecCtx& ctx, int node, int step, ProgressState& state) {
  std::map<std::string, std::string> metaData = {
      {"peerNode", std::to_string(node)},
      {"step", std::to_string(step)},
      {"tmpChunkSize", std::to_string(NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE)}};
  ctx.ts->startInterval(
      "Fwd",
      step,
      400 + node * NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS +
          (step % NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS),
      metaData);

  auto& algoRes = ctx.resource;

  auto& fwdState = state.recvFwd[node];
  algoRes->recvFwdGKSyncs[node]->post(step);
  fwdState.posted++;
  DEBUG_LOG(
      "Rank {} recvFwd posted for node {} step {}; update recvFwd.posted={}/{}",
      ctx.commStatex->rank(),
      node,
      step,
      fwdState.done,
      state.numFwdSteps[node]);
}

inline commResult_t
postFwdChunkReady(ExecCtx& ctx, ProgressState& state, int node, int step) {
  const auto& statex = ctx.commStatex;
  auto& mapper = ctx.mapper;

  const int totalNSteps = state.numFwdSteps[node];
  // No need to post the last NUM_CHUNKS, since sendTrans will be done by
  // that
  if (totalNSteps - step <= NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS) {
    return commSuccess;
  }
  int myLocalRank = statex->localRank();
  int railPeerRank = statex->localRankToRank(myLocalRank, node);
  CtranMapperRequest* req = nullptr;
  FB_COMMCHECK(mapper->isendCtrl(railPeerRank, &req));
  state.fwdChunkSSyncReqs[node][step] =
      std::unique_ptr<CtranMapperRequest>(req);
  return commSuccess;
}

inline commResult_t
pollFwdProgress(ExecCtx& ctx, ProgressState& state, int node) {
  auto& algoRes = ctx.resource;

  auto& fwdState = state.recvFwd[node];
  auto& sync = algoRes->recvFwdGKSyncs[node];
  const int totalNSteps = state.numFwdSteps[node];
  for (int step = fwdState.done; step < fwdState.posted; step++) {
    if (sync->isComplete(step)) {
      ctx.ts->endInterval("Fwd", step);

      fwdState.done++;
      if (fwdState.done == totalNSteps) {
        // Already finished the last recvStep, mark this node is done
        state.numPendingRecvFwdNodes--;
      } else {
        // sendCtrl to sender as the fwdChunk is ready for next sendTrans
        FB_COMMCHECK(postFwdChunkReady(ctx, state, node, step));
      }
      DEBUG_LOG(
          "recvFwd done for node {} step {}; update recvFwd.done={}/{}, numPendingRecvFwdNodes {}",
          node,
          step,
          fwdState.done,
          totalNSteps,
          state.numPendingRecvFwdNodes);
    } else {
      break;
    }
  }
  return commSuccess;
}

inline commResult_t
pollFwdTransProgress(ExecCtx& ctx, ProgressState& state, int node) {
  auto& mapper = ctx.mapper;

  auto& transState = state.recvTrans[node];
  auto& fwdState = state.recvFwd[node];
  int totalNSteps = state.numFwdSteps[node];
  auto& notify = state.recvTransNotifies[node];
  for (int step = transState.done; step < totalNSteps; step++) {
    bool done = false;
    FB_COMMCHECK(mapper->checkNotify(&notify, &done));
    if (done) {
      std::map<std::string, std::string> metaData = {
          {"peerNode", std::to_string(node)}, {"step", std::to_string(step)}};
      ctx.ts->addPoint(
          "recvTrans",
          step,
          300 + node * NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS +
              (step % NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS),
          metaData);
      transState.done++;
      // A recvTrans completion enables consequent recvFwd by kernel
      fwdState.ready++;
      DEBUG_LOG(
          "Rank {} recvTrans done for node {} step {}; update recvTrans.done={}, fwdState.ready={}/{}",
          ctx.commStatex->rank(),
          node,
          step,
          transState.done,
          fwdState.ready,
          totalNSteps);
    }
  }
  return commSuccess;
}

inline commResult_t progressFwd(ExecCtx& ctx, ProgressState& state) {
  if (state.numPendingRecvFwdNodes == 0) {
    return commSuccess;
  }

  const auto& statex = ctx.commStatex;
  const int myNode = statex->node();
  const int nNodes = statex->nNodes();

  for (int n = 0; n < nNodes; n++) {
    // Skip if no more pending sendBlocks to the peer node
    if (state.numFwdSteps[n] == state.recvFwd[n].done || n == myNode) {
      continue;
    }

    // - Check recvTrans complete
    if (state.recvTrans[n].done < state.numFwdSteps[n]) {
      FB_COMMCHECK(pollFwdTransProgress(ctx, state, n));
    }

    // - Try post forward if ready
    if (state.recvFwd[n].ready > state.recvFwd[n].posted) {
      int posted = state.recvFwd[n].posted;
      for (int step = posted; step < state.recvFwd[n].ready; step++) {
        syncPostFwd(ctx, n, step, state);
      }
    }

    // - Check completed forward and post fwdChunk ready
    if (state.recvFwd[n].posted > state.recvFwd[n].done) {
      FB_COMMCHECK(pollFwdProgress(ctx, state, n));
    }
  }
  return commSuccess;
}

inline commResult_t progressSend(ExecCtx& ctx, ProgressState& state) {
  if (state.numPendingSendNodes == 0) {
    return commSuccess;
  }

  const auto& statex = ctx.commStatex;
  const int myNode = statex->node();
  const int nNodes = statex->nNodes();

  for (int n = 0; n < nNodes; n++) {
    // Skip if no more pending sendBlocks to the peer node
    if (state.numSendSteps[n] == 0 || n == myNode) {
      continue;
    }

    // try post sendCopy if ready
    if (state.sendCopy[n].ready > state.sendCopy[n].posted) {
      int posted = state.sendCopy[n].posted;
      for (int step = posted; step < state.sendCopy[n].ready; step++) {
        std::map<std::string, std::string> metaData = {
            {"peerNode", std::to_string(n)},
            {"step", std::to_string(step)},
            {"tmpChunkSize",
             std::to_string(NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE)}};

        ctx.ts->startInterval(
            "sendCopy",
            step,
            100 + n * NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS +
                (step % NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS),
            metaData);
        syncPostSendCopy(ctx, n, step, state);
      }
    }

    // check sendCopy complete
    if (state.sendCopy[n].posted > state.sendCopy[n].done) {
      pollSendCopyProgress(ctx, n, state);
    }

    // put sendTrans if ready
    if (state.sendTrans[n].ready > state.sendTrans[n].posted) {
      int posted = state.sendTrans[n].posted;
      for (int step = posted; step < state.sendTrans[n].ready; step++) {
        bool ready = false;
        FB_COMMCHECK(checkSendTransReady(ctx, state, n, step, ready));
        if (!ready) {
          break; // skip all remaining steps if remote fwdChunk is not ready
        }
        FB_COMMCHECK(postSendTrans(ctx, state, n, step));
      }
    }

    // check sendTrans complete
    FB_COMMCHECK(pollSendTransProgress(ctx, state, n));
  }
  return commSuccess;
}

inline commResult_t updateProgressSendState(
    ExecCtx& ctx,
    ProgressState& state) {
  const auto& statex = ctx.commStatex;
  const auto& pArgs = ctx.pArgs;
  const int nNodes = statex->nNodes();
  const int myNode = statex->node();
  const int nLocalRanks = statex->nLocalRanks();

  // tmpNumSendBlocksBuffH has been updated in prepare
  auto& tmpNumSendBlocksBuffH =
      ctx.resource->getBuf(ResourceBufName::kTmpNumSendBlocksBuffH);

  const auto nLocalBuckets = pArgs.numRecvBuckets * nLocalRanks;
  const auto tmpNumSendBlocks = ptrElemOffset<size_t>(
      tmpNumSendBlocksBuffH.ptr,
      myNode * nNodes * (nLocalBuckets + nLocalRanks + 1));

  DEBUG_LOG(
      "Rank {} loaded tmpNumSendBlocks: {}",
      statex->rank(),
      array2DToStr(
          tmpNumSendBlocks, nNodes, (nLocalBuckets + nLocalRanks + 1)));

  const size_t maxNumBlocksPerChunk =
      getMaxNumBlocksPerChunk(ctx.config, pArgs, commTypeSize(pArgs.datatype));
  for (int n = 0; n < nNodes; n++) {
    const auto numSendBlocksToNode = tmpNumSendBlocks
        [n * (nLocalBuckets + nLocalRanks + 1) + nLocalBuckets + nLocalRanks];
    // Skip empty peer node or self, since kernel will handle intra-node
    // forward
    if (numSendBlocksToNode == 0 || n == myNode) {
      continue;
    }

    // Update total number of steps to the peer node
    state.numSendSteps[n] =
        (numSendBlocksToNode + maxNumBlocksPerChunk - 1) / maxNumBlocksPerChunk;
    state.numPendingSendNodes++;
    DEBUG_LOG(
        "Rank {} updated numSendSteps[{}] = {}, numPendingSendNodes {}, based on maxNumBlocksPerChunk {}",
        statex->rank(),
        n,
        state.numSendSteps[n],
        state.numPendingSendNodes,
        maxNumBlocksPerChunk);

    // First NUM_CHUNKS sendCopy are ready to post, since tmpSendBuff is not
    // yet used.
    state.sendCopy[n].ready =
        std::min(state.numSendSteps[n], NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS);
  }

  return commSuccess;
}

inline commResult_t updateProgressFwdState(ExecCtx& ctx, ProgressState& state) {
  const auto& statex = ctx.commStatex;
  auto& mapper = ctx.mapper;

  const int nNodes = statex->nNodes();
  const int myLocalRank = statex->localRank();
  const int myNode = statex->node();
  const int myRank = statex->rank();
  const int nLocalRanks = statex->nLocalRanks();

  const size_t numBlocksPerChunk = getMaxNumBlocksPerChunk(
      ctx.config, ctx.pArgs, commTypeSize(ctx.pArgs.datatype));

  // numForwardBlocksH has been updated in prepare
  size_t* numForwardBlocks = nullptr;
  GET_RESOURCE_BUFPTR(ctx.resource, kNumForwardBlocksH, numForwardBlocks);

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "Rank {} loaded numForwardBlocksH: {}",
      myRank,
      array2DToStr(numForwardBlocks, nNodes, nLocalRanks));

  // Update ProgressState to track progress completion
  for (int n = 0; n < nNodes; n++) {
    // intra-node recvFwd is fully handled by kernel
    if (n == myNode) {
      continue;
    }
    // Update total number of pending fwdBlocks from the peer node
    int sendRank = statex->localRankToRank(myLocalRank, n);
    size_t numFwds = numForwardBlocks[sendRank];
    state.numFwdSteps[n] =
        (numFwds + numBlocksPerChunk - 1) / numBlocksPerChunk;
  }

  state.numPendingRecvFwdNodes = 0;
  auto& tmpFwdBuff = ctx.resource->getBuf(ResourceBufName::kTmpFwdBuff);
  for (int n = 0; n < nNodes; n++) {
    if (state.numFwdSteps[n] > 0) {
      state.numPendingRecvFwdNodes++;
    }
    // Always insert a notify for index based on nodeId
    int railPeerRank = statex->localRankToRank(myLocalRank, n);
    FB_COMMCHECK(mapper->initNotify(
        railPeerRank, tmpFwdBuff.regHdl, &state.recvTransNotifies[n]));
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "Rank {} updated numFwdSteps[{}] = {}, numPendingRecvFwdNodes {}",
        myRank,
        n,
        state.numFwdSteps[n],
        state.numPendingRecvFwdNodes);
  }
  return commSuccess;
}

void resetSync(ExecCtx& ctx) {
  auto& algoRes = ctx.resource;

  for (auto& sync : algoRes->sendCopyGKSyncs) {
    sync->resetStatus();
  }
  for (auto& sync : algoRes->recvFwdGKSyncs) {
    sync->resetStatus();
  }
}

inline void metadataSyncPost(
    ctran::algos::GpeKernelSync* sync,
    int myRank,
    const std::string& name) {
  sync->post(0);
  DEBUG_LOG("Rank {} posted {} {} = 0", myRank, name, sync);
}

inline ExecArgs opToExecArgs(struct OpElem* op) {
  return ExecArgs{
      .sendBuff = op->alltoallv_dedup_exec.sendBuff,
      .blockRecvBuckets = op->alltoallv_dedup_exec.blockRecvBuckets,
      .numSendBlocks = op->alltoallv_dedup_exec.numSendBlocks,
      .numRecvBlocks = op->alltoallv_dedup_exec.numRecvBlocks,
      .recvOffsets = op->alltoallv_dedup_exec.recvOffsets,
      .numForwardBlocks = op->alltoallv_dedup_exec.numForwardBlocks,
      .recvBuff = op->alltoallv_dedup_exec.recvBuff,
      .blockSendRanks = op->alltoallv_dedup_exec.blockSendRanks};
}

inline void execArgsToOp(const ExecArgs& args, struct OpElem* op) {
  op->alltoallv_dedup_exec.sendBuff = args.sendBuff;
  op->alltoallv_dedup_exec.blockRecvBuckets = args.blockRecvBuckets;
  op->alltoallv_dedup_exec.numSendBlocks = args.numSendBlocks;
  op->alltoallv_dedup_exec.numRecvBlocks = args.numRecvBlocks;
  op->alltoallv_dedup_exec.recvOffsets = args.recvOffsets;
  op->alltoallv_dedup_exec.numForwardBlocks = args.numForwardBlocks;
  op->alltoallv_dedup_exec.recvBuff = args.recvBuff;
  op->alltoallv_dedup_exec.blockSendRanks = args.blockSendRanks;
}

commResult_t execGpeFn(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  auto op = opGroup[0].get();
  auto comm = op->comm_;
  thOpCount = op->opCount;

  auto statex = comm->statex_.get();
  const auto myRank = statex->rank();

  const auto algoConfig =
      reinterpret_cast<PersistConfig*>(op->alltoallv_dedup_exec.algoConfig);
  CtranAlgoLogger logger(
      AlgoImpl::algoName(AlgoImpl::Phase::kExec), op->opCount, comm);
  CtranMapper* mapper = comm->ctran_->mapper.get();
  const int nNodes = statex->nNodes();
  const int nLocalRanks = statex->nLocalRanks();

  auto ctran_trace_logger = reinterpret_cast<utils::TraceLogger*>(
      op->alltoallv_dedup_exec.ctran_trace_logger);
  // Always create traceRecord for code simplicity, all recording should be
  // no-op if trace is disabled
  auto ts = std::make_unique<utils::TraceRecord>(
      fmt::format("allToAllvDedupExec_{}", thOpCount), myRank);
  ts->addMetadata("opCount", std::to_string(thOpCount));
  ts->addMetadata("rank", std::to_string(myRank));
  ts->addMetadata("localRank", std::to_string(statex->localRank()));
  ts->addMetadata("numRanks", std::to_string(statex->nRanks()));
  ts->addMetadata("numNodes", std::to_string(statex->nNodes()));
  ts->addMetadata("numLocalRanks", std::to_string(statex->nLocalRanks()));
  ts->addMetadata("numSendGroups", std::to_string(algoConfig->numSendGroups));
  ts->addMetadata("numSendWorkers", std::to_string(algoConfig->numSendWorkers));
  ts->addMetadata("numFwdGroups", std::to_string(algoConfig->numFwdGroups));
  ts->addMetadata("numFwdWorkers", std::to_string(algoConfig->numFwdWorkers));
  ts->addMetadata("numRecvGroups", std::to_string(algoConfig->numRecvGroups));
  ts->addMetadata("numRecvWorkers", std::to_string(algoConfig->numRecvWorkers));

  ExecCtx ctx = {
      .args = opToExecArgs(op),
      .pArgs = *reinterpret_cast<PersistArgs*>(op->alltoallv_dedup_exec.pArgs),
      .resource =
          reinterpret_cast<ResourceRef*>(op->alltoallv_dedup_exec.algoResource),
      .config =
          reinterpret_cast<PersistConfig*>(op->alltoallv_dedup_exec.algoConfig),
      .commStatex = statex,
      .mapper = mapper,
      .ts = ts.get(),
      .opCount = op->opCount};

  // initialize progressState
  ProgressState state = ProgressState(nNodes, nLocalRanks);

  auto& algoRes = ctx.resource;
  auto myNode = statex->node();
  ctx.ts->startInterval(
      "fwdIntra", 0, 400 + myNode * NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS);
  algoRes->recvFwdGKSyncs[myNode]->post(0);

  FB_COMMCHECK(updateProgressSendState(ctx, state));
  FB_COMMCHECK(updateProgressFwdState(ctx, state));
  bool fwdIntraDone = false;
  do {
    FB_COMMCHECK(progressSend(ctx, state));
    if (!fwdIntraDone && algoRes->recvFwdGKSyncs[myNode]->isComplete(1)) {
      ctx.ts->endInterval("fwdIntra", 0);
      fwdIntraDone = true;
    }
    if (fwdIntraDone) {
      FB_COMMCHECK(progressFwd(ctx, state));
    }
    pollRecvCopyProgress(ctx, state);
  } while (state.numPendingSendNodes > 0 || state.numPendingRecvFwdNodes > 0 ||
           !fwdIntraDone);

  // Now both send and forward must be finished on both GPE thread and
  // kernel; reset internal GPE-Kernel sync for next use.
  resetSync(ctx);

  ctran_trace_logger->addTraceRecord(std::move(ts));
  return commSuccess;
}

void setupKernelArgs(
    const PersistArgs& pArgs,
    const PersistConfig& config,
    const ExecArgs& execArgs,
    const ncclx::CommStateX* statex,
    ResourceRef& resRef,
    ExecKernArgs& kernArgs) {
  const auto nLocalRanks = statex->nLocalRanks();

  kernArgs.pArgs = pArgs;
  kernArgs.execArgs = execArgs;

  // Resource arguments
  kernArgs.config = config;

  GET_RESOURCE_BUFPTR(&resRef, kTmpFwdBuff, kernArgs.tmpFwdBuff);
  GET_RESOURCE_BUFPTR(&resRef, kTmpSendBuff, kernArgs.tmpSendBuff);
  GET_RESOURCE_BUFPTR(&resRef, kTmpSendIdx, kernArgs.tmpSendIdx);
  GET_RESOURCE_BUFPTR(&resRef, kTmpIntraFwdIdx, kernArgs.tmpIntraFwdIdx);
  GET_RESOURCE_BUFPTR(&resRef, kTmpRecvBuff, kernArgs.tmpRecvBuff);
  GET_RESOURCE_BUFPTR(&resRef, kTmpRecvOffsets, kernArgs.tmpRecvOffsets);

  std::vector<size_t*> tmpRemRecvBuffs;
  GET_RESOURCE_REM_BUFPTRS(&resRef, kTmpRecvBuff, nLocalRanks, tmpRemRecvBuffs);
  for (int i = 0; i < nLocalRanks; i++) {
    kernArgs.remTmpRecvBuffs[i] = tmpRemRecvBuffs[i];
  }

  kernArgs.kSync = resRef.kSync;
}

void setupKernelConfig(
    const ICtran* ctran,
    const PersistConfig& pConfig,
    KernelConfig& config) {
  config.numThreads = pConfig.numThreads;
  config.numBlocks = pConfig.numThreadBlocks;

  config.args.devState_d = ctran->algo->getDevState();
}

void setupGpeOp(
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
  execArgsToOp(execArgs, op.get());

  opGroup.push_back(std::move(op));
}

} // namespace

commResult_t AlgoImpl::exec(const ExecArgs& execArgs, const uint64_t opCount) {
  // prepare kernel config for self and NVL copies
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALLV_DEDUP,
      stream_,
      algoName(Phase::kExec),
      opCount);
  ExecKernArgs kernArgs;
  auto& resRef = resource_->getRef();
  setupKernelConfig(ctran_, config_, config);
  setupKernelArgs(pArgs, config_, execArgs, statex_, resRef, kernArgs);
  config.algoArgs = &kernArgs; // copied to kernel within submit()

  // TODO: opCount should be passed to all kernels
  kernArgs.opCount = opCount;

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  setupGpeOp(
      pArgs,
      execArgs,
      opCount,
      resRef,
      config_,
      comm_,
      opGroup,
      ctran_trace_logger.get());

  FB_COMMCHECK(ctran_->gpe->submit(
      std::move(opGroup),
      execGpeFn,
      config,
      reinterpret_cast<void*>(alltoallv_dedup_dpKerns[pArgs.datatype])));

  return commSuccess;
}

} // namespace ctran::alltoallvdedup
