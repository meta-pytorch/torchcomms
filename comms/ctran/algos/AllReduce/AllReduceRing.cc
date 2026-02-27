// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <chrono>
#include <mutex>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

#include "comms/ctran/CtranComm.h"

#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/AllReduce/AllReduceRingAutoTune.h"
#include "comms/ctran/algos/AllReduce/AllReduceRingCommon.cuh"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

// Helper macro for 3-arg template kernel function map entries
#define CTRAN_RING_REDOP_FUNCMAP(dtype, type, bidir)               \
  {std::make_pair(dtype, commSum),                                 \
   reinterpret_cast<const void*>(                                  \
       &ncclKernelAllReduceCtranRing<type, commSum, bidir>)},      \
      {std::make_pair(dtype, commProd),                            \
       reinterpret_cast<const void*>(                              \
           &ncclKernelAllReduceCtranRing<type, commProd, bidir>)}, \
      {std::make_pair(dtype, commAvg),                             \
       reinterpret_cast<const void*>(                              \
           &ncclKernelAllReduceCtranRing<type, commAvg, bidir>)},  \
      {std::make_pair(dtype, commMax),                             \
       reinterpret_cast<const void*>(                              \
           &ncclKernelAllReduceCtranRing<type, commMax, bidir>)},  \
  {                                                                \
    std::make_pair(dtype, commMin),                                \
        reinterpret_cast<const void*>(                             \
            &ncclKernelAllReduceCtranRing<type, commMin, bidir>)   \
  }

// Bi-directional AG kernel map (EnableBidirAg=true)
static const std::unordered_map<
    std::pair<commDataType_t, commRedOp_t>,
    const void*,
    CtranPairHash>
    typeToFuncBidir = {
        CTRAN_RING_REDOP_FUNCMAP(commInt8, int8_t, true),
        CTRAN_RING_REDOP_FUNCMAP(commUint8, uint8_t, true),
        CTRAN_RING_REDOP_FUNCMAP(commInt32, int32_t, true),
        CTRAN_RING_REDOP_FUNCMAP(commUint32, uint32_t, true),
        CTRAN_RING_REDOP_FUNCMAP(commInt64, int64_t, true),
        CTRAN_RING_REDOP_FUNCMAP(commUint64, uint64_t, true),
        CTRAN_RING_REDOP_FUNCMAP(commFloat16, half, true),
        CTRAN_RING_REDOP_FUNCMAP(commFloat32, float, true),
        CTRAN_RING_REDOP_FUNCMAP(commFloat64, double, true),
#if defined(__CUDA_BF16_TYPES_EXIST__)
        CTRAN_RING_REDOP_FUNCMAP(commBfloat16, __nv_bfloat16, true),
#endif
};

// Simple kernel map (EnableBidirAg=false, lower register usage)
static const std::unordered_map<
    std::pair<commDataType_t, commRedOp_t>,
    const void*,
    CtranPairHash>
    typeToFuncSimple = {
        CTRAN_RING_REDOP_FUNCMAP(commInt8, int8_t, false),
        CTRAN_RING_REDOP_FUNCMAP(commUint8, uint8_t, false),
        CTRAN_RING_REDOP_FUNCMAP(commInt32, int32_t, false),
        CTRAN_RING_REDOP_FUNCMAP(commUint32, uint32_t, false),
        CTRAN_RING_REDOP_FUNCMAP(commInt64, int64_t, false),
        CTRAN_RING_REDOP_FUNCMAP(commUint64, uint64_t, false),
        CTRAN_RING_REDOP_FUNCMAP(commFloat16, half, false),
        CTRAN_RING_REDOP_FUNCMAP(commFloat32, float, false),
        CTRAN_RING_REDOP_FUNCMAP(commFloat64, double, false),
#if defined(__CUDA_BF16_TYPES_EXIST__)
        CTRAN_RING_REDOP_FUNCMAP(commBfloat16, __nv_bfloat16, false),
#endif
};

namespace ctran::allreduce::ring {

// Check if bi-directional AllGather should be enabled based on CVAR and
// message size.
// Returns true if bidir AG should be used, false for simple kernel.
inline bool shouldEnableBidirAg(size_t messageBytes) {
  int64_t maxSize = NCCL_CTRAN_ALLREDUCE_RING_BIDIR_AG_MAX_SIZE;
  if (maxSize == 0) {
    // Explicitly disabled
    return false;
  }
  if (maxSize < 0) {
    // -1 means enable for all sizes
    return true;
  }
  // Enable only for messages up to maxSize
  return messageBytes <= static_cast<size_t>(maxSize);
}

struct HostArgs {
  int32_t rank{-1};
  int32_t leftRank{-1};
  int32_t rightRank{-1};

  size_t minShardSize{0};

  unsigned int numBlocks{0};
  unsigned int numThreads{0};

  // Enable bi-directional AllGather optimization
  bool enableBidirAg{true};

  // Forward: remote receive buffer on right
  void* rightRemBuf{nullptr};
  CtranMapperRemoteAccessKey rightRemKey;

  // Forward: receive notifications from left
  std::unique_ptr<CtranMapperNotify> leftNotify{nullptr};

  // Reverse: remote receive buffer on left (left neighbor's tmpRecvBufRev)
  void* leftRemBufRev{nullptr};
  CtranMapperRemoteAccessKey leftRemKeyRev;

  // Reverse: receive notifications from right
  std::unique_ptr<CtranMapperNotify> rightNotify{nullptr};
};
struct HostResource {
  CtranComm* comm{nullptr};

  ctran::algos::GpeKernelSync* sendCopySync{nullptr};
  ctran::algos::GpeKernelSync* recvRedCopySync{nullptr};
  ctran::algos::GpeKernelSync* partitionSync{nullptr};

  size_t chunkSize{0};
  size_t numChunks{0};
  void* tmpSendBuf{nullptr};
  void* tmpSendBufHdl{nullptr};
  void* tmpRecvBuf{nullptr};
  void* tmpRecvBufHdl{nullptr};

  // Reverse direction
  ctran::algos::GpeKernelSync* revSendCopySync{nullptr};
  ctran::algos::GpeKernelSync* revRecvCopySync{nullptr};
  void* tmpSendBufRev{nullptr};
  void* tmpSendBufRevHdl{nullptr};
  void* tmpRecvBufRev{nullptr};
  void* tmpRecvBufRevHdl{nullptr};
};

namespace {

const auto myAlgo = NCCL_ALLREDUCE_ALGO::ctring;

inline std::string toHexStr(void* ptr) {
  std::stringstream ss;
  ss << "[" << ptr << "]";
  return ss.str();
}

template <Op op>
inline const std::string
roundLogPrefix(const int round, const int step, const AlgoContext& algoCtx) {
  return fmt::format(
      "partition {} round {}/{} ready {} step {}/{}:",
      algoCtx.partition,
      round,
      algoCtx.opRounds[op].totalRounds,
      algoCtx.opRounds[op].ready,
      step,
      algoCtx.numSteps);
}

inline bool progressSendCheckSendBuf(const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kSendCopy].post;
  int step = algoCtx.opRounds[Op::kSendCopy].postStep.step;

  // don't need to check for first numChunks round
  if (round < algoCtx.numChunks) {
    return true;
  }

  // Check if the previous round used the same tmpSendBuf chunk has finished so
  // we can reuse in this round
  int prevRound = round - algoCtx.numChunks;
  int tmpChunkId = getTmpChunkId(algoCtx, round);

  bool done = algoCtx.opRounds[Op::kSendTrans].done > prevRound;
  if (done) {
    CLOGF_TRACE(
        COLL,
        "{} waited tmpChunkId {} algoCtx.numChunks {} prevRound {}",
        roundLogPrefix<Op::kSendCopy>(round, step, algoCtx),
        tmpChunkId,
        algoCtx.numChunks,
        prevRound);
  }
  return done;
}

inline void prePostRecvRemRecvBuf(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& bufSyncRResps) {
  int totalRounds = algoCtx.opRounds[Op::kSendTrans].totalRounds;

  // Pre-post recvCtrls to receive postRecvBuf sync from right neighbor
  bufSyncRResps.resize(totalRounds);
  for (int round = 0; round < totalRounds; round++) {
    CtranMapperRequest* req;
    FB_COMMCHECKTHROW_EX(
        resource.comm->ctran_->mapper->irecvCtrl(args.rightRank, &req),
        resource.comm->logMetaData_);
    bufSyncRResps.at(round).reset(req);
  }
}

inline bool progressSendCheckRemRecvBuf(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& bufSyncRResps) {
  int round = algoCtx.opRounds[Op::kSendTrans].post;
  int step = algoCtx.opRounds[Op::kSendTrans].postStep.step;
  int prevRound = round - algoCtx.numChunks;
  // Skip for first numChunks round
  if (prevRound < 0) {
    return true;
  }

  auto& resp = bufSyncRResps.at(prevRound);
  FB_CHECKTHROW_EX(
      resp != nullptr,
      resource.comm->logMetaData_,
      fmt::format("bufSyncRResps is not initialized at round {}", prevRound));

  if (resp) {
    bool isComplete = false;
    FB_COMMCHECKTHROW_EX(
        resource.comm->ctran_->mapper->testRequest(resp.get(), &isComplete),
        resource.comm->logMetaData_);
    if (isComplete) {
      int tmpChunkId = getTmpChunkId(algoCtx, round);
      CLOGF_TRACE(
          COLL,
          "{} done tmpChunkId {}",
          roundLogPrefix<Op::kSendTrans>(round, step, algoCtx),
          tmpChunkId);
      return true;
    }
  }
  return false;
}

inline void progressSendCheckTrans(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& dataSResps) {
  int startRound = algoCtx.opRounds[Op::kSendTrans].done;
  int lastRound = algoCtx.opRounds[Op::kSendTrans].post;
  int step = algoCtx.opRounds[Op::kSendTrans].doneStep.step;

  // Check if any round between previous finished round and current posted round
  // has been done
  for (int r = startRound; r < lastRound; r++) {
    auto& resp = dataSResps.at(r);
    if (resp) {
      bool isComplete = false;
      FB_COMMCHECKTHROW_EX(
          resource.comm->ctran_->mapper->testRequest(resp.get(), &isComplete),
          resource.comm->logMetaData_);
      if (isComplete) {
        // FIXME: step might be incorrect
        CLOGF_TRACE(
            COLL,
            "progressSendCheckTrans {} done",
            roundLogPrefix<Op::kSendTrans>(r, step, algoCtx));
        opUpdateDone<Op::kSendTrans>(algoCtx);
      }
    }
  }
}

inline void progressSendPostCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kSendCopy].post;
  int step = algoCtx.opRounds[Op::kSendCopy].postStep.step;
  CLOGF_TRACE(
      COLL, "{} posted", roundLogPrefix<Op::kSendCopy>(round, step, algoCtx));
  resource.sendCopySync->post(round);
}

inline bool progressSendCheckCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kSendCopy].done;
  auto& opStep = algoCtx.opRounds[Op::kSendCopy].doneStep;
  int step = opStep.step;
  int tmpChunkId = getTmpChunkId(algoCtx, round);
  bool done = resource.sendCopySync->isComplete(round);

  if (done) {
    CLOGF_TRACE(
        COLL,
        "{} done: tmpChunkId {}",
        roundLogPrefix<Op::kSendCopy>(round, step, algoCtx),
        tmpChunkId);
  }
  return done;
}

inline void progressSendPostTrans(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& dataSResps) {
  int round = algoCtx.opRounds[Op::kSendTrans].post;
  auto& opStep = algoCtx.opRounds[Op::kSendTrans].postStep;
  int step = opStep.step;

  int tmpChunkId = getTmpChunkId(algoCtx, round);
  auto chunkArg = getRoundArgs<Op::kSendTrans>(algoCtx, round, opStep);
  // A ready to send round should never be with empty chunk
  FB_CHECKTHROW_EX(
      chunkArg.numel > 0,
      resource.comm->logMetaData_,
      "Unexpected empty chunk");

  char* tmpRemoteRecvBuf = reinterpret_cast<char*>(args.rightRemBuf) +
      tmpChunkId * algoCtx.chunkSize;
  char* tmpSendBuf = reinterpret_cast<char*>(resource.tmpSendBuf) +
      tmpChunkId * algoCtx.chunkSize;

  // Get allreduce specific IB config
  static thread_local auto allReduceConfig =
      resource.comm->ctran_->algo->getCollToVcConfig(CollType::ALLREDUCE);

  CtranMapperRequest* req;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->iput(
          tmpSendBuf,
          tmpRemoteRecvBuf,
          chunkArg.numel * algoCtx.typeSize,
          args.rightRank,
          CtranMapperConfig{
              .memHdl_ = resource.tmpSendBufHdl,
              .remoteAccessKey_ = args.rightRemKey,
              .notify_ = true,
              .ibConfig_ = allReduceConfig},
          &req),
      resource.comm->logMetaData_);
  dataSResps.at(round).reset(req);

  CLOGF_TRACE(
      COLL,
      "{} from tmpSendBuf {} to tmpRemoteRecvBuf {} shardId {} shardDataChunkId {} dataOffsetElem {} tmpChunkId {} numel {}",
      roundLogPrefix<Op::kSendTrans>(round, step, algoCtx),
      toHexStr(tmpSendBuf),
      toHexStr(tmpRemoteRecvBuf),
      chunkArg.shardId,
      chunkArg.shardDataChunkId,
      chunkArg.dataOffsetElem,
      tmpChunkId,
      chunkArg.numel);
}

inline bool progressRecvCheckTrans(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRecvTrans].post;
  auto& opStep = algoCtx.opRounds[Op::kRecvTrans].postStep;
  int step = opStep.step;
  int tmpChunkId = getTmpChunkId(algoCtx, round);

  auto chunkArg = getRoundArgs<Op::kRecvTrans>(algoCtx, round, opStep);
  char* tmpRecvBuf = reinterpret_cast<char*>(resource.tmpRecvBuf) +
      tmpChunkId * algoCtx.chunkSize;

  bool done = false;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->checkNotify(args.leftNotify.get(), &done),
      resource.comm->logMetaData_);
  if (done) {
    CLOGF_TRACE(
        COLL,
        "{} to tmpRecvBuf {} shardId {} shardDataChunkId {} dataOffsetElem {} tmpChunkId {}",
        roundLogPrefix<Op::kRecvTrans>(round, step, algoCtx),
        toHexStr(tmpRecvBuf),
        chunkArg.shardId,
        chunkArg.shardDataChunkId,
        chunkArg.dataOffsetElem,
        tmpChunkId);
  }
  return done;
}

inline void progressRecvPostFlush(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& flushResps) {
  int round = algoCtx.opRounds[Op::kRecvFlush].post;
  int step = algoCtx.opRounds[Op::kRecvFlush].postStep.step;
  std::map<std::string, std::string> metaData = {
      {"step", std::to_string(step)}, {"round", std::to_string(round)}};

  int tmpChunkId = getTmpChunkId(algoCtx, round);
  char* tmpRecvBuf = reinterpret_cast<char*>(resource.tmpRecvBuf) +
      tmpChunkId * algoCtx.chunkSize;

  CtranMapperRequest* req;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->iflush(
          tmpRecvBuf, resource.tmpRecvBufHdl, &req),
      resource.comm->logMetaData_);
  flushResps.at(round).reset(req);
}

inline bool progressRecvCheckFlush(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& flushResps) {
  int round = algoCtx.opRounds[Op::kRecvFlush].done;
  int step = algoCtx.opRounds[Op::kRecvFlush].doneStep.step;
  int chunkId = getTmpChunkId(algoCtx, round);

  FB_CHECKTHROW_EX(
      flushResps.at(round) != nullptr,
      resource.comm->logMetaData_,
      fmt::format(
          "Flush resp is not initialized at round {} step {} chunkId {}",
          round,
          step,
          chunkId));
  auto& resp = flushResps.at(round);

  bool isComplete = false;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->testRequest(resp.get(), &isComplete),
      resource.comm->logMetaData_);
  if (isComplete) {
    CLOGF_TRACE(
        COLL, "{} done", roundLogPrefix<Op::kRecvFlush>(round, step, algoCtx));
  }
  return isComplete;
}

inline bool progressRecvCheckSendBuf(const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRecvRedCopy].post;
  int step = algoCtx.opRounds[Op::kRecvRedCopy].postStep.step;

  // Skip check if it is not a forwarding round
  if (!isRecvFwd(algoCtx, step)) {
    return true;
  }

  // Check if the previous round used the same tmpSendBuf chunk has finished
  // so we can reuse in the forwarding send round
  int fwdRound = getRecvFwdSendRound(algoCtx, round);
  int prevRound = fwdRound - algoCtx.numChunks;
  int tmpChunkId = getTmpChunkId(algoCtx, fwdRound);

  bool done = algoCtx.opRounds[Op::kSendTrans].done > prevRound;
  if (done) {
    CLOGF_TRACE(
        COLL,
        "{} waited tmpChunkId {} algoCtx.numChunks {} prevRound {}",
        roundLogPrefix<Op::kRecvRedCopy>(round, step, algoCtx),
        tmpChunkId,
        algoCtx.numChunks,
        prevRound);
  }
  return done;
}

inline void progressRecvPostRedCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRecvRedCopy].post;
  int step = algoCtx.opRounds[Op::kRecvRedCopy].postStep.step;

  CLOGF_TRACE(
      COLL,
      "{} posted",
      roundLogPrefix<Op::kRecvRedCopy>(round, step, algoCtx));
  resource.recvRedCopySync->post(round);
}

inline bool progressRecvCheckRedCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRecvRedCopy].done;
  auto& opStep = algoCtx.opRounds[Op::kRecvRedCopy].doneStep;
  int tmpChunkId = getTmpChunkId(algoCtx, round);
  bool done = resource.recvRedCopySync->isComplete(round);

  if (done) {
    bool isRecvFwd_ = isRecvFwd(algoCtx, opStep.step);
    int fwdRound = isRecvFwd_ ? getRecvFwdSendRound(algoCtx, round) : -1;
    int tmpFwdChunkId = isRecvFwd_ ? getTmpChunkId(algoCtx, fwdRound) : -1;

    CLOGF_TRACE(
        COLL,
        "{} done: tmpChunkId {} fwdRound {} tmpFwdChunkId {} ",
        roundLogPrefix<Op::kRecvRedCopy>(round, opStep.step, algoCtx),
        tmpChunkId,
        fwdRound,
        tmpFwdChunkId);
  }
  return done;
}

inline void progressRecvPostRecvBuf(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& bufSyncSResps) {
  int round = algoCtx.opRounds[Op::kRecvRedCopy].done;
  int step = algoCtx.opRounds[Op::kRecvRedCopy].doneStep.step;
  int tmpChunkId = getTmpChunkId(algoCtx, round);

  CLOGF_TRACE(
      COLL,
      "{} posted tmpChunkId {}",
      roundLogPrefix<Op::kRecvRedCopy>(round, step, algoCtx),
      tmpChunkId);

  CtranMapperRequest* req;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->isendCtrl(args.leftRank, &req),
      resource.comm->logMetaData_);
  bufSyncSResps.at(round).reset(req);
}

inline void progressSend(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& dataSResps,
    std::vector<std::unique_ptr<CtranMapperRequest>>& bufSyncRResps) {
  // Try post copy to kernel if the send data is ready
  if (opReadyToPost<Op::kSendCopy>(algoCtx) &&
      progressSendCheckSendBuf(algoCtx)) {
    progressSendPostCopyKern(args, resource, algoCtx);
    opUpdatePost<Op::kSendCopy>(algoCtx);
  }

  // Check if any outstanding copy is done
  if (opHasPosted<Op::kSendCopy>(algoCtx) &&
      progressSendCheckCopyKern(args, resource, algoCtx)) {
    opUpdateDone<Op::kSendCopy>(algoCtx);
  }

  // Try post network transmission if the send data has been copied to tmpbuf
  if (opReadyToPost<Op::kSendTrans>(algoCtx)) {
    // Check if right neighbor has consumed the tmpRecvBuf chunk
    if (progressSendCheckRemRecvBuf(args, resource, algoCtx, bufSyncRResps)) {
      progressSendPostTrans(args, resource, algoCtx, dataSResps);
      opUpdatePost<Op::kSendTrans>(algoCtx);
    }
  }

  // Check if any outstanding transmission has been done
  progressSendCheckTrans(args, resource, algoCtx, dataSResps);
}

inline void progressRecv(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& bufSyncSResps,
    std::vector<std::unique_ptr<CtranMapperRequest>>& flushResps) {
  // Check if have received a chunk from left
  // Data receive doesn't need specific post, thus updating post & done
  // together
  if (opReadyToPost<Op::kRecvTrans>(algoCtx) &&
      progressRecvCheckTrans(args, resource, algoCtx)) {
    opUpdatePost<Op::kRecvTrans>(algoCtx);
    opUpdateDone<Op::kRecvTrans>(algoCtx);
  }

  // Check if any received chunk is ready to flush
  if (opReadyToPost<Op::kRecvFlush>(algoCtx)) {
    progressRecvPostFlush(args, resource, algoCtx, flushResps);
    opUpdatePost<Op::kRecvFlush>(algoCtx);
  }

  // Check if any outstanding flush is done
  if (opHasPosted<Op::kRecvFlush>(algoCtx)) {
    if (progressRecvCheckFlush(args, resource, algoCtx, flushResps)) {
      opUpdateDone<Op::kRecvFlush>(algoCtx);
    }
  }

  // Check if any received chunk is ready to reduce with local data
  if (opReadyToPost<Op::kRecvRedCopy>(algoCtx)) {
    int step = algoCtx.opRounds[Op::kRecvRedCopy].postStep.step;
    // Combine reduce and sendCopy.
    // ## When it is not a isRecvFwd round:
    // - For last step, we don't need sendCopy.
    // ## When it is a isRecvFwd round:
    // - Combine reduce and next step's sendCopy. Consequently, need check
    // sendBuf availability before reduce.
    if (!isRecvFwd(algoCtx, step) || progressRecvCheckSendBuf(algoCtx)) {
      progressRecvPostRedCopyKern(args, resource, algoCtx);
      opUpdatePost<Op::kRecvRedCopy>(algoCtx);
    }
  }

  // Check if any outstanding reduceCopy is done
  if (opHasPosted<Op::kRecvRedCopy>(algoCtx)) {
    if (progressRecvCheckRedCopyKern(args, resource, algoCtx)) {
      // Post buffer-ready sync after local reduce used the data.
      progressRecvPostRecvBuf(args, resource, algoCtx, bufSyncSResps);
      opUpdateDone<Op::kRecvRedCopy>(algoCtx);
    }
  }
}

// ===== Reverse direction progress functions for bi-directional AllGather =====

inline void prePostRevRecvRemRecvBuf(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revBufSyncRResps) {
  int totalRounds = algoCtx.opRounds[Op::kRevSendTrans].totalRounds;
  if (totalRounds == 0)
    return;

  // Pre-post recvCtrls to receive postRecvBuf sync from left neighbor
  revBufSyncRResps.resize(totalRounds);
  for (int round = 0; round < totalRounds; round++) {
    CtranMapperRequest* req;
    FB_COMMCHECKTHROW_EX(
        resource.comm->ctran_->mapper->irecvCtrl(args.leftRank, &req),
        resource.comm->logMetaData_);
    revBufSyncRResps.at(round).reset(req);
  }
}

inline bool progressRevSendCheckSendBuf(const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRevSendCopy].post;
  if (round < algoCtx.numChunks)
    return true;
  int prevRound = round - algoCtx.numChunks;
  return algoCtx.opRounds[Op::kRevSendTrans].done > prevRound;
}

inline void progressRevSendPostCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRevSendCopy].post;
  resource.revSendCopySync->post(round);
}

inline bool progressRevSendCheckCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRevSendCopy].done;
  return resource.revSendCopySync->isComplete(round);
}

inline bool progressRevSendCheckRemRecvBuf(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revBufSyncRResps) {
  int round = algoCtx.opRounds[Op::kRevSendTrans].post;
  int prevRound = round - algoCtx.numChunks;
  if (prevRound < 0)
    return true;

  auto& resp = revBufSyncRResps.at(prevRound);
  FB_CHECKTHROW_EX(
      resp != nullptr,
      resource.comm->logMetaData_,
      fmt::format(
          "revBufSyncRResps is not initialized at round {}", prevRound));
  bool isComplete = false;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->testRequest(resp.get(), &isComplete),
      resource.comm->logMetaData_);
  return isComplete;
}

inline void progressRevSendPostTrans(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revDataSResps) {
  int round = algoCtx.opRounds[Op::kRevSendTrans].post;
  auto& opStep = algoCtx.opRounds[Op::kRevSendTrans].postStep;

  int tmpChunkId = getTmpChunkId(algoCtx, round);
  auto chunkArg = getRevRoundArgs<Op::kRevSendTrans>(algoCtx, round, opStep);
  FB_CHECKTHROW_EX(
      chunkArg.numel > 0,
      resource.comm->logMetaData_,
      "Unexpected empty chunk in rev send");

  // iput to left neighbor's tmpRecvBufRev
  char* tmpRemoteRecvBufRev = reinterpret_cast<char*>(args.leftRemBufRev) +
      tmpChunkId * algoCtx.chunkSize;
  char* tmpSendBufRev = reinterpret_cast<char*>(resource.tmpSendBufRev) +
      tmpChunkId * algoCtx.chunkSize;

  static thread_local auto allReduceConfig =
      resource.comm->ctran_->algo->getCollToVcConfig(CollType::ALLREDUCE);

  CtranMapperRequest* req;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->iput(
          tmpSendBufRev,
          tmpRemoteRecvBufRev,
          chunkArg.numel * algoCtx.typeSize,
          args.leftRank,
          CtranMapperConfig{
              .memHdl_ = resource.tmpSendBufRevHdl,
              .remoteAccessKey_ = args.leftRemKeyRev,
              .notify_ = true,
              .ibConfig_ = allReduceConfig},
          &req),
      resource.comm->logMetaData_);
  revDataSResps.at(round).reset(req);
}

inline void progressRevSendCheckTrans(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revDataSResps) {
  int startRound = algoCtx.opRounds[Op::kRevSendTrans].done;
  int lastRound = algoCtx.opRounds[Op::kRevSendTrans].post;
  for (int r = startRound; r < lastRound; r++) {
    auto& resp = revDataSResps.at(r);
    if (resp) {
      bool isComplete = false;
      FB_COMMCHECKTHROW_EX(
          resource.comm->ctran_->mapper->testRequest(resp.get(), &isComplete),
          resource.comm->logMetaData_);
      if (isComplete) {
        opUpdateDone<Op::kRevSendTrans>(algoCtx);
      }
    }
  }
}

inline bool progressRevRecvCheckTrans(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  bool done = false;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->checkNotify(args.rightNotify.get(), &done),
      resource.comm->logMetaData_);
  return done;
}

inline void progressRevRecvPostFlush(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revFlushResps) {
  int round = algoCtx.opRounds[Op::kRevRecvFlush].post;
  int tmpChunkId = getTmpChunkId(algoCtx, round);
  char* tmpRecvBufRev = reinterpret_cast<char*>(resource.tmpRecvBufRev) +
      tmpChunkId * algoCtx.chunkSize;

  CtranMapperRequest* req = nullptr;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->iflush(
          tmpRecvBufRev, resource.tmpRecvBufRevHdl, &req),
      resource.comm->logMetaData_);
  revFlushResps.at(round).reset(req);
}

inline bool progressRevRecvCheckFlush(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revFlushResps) {
  int round = algoCtx.opRounds[Op::kRevRecvFlush].done;
  auto& resp = revFlushResps.at(round);
  FB_CHECKTHROW_EX(
      resp != nullptr,
      resource.comm->logMetaData_,
      fmt::format("Rev flush resp is not initialized at round {}", round));
  bool isComplete = false;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->testRequest(resp.get(), &isComplete),
      resource.comm->logMetaData_);
  return isComplete;
}

inline bool progressRevRecvCheckSendBuf(const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRevRecvCopy].post;
  int step = algoCtx.opRounds[Op::kRevRecvCopy].postStep.step;
  if (!isRevRecvFwd(algoCtx, step))
    return true;

  // The rev recv copy also copies to tmpSendBufRev for forwarding.
  // The corresponding send round is firstRevShardChunks + round.
  int firstRevShardChunks = algoCtx.opRounds[Op::kRevSendCopy].totalRounds;
  int fwdRound = firstRevShardChunks + round;
  int prevRound = fwdRound - algoCtx.numChunks;
  return algoCtx.opRounds[Op::kRevSendTrans].done > prevRound;
}

inline void progressRevRecvPostCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRevRecvCopy].post;
  resource.revRecvCopySync->post(round);
}

inline bool progressRevRecvCheckCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx) {
  int round = algoCtx.opRounds[Op::kRevRecvCopy].done;
  return resource.revRecvCopySync->isComplete(round);
}

inline void progressRevRecvPostRecvBuf(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revBufSyncSResps) {
  int round = algoCtx.opRounds[Op::kRevRecvCopy].done;

  CtranMapperRequest* req;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->isendCtrl(args.rightRank, &req),
      resource.comm->logMetaData_);
  revBufSyncSResps.at(round).reset(req);
}

inline void progressRevSend(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revDataSResps,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revBufSyncRResps) {
  if (algoCtx.numRevAgSteps == 0)
    return;

  if (opReadyToPost<Op::kRevSendCopy>(algoCtx) &&
      progressRevSendCheckSendBuf(algoCtx)) {
    progressRevSendPostCopyKern(args, resource, algoCtx);
    opUpdatePost<Op::kRevSendCopy>(algoCtx);
  }

  if (opHasPosted<Op::kRevSendCopy>(algoCtx) &&
      progressRevSendCheckCopyKern(args, resource, algoCtx)) {
    opUpdateDone<Op::kRevSendCopy>(algoCtx);
  }

  if (opReadyToPost<Op::kRevSendTrans>(algoCtx)) {
    if (progressRevSendCheckRemRecvBuf(
            args, resource, algoCtx, revBufSyncRResps)) {
      progressRevSendPostTrans(args, resource, algoCtx, revDataSResps);
      opUpdatePost<Op::kRevSendTrans>(algoCtx);
    }
  }

  progressRevSendCheckTrans(args, resource, algoCtx, revDataSResps);
}

inline void progressRevRecv(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revBufSyncSResps,
    std::vector<std::unique_ptr<CtranMapperRequest>>& revFlushResps) {
  if (algoCtx.numRevAgSteps == 0)
    return;

  if (opReadyToPost<Op::kRevRecvTrans>(algoCtx) &&
      progressRevRecvCheckTrans(args, resource, algoCtx)) {
    opUpdatePost<Op::kRevRecvTrans>(algoCtx);
    opUpdateDone<Op::kRevRecvTrans>(algoCtx);
  }

  if (opReadyToPost<Op::kRevRecvFlush>(algoCtx)) {
    progressRevRecvPostFlush(args, resource, algoCtx, revFlushResps);
    opUpdatePost<Op::kRevRecvFlush>(algoCtx);
  }

  if (opHasPosted<Op::kRevRecvFlush>(algoCtx)) {
    if (progressRevRecvCheckFlush(args, resource, algoCtx, revFlushResps)) {
      opUpdateDone<Op::kRevRecvFlush>(algoCtx);
    }
  }

  if (opReadyToPost<Op::kRevRecvCopy>(algoCtx)) {
    int step = algoCtx.opRounds[Op::kRevRecvCopy].postStep.step;
    if (!isRevRecvFwd(algoCtx, step) || progressRevRecvCheckSendBuf(algoCtx)) {
      progressRevRecvPostCopyKern(args, resource, algoCtx);
      opUpdatePost<Op::kRevRecvCopy>(algoCtx);
    }
  }

  if (opHasPosted<Op::kRevRecvCopy>(algoCtx)) {
    if (progressRevRecvCheckCopyKern(args, resource, algoCtx)) {
      progressRevRecvPostRecvBuf(args, resource, algoCtx, revBufSyncSResps);
      opUpdateDone<Op::kRevRecvCopy>(algoCtx);
    }
  }
}

// ===== End reverse direction progress functions =====

inline int waitAllResps(
    std::vector<std::unique_ptr<CtranMapperRequest>>& reqs,
    CtranComm* comm,
    const std::string& ctx) {
  int numComplete = 0;
  for (auto& req : reqs) {
    if (req) {
      numComplete++;
      FB_COMMCHECKTHROW_EX(
          comm->ctran_->mapper->waitRequest(req.get()), comm->logMetaData_);
    }
  }
  return numComplete;
}

inline void updatePartitionCtxHost(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx) {
  if (args.enableBidirAg) {
    updatePartitionCtx<true>(algoCtx);
  } else {
    updatePartitionCtx<false>(algoCtx);
  }
  if (algoCtx.partition > 0) {
    // Sync with kernel to start the new partition if not the first one.
    resource.partitionSync->post(algoCtx.partition);
  }
}

inline void exchangePeerTmpBufs(
    CtranComm* comm,
    ctran::allreduce::ring::HostArgs& args) {
  // complete resource setup for ones needing the EpochLock
  // Capture both right (for forward) and left (for reverse) remote buf info
  if (comm->statex_->rank() % 2 == 0) {
    std::tie(args.rightRemBuf, args.rightRemKey) =
        comm->ctran_->algo->getRemoteTmpBufInfo(args.rightRank);
    std::tie(args.leftRemBufRev, args.leftRemKeyRev) =
        comm->ctran_->algo->getRemoteTmpBufInfo(args.leftRank);
  } else {
    std::tie(args.leftRemBufRev, args.leftRemKeyRev) =
        comm->ctran_->algo->getRemoteTmpBufInfo(args.leftRank);
    std::tie(args.rightRemBuf, args.rightRemKey) =
        comm->ctran_->algo->getRemoteTmpBufInfo(args.rightRank);
  }
}

inline commResult_t completeHostResourceSetup(
    CtranComm* comm,
    ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource) {
  exchangePeerTmpBufs(comm, args);

  // Forward: notifications from left on tmpRecvBuf
  args.leftNotify.reset(new CtranMapperNotify());
  FB_COMMCHECK(comm->ctran_->mapper->initNotify(
      args.leftRank, resource.tmpRecvBufHdl, args.leftNotify.get()));

  size_t offsetRingTmpRecv = comm->ctran_->algo->getTmpBufOffset(
      CtranAlgo::TmpbufType::RING_TMP_RECV_BUF);
  args.rightRemBuf = (char*)args.rightRemBuf + offsetRingTmpRecv;

  // Reverse: notifications from right on tmpRecvBufRev
  args.rightNotify.reset(new CtranMapperNotify());
  FB_COMMCHECK(comm->ctran_->mapper->initNotify(
      args.rightRank, resource.tmpRecvBufRevHdl, args.rightNotify.get()));

  // Point leftRemBufRev to left neighbor's tmpRecvBufRev segment
  size_t offsetRingTmpRecvRev = comm->ctran_->algo->getTmpBufOffset(
      CtranAlgo::TmpbufType::RING_TMP_RECV_BUF_REV);
  args.leftRemBufRev = (char*)args.leftRemBufRev + offsetRingTmpRecvRev;

  return commSuccess;
}

} // namespace

#define HOST_ABORT()                                                \
  if (comm->testAbort()) {                                          \
    throw ctran::utils::Exception("comm aborted", commRemoteError); \
  }

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  FB_CHECKTHROW_EX_NOCOMM(
      opGroup.size() == 1, "ctring opGroup expected exactly one op");
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = opGroup.front()->comm_;
  CtranAlgoLogger logger(allReduceAlgoName(myAlgo), op->opCount, comm);

  using HostArgs = ctran::allreduce::ring::HostArgs;
  using HostResource = ctran::allreduce::ring::HostResource;
  auto argsGuard = std::unique_ptr<HostArgs>(
      reinterpret_cast<HostArgs*>(op->allreduce.args));
  auto resourceGuard = std::unique_ptr<HostResource>(
      reinterpret_cast<HostResource*>(op->allreduce.resource));
  auto& args = *argsGuard;
  auto& resource = *resourceGuard;

  FB_COMMCHECK(completeHostResourceSetup(comm, args, resource));

  // setup algoCtx
  AlgoContext algoCtx = {
      .numElements = op->allreduce.count,
      .rank = op->comm_->statex_->rank(),
      .nRanks = op->comm_->statex_->nRanks(),
      .chunkSize = resource.chunkSize,
      .numChunks = resource.numChunks,
      .minShardSize = args.minShardSize,
      .typeSize = static_cast<size_t>(commTypeSize(op->allreduce.datatype)),
  };
  setupAlgoCtxImpl(algoCtx);

  // Forward direction request vectors
  std::vector<std::unique_ptr<CtranMapperRequest>> dataSResps;
  std::vector<std::unique_ptr<CtranMapperRequest>> bufSyncSResps;
  std::vector<std::unique_ptr<CtranMapperRequest>> bufSyncRResps;
  std::vector<std::unique_ptr<CtranMapperRequest>> flushResps;

  // Reverse direction request vectors
  std::vector<std::unique_ptr<CtranMapperRequest>> revDataSResps;
  std::vector<std::unique_ptr<CtranMapperRequest>> revBufSyncSResps;
  std::vector<std::unique_ptr<CtranMapperRequest>> revBufSyncRResps;
  std::vector<std::unique_ptr<CtranMapperRequest>> revFlushResps;

  while (algoCtx.partitionOffset < algoCtx.numElements) {
    updatePartitionCtxHost(args, resource, algoCtx);
    CLOGF_TRACE(
        COLL,
        ALGO_CXT_LOG_FMT_HOST,
        ALGO_CXT_LOG_FIELDS(algoCtx, args.numBlocks));

    int totalSendTrans = algoCtx.opRounds[Op::kSendTrans].totalRounds;
    int totalRecvTrans = algoCtx.opRounds[Op::kRecvTrans].totalRounds;
    dataSResps.resize(totalSendTrans);
    bufSyncSResps.resize(totalRecvTrans);
    bufSyncRResps.resize(totalSendTrans);
    flushResps.resize(totalRecvTrans);

    int totalRevSendTrans = algoCtx.opRounds[Op::kRevSendTrans].totalRounds;
    int totalRevRecvTrans = algoCtx.opRounds[Op::kRevRecvTrans].totalRounds;
    revDataSResps.resize(totalRevSendTrans);
    revBufSyncSResps.resize(totalRevRecvTrans);
    revBufSyncRResps.resize(totalRevSendTrans);
    revFlushResps.resize(totalRevRecvTrans);

    prePostRecvRemRecvBuf(args, resource, algoCtx, bufSyncRResps);
    prePostRevRecvRemRecvBuf(args, resource, algoCtx, revBufSyncRResps);

    // Ring main loop: forward + reverse directions
    auto fwdNotDone = [&]() {
      return algoCtx.opRounds[Op::kSendTrans].done <
          algoCtx.opRounds[Op::kSendTrans].totalRounds ||
          algoCtx.opRounds[Op::kRecvRedCopy].done <
          algoCtx.opRounds[Op::kRecvRedCopy].totalRounds;
    };
    auto revNotDone = [&]() {
      return algoCtx.opRounds[Op::kRevSendTrans].done <
          algoCtx.opRounds[Op::kRevSendTrans].totalRounds ||
          algoCtx.opRounds[Op::kRevRecvCopy].done <
          algoCtx.opRounds[Op::kRevRecvCopy].totalRounds;
    };

    while (fwdNotDone() || revNotDone()) {
      if (op->allreduce.datatype == commInt8 ||
          op->allreduce.datatype == commChar ||
          op->allreduce.datatype == commUint8 ||
          op->allreduce.datatype == commInt32 ||
          op->allreduce.datatype == commInt ||
          op->allreduce.datatype == commUint32 ||
          op->allreduce.datatype == commInt64 ||
          op->allreduce.datatype == commUint64 ||
          op->allreduce.datatype == commFloat16 ||
          op->allreduce.datatype == commHalf ||
          op->allreduce.datatype == commFloat32 ||
          op->allreduce.datatype == commFloat ||
          op->allreduce.datatype == commFloat64 ||
          op->allreduce.datatype == commDouble) {
        // TODO: enable other data types
      } else {
        throw ctran::utils::Exception(
            fmt::format("Unsupported data type {}", op->allreduce.datatype),
            commInvalidArgument);
      }
      progressSend(args, resource, algoCtx, dataSResps, bufSyncRResps);
      progressRecv(args, resource, algoCtx, bufSyncSResps, flushResps);
      progressRevSend(args, resource, algoCtx, revDataSResps, revBufSyncRResps);
      progressRevRecv(args, resource, algoCtx, revBufSyncSResps, revFlushResps);
      HOST_ABORT();
    }

    // Release any remaining resps before moving to next partition
    waitAllResps(dataSResps, comm, "wait final dataSResps");
    waitAllResps(bufSyncSResps, comm, "wait final bufSyncSResps");
    waitAllResps(bufSyncRResps, comm, "wait final bufSyncRResps");
    waitAllResps(revDataSResps, comm, "wait final revDataSResps");
    waitAllResps(revBufSyncSResps, comm, "wait final revBufSyncSResps");
    waitAllResps(revBufSyncRResps, comm, "wait final revBufSyncRResps");

    // Reset flags for next partition to reuse
    resource.sendCopySync->resetStatus();
    resource.recvRedCopySync->resetStatus();
    resource.revSendCopySync->resetStatus();
    resource.revRecvCopySync->resetStatus();

    updatePartitionDone(algoCtx);
    HOST_ABORT();
  } // end of partition loop

  // Reset flags for next allreduce to reuse
  resource.sendCopySync->reset();
  resource.recvRedCopySync->reset();
  resource.partitionSync->reset();
  resource.revSendCopySync->reset();
  resource.revRecvCopySync->reset();

  return commSuccess;
}

commResult_t getNumBlocksAndThreads(
    int* numBlocks,
    int* numThreads,
    GpuArch* arch,
    const void* func) {
  FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
      numBlocks,
      numThreads,
      func,
      0 /* dynamicSMemSize */,
      0 /* blockSizeLimit */));

  *arch = GpuArch::Default;
  int cudaDev = 0;
  FB_CUDACHECK(cudaGetDevice(&cudaDev));
  int smMajor = 0;
  FB_CUDACHECK(cudaDeviceGetAttribute(
      &smMajor, cudaDevAttrComputeCapabilityMajor, cudaDev));
  if (smMajor < 10) {
    *arch = GpuArch::Hopper;
  }

  return commSuccess;
}

commResult_t getPipelineConfiguration(
    size_t messageBytes,
    int nRanks,
    const void* func,
    int* numBlocks,
    int* numThreads,
    size_t* pipelineNumChunks,
    size_t* pipelineChunkSize,
    bool log_decision,
    size_t typeSize) {
  auto arch = ctran::allreduce::ring::GpuArch::Default;
  int cudaOccupancyNumBlocks, cudaOccupancyBlockSize;
  FB_COMMCHECK(
      ctran::allreduce::ring::getNumBlocksAndThreads(
          &cudaOccupancyNumBlocks, &cudaOccupancyBlockSize, &arch, func));

  if (log_decision) {
    static std::once_flag logFlag;
    std::call_once(logFlag, [&] {
      ctran::allreduce::ring::logAutoTuneDecisions(
          nRanks,
          cudaOccupancyNumBlocks,
          cudaOccupancyBlockSize,
          typeSize,
          arch);
    });
  }

  auto params = ctran::allreduce::ring::getAutoTunedParams(
      messageBytes,
      nRanks,
      cudaOccupancyNumBlocks,
      cudaOccupancyBlockSize,
      typeSize,
      arch);
  *pipelineChunkSize = params.pipeline.chunkSize;
  *pipelineNumChunks = params.pipeline.numChunks;
  *numBlocks = params.block.numBlocks;
  *numThreads = params.block.blockSize;

  return commSuccess;
}

} // namespace ctran::allreduce::ring

commResult_t ctranAllReduceRing(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout) {
  // Check for minimum message size requirement for ctring algorithm.
  // The ctring algorithm uses a ring-based approach that shards data across all
  // ranks. Each rank must have at least one element in its shard to avoid empty
  // chunk transfers that can lead to synchronization deadlocks. Therefore, we
  // need at least nRanks elements.
  const auto& statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();
  const auto rank = statex->rank();
  const size_t typeSize = static_cast<size_t>(commTypeSize(datatype));
  const size_t minRequiredElements = nRanks;
  const size_t minRequiredBytes = minRequiredElements * typeSize;

  if (count < minRequiredElements) {
    std::string errorMsg = fmt::format(
        "ctring algorithm requires at least {} elements ({} bytes) for {} ranks, "
        "but rank {} got {} elements ({} bytes) with datatype size={} bytes. "
        "Each rank needs at least one element per shard. "
        "Please use a larger message size or a different allreduce algorithm (e.g., ctdirect).",
        minRequiredElements,
        minRequiredBytes,
        nRanks,
        rank,
        count,
        count * typeSize,
        typeSize);
    CLOGF(ERR, "{}", errorMsg);
    throw ctran::utils::Exception(errorMsg, commInvalidArgument);
  }

  auto opCount = comm->ctran_->getOpCount();
  CTRAN_REDCOLL_INFO(
      allReduceAlgoName(ctran::allreduce::ring::myAlgo),
      sendbuff,
      recvbuff,
      count,
      datatype,
      redOp,
      -1,
      comm,
      stream);

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;

  const size_t messageBytes = count * typeSize;

  // Select kernel based on bi-directional AG CVAR and message size
  const bool enableBidirAg =
      ctran::allreduce::ring::shouldEnableBidirAg(messageBytes);
  const auto& typeToFunc = enableBidirAg ? typeToFuncBidir : typeToFuncSimple;

  FB_CHECKTHROW_EX(
      typeToFunc.contains(std::make_pair(datatype, redOp)),
      comm->logMetaData_,
      fmt::format(
          "typeToFunc does not contain datatype {} with op {}",
          datatype,
          redOp));
  const void* func = typeToFunc.at(std::make_pair(datatype, redOp));

  int numBlocks = 0;
  int numThreads = 0;
  size_t pipelineChunkSize = 0;
  size_t pipelineNumChunks = 0;

  FB_COMMCHECK(
      ctran::allreduce::ring::getPipelineConfiguration(
          messageBytes,
          nRanks,
          func,
          &numBlocks,
          &numThreads,
          &pipelineNumChunks,
          &pipelineChunkSize,
          /*log_decision=*/rank == 0,
          typeSize));

  FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());

  // construct op

  // host side
  op = std::make_unique<OpElem>(OpElem::opType::ALLREDUCE, comm, opCount);
  op->allreduce.sendbuff = sendbuff;
  op->allreduce.recvbuff = recvbuff;
  op->allreduce.count = count;
  op->allreduce.datatype = datatype;
  op->allreduce.op = redOp;

  auto* hostResource = new ctran::allreduce::ring::HostResource();
  op->allreduce.resource = hostResource;
  hostResource->comm = comm;
  std::vector<ctran::algos::GpeKernelSync*> gpeKernelSyncs;
  // 3 forward (sendCopy, recvRedCopy, partition) + 2 reverse (revSendCopy,
  // revRecvCopy)
  constexpr size_t kAllReduceRingNumSyncs = 5;
  FB_COMMCHECK(comm->ctran_->gpe->allocGpeKernelSyncs(
      kAllReduceRingNumSyncs, numBlocks, gpeKernelSyncs));
  FB_CHECKTHROW_EX(
      gpeKernelSyncs.size() == kAllReduceRingNumSyncs,
      comm->logMetaData_,
      "Failed to allocate GpeKernelSync");
  hostResource->sendCopySync = gpeKernelSyncs[0];
  hostResource->recvRedCopySync = gpeKernelSyncs[1];
  hostResource->partitionSync = gpeKernelSyncs[2];
  hostResource->revSendCopySync = gpeKernelSyncs[3];
  hostResource->revRecvCopySync = gpeKernelSyncs[4];
  hostResource->chunkSize = pipelineChunkSize;
  hostResource->numChunks = pipelineNumChunks;

  std::tie(hostResource->tmpSendBuf, hostResource->tmpSendBufHdl) =
      comm->ctran_->algo->getTmpBufInfo(
          CtranAlgo::TmpbufType::RING_TMP_SEND_BUF);
  std::tie(hostResource->tmpRecvBuf, hostResource->tmpRecvBufHdl) =
      comm->ctran_->algo->getTmpBufInfo(
          CtranAlgo::TmpbufType::RING_TMP_RECV_BUF);
  std::tie(hostResource->tmpSendBufRev, hostResource->tmpSendBufRevHdl) =
      comm->ctran_->algo->getTmpBufInfo(
          CtranAlgo::TmpbufType::RING_TMP_SEND_BUF_REV);
  std::tie(hostResource->tmpRecvBufRev, hostResource->tmpRecvBufRevHdl) =
      comm->ctran_->algo->getTmpBufInfo(
          CtranAlgo::TmpbufType::RING_TMP_RECV_BUF_REV);
  CLOGF(
      DBG,
      "AutoTune: {} blocks of {} threads, tmpbuf {} x {} chunks, bidirAg={}",
      numBlocks,
      numThreads,
      hostResource->numChunks,
      hostResource->chunkSize,
      enableBidirAg);

  auto* hostArgs = new ctran::allreduce::ring::HostArgs();
  op->allreduce.args = hostArgs;
  hostArgs->rank = rank;
  hostArgs->leftRank = (rank - 1 + nRanks) % nRanks;
  hostArgs->rightRank = (rank + 1) % nRanks;
  hostArgs->minShardSize = NCCL_CTRAN_ALLREDUCE_RING_MIN_SHARD_SIZE;
  hostArgs->numBlocks = numBlocks;
  hostArgs->numThreads = numThreads;
  hostArgs->enableBidirAg = enableBidirAg;
  // rightRemBuf, rightRemKey, leftNotify init from gpe thread for EpochLock

  opGroup.push_back(std::move(op));

  // device side
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLREDUCE,
      stream,
      allReduceAlgoName(ctran::allreduce::ring::myAlgo),
      opCount);
  config.numBlocks = numBlocks;
  config.numThreads = numThreads;
  config.args.devState_d = comm->ctran_->algo->getDevState();
  ctran::allreduce::ring::KernArgs kernArgs{
      .sendbuff = sendbuff,
      .recvbuff = recvbuff,
      .datatype = datatype,
      .redOp = redOp,
      .count = count,
      .chunkSize = hostResource->chunkSize,
      .numChunks = hostResource->numChunks,
      .minShardSize = hostArgs->minShardSize,
      .sendCopySync = hostResource->sendCopySync,
      .recvRedCopySync = hostResource->recvRedCopySync,
      .partitionSync = hostResource->partitionSync,
      .tmpSendBuf = hostResource->tmpSendBuf,
      .tmpRecvBuf = hostResource->tmpRecvBuf,
      .revSendCopySync = hostResource->revSendCopySync,
      .revRecvCopySync = hostResource->revRecvCopySync,
      .tmpSendBufRev = hostResource->tmpSendBufRev,
      .tmpRecvBufRev = hostResource->tmpRecvBufRev,
  };
  // Used only in gpe->submit, copied as a Kernel Launch Arg.
  config.algoArgs = &kernArgs;

  // TODO: delete, this is for colltrace: Find a way to make colltrace use
  // settings from above. Currently colltrace cannot fetch information from
  // ctran::allreduce::ring::KernArgs yet
  config.args.collective.allreduce.sendbuff = kernArgs.sendbuff;
  config.args.collective.allreduce.recvbuff = kernArgs.recvbuff;
  config.args.collective.allreduce.redOp = kernArgs.redOp;
  config.args.collective.allreduce.count = kernArgs.count;
  config.args.collective.allreduce.datatype = kernArgs.datatype;

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup), ctran::allreduce::ring::impl, config, func, timeout));

  return commSuccess;
}
