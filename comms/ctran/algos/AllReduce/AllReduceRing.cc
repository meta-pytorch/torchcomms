// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <chrono>

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
#include "comms/ctran/algos/perftrace/Record.h"
#include "comms/ctran/algos/perftrace/Tracer.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

CTRAN_DATATYPE_REDOP_TO_FUNC_MAPPER(typeToFunc, ncclKernelAllReduceCtranRing);

namespace ctran::allreduce::ring {

struct HostArgs {
  int32_t rank{-1};
  int32_t leftRank{-1};
  int32_t rightRank{-1};

  size_t minShardSize{0};

  unsigned int numBlocks{0};
  unsigned int numThreads{0};

  // Remote receive buffer on right
  void* rightRemBuf{nullptr};
  CtranMapperRemoteAccessKey rightRemKey;

  // Local receive buffer notify from left
  std::unique_ptr<CtranMapperNotify> leftNotify{nullptr};
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
    std::vector<std::unique_ptr<CtranMapperRequest>>& dataSResps,
    perftrace::Record* ts) {
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
        if (ts) {
          ts->endInterval(
              "SendTrans_mapper_rdma", algoCtx.partitionStartSendRounds + r);
        }
        opUpdateDone<Op::kSendTrans>(algoCtx);
      }
    }
  }
}

inline void progressSendPostCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    perftrace::Record* ts) {
  int round = algoCtx.opRounds[Op::kSendCopy].post;
  int step = algoCtx.opRounds[Op::kSendCopy].postStep.step;
  CLOGF_TRACE(
      COLL, "{} posted", roundLogPrefix<Op::kSendCopy>(round, step, algoCtx));
  resource.sendCopySync->post(round);
  if (ts) {
    int seqNum = algoCtx.partitionStartSendRounds + round;
    std::map<std::string, std::string> md = {
        {"partition", std::to_string(algoCtx.partition)},
        {"round", std::to_string(round)},
        {"step", std::to_string(step)}};
    ts->startInterval(
        "SendCopy_kernelsync_device", seqNum, args.rightRank, std::move(md));
  }
}

inline bool progressSendCheckCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    perftrace::Record* ts) {
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
    if (ts) {
      ts->endInterval(
          "SendCopy_kernelsync_device",
          algoCtx.partitionStartSendRounds + round);
    }
  }
  return done;
}

inline void progressSendPostTrans(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& dataSResps,
    perftrace::Record* ts) {
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
  if (ts) {
    int seqNum = algoCtx.partitionStartSendRounds + round;
    std::map<std::string, std::string> md = {
        {"partition", std::to_string(algoCtx.partition)},
        {"round", std::to_string(round)},
        {"step", std::to_string(step)},
        {"bytes", std::to_string(chunkArg.numel * algoCtx.typeSize)}};
    ts->startInterval(
        "SendTrans_mapper_rdma", seqNum, args.rightRank, std::move(md));
  }
}

inline bool progressRecvCheckTrans(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    perftrace::Record* ts) {
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
    if (ts) {
      int seqNum = algoCtx.partitionStartRecvRounds + round;
      std::map<std::string, std::string> md = {
          {"partition", std::to_string(algoCtx.partition)},
          {"round", std::to_string(round)},
          {"step", std::to_string(step)},
          {"bytes", std::to_string(chunkArg.numel * algoCtx.typeSize)}};
      ts->addPoint(
          "RecvTrans_mapper_rdma", seqNum, args.leftRank, std::move(md));
    }
  }
  return done;
}

inline void progressRecvPostFlush(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& flushResps,
    perftrace::Record* ts) {
  int round = algoCtx.opRounds[Op::kRecvFlush].post;
  int step = algoCtx.opRounds[Op::kRecvFlush].postStep.step;

  int tmpChunkId = getTmpChunkId(algoCtx, round);
  char* tmpRecvBuf = reinterpret_cast<char*>(resource.tmpRecvBuf) +
      tmpChunkId * algoCtx.chunkSize;

  CtranMapperRequest* req;
  FB_COMMCHECKTHROW_EX(
      resource.comm->ctran_->mapper->iflush(
          tmpRecvBuf, resource.tmpRecvBufHdl, &req),
      resource.comm->logMetaData_);
  flushResps.at(round).reset(req);
  if (ts) {
    int seqNum = algoCtx.partitionStartRecvRounds + round;
    std::map<std::string, std::string> md = {
        {"partition", std::to_string(algoCtx.partition)},
        {"round", std::to_string(round)},
        {"step", std::to_string(step)}};
    ts->startInterval(
        "RecvFlush_mapper_rdma", seqNum, args.leftRank, std::move(md));
  }
}

inline bool progressRecvCheckFlush(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& flushResps,
    perftrace::Record* ts) {
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
    if (ts) {
      ts->endInterval(
          "RecvFlush_mapper_rdma", algoCtx.partitionStartRecvRounds + round);
    }
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
    const AlgoContext& algoCtx,
    perftrace::Record* ts) {
  int round = algoCtx.opRounds[Op::kRecvRedCopy].post;
  int step = algoCtx.opRounds[Op::kRecvRedCopy].postStep.step;

  CLOGF_TRACE(
      COLL,
      "{} posted",
      roundLogPrefix<Op::kRecvRedCopy>(round, step, algoCtx));
  resource.recvRedCopySync->post(round);
  if (ts) {
    int seqNum = algoCtx.partitionStartRecvRounds + round;
    std::map<std::string, std::string> md = {
        {"partition", std::to_string(algoCtx.partition)},
        {"round", std::to_string(round)},
        {"step", std::to_string(step)}};
    ts->startInterval(
        "RecvRedCopy_kernelsync_device", seqNum, args.leftRank, std::move(md));
  }
}

inline bool progressRecvCheckRedCopyKern(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    const AlgoContext& algoCtx,
    perftrace::Record* ts) {
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
    if (ts) {
      ts->endInterval(
          "RecvRedCopy_kernelsync_device",
          algoCtx.partitionStartRecvRounds + round);
    }
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
    std::vector<std::unique_ptr<CtranMapperRequest>>& bufSyncRResps,
    perftrace::Record* ts) {
  // Try post copy to kernel if the send data is ready
  if (opReadyToPost<Op::kSendCopy>(algoCtx) &&
      progressSendCheckSendBuf(algoCtx)) {
    progressSendPostCopyKern(args, resource, algoCtx, ts);
    opUpdatePost<Op::kSendCopy>(algoCtx);
  }

  // Check if any outstanding copy is done
  if (opHasPosted<Op::kSendCopy>(algoCtx) &&
      progressSendCheckCopyKern(args, resource, algoCtx, ts)) {
    opUpdateDone<Op::kSendCopy>(algoCtx);
  }

  // Try post network transmission if the send data has been copied to tmpbuf
  if (opReadyToPost<Op::kSendTrans>(algoCtx)) {
    // Check if right neighbor has consumed the tmpRecvBuf chunk
    if (progressSendCheckRemRecvBuf(args, resource, algoCtx, bufSyncRResps)) {
      progressSendPostTrans(args, resource, algoCtx, dataSResps, ts);
      opUpdatePost<Op::kSendTrans>(algoCtx);
    }
  }

  // Check if any outstanding transmission has been done
  progressSendCheckTrans(args, resource, algoCtx, dataSResps, ts);
}

inline void progressRecv(
    const ctran::allreduce::ring::HostArgs& args,
    ctran::allreduce::ring::HostResource& resource,
    AlgoContext& algoCtx,
    std::vector<std::unique_ptr<CtranMapperRequest>>& bufSyncSResps,
    std::vector<std::unique_ptr<CtranMapperRequest>>& flushResps,
    perftrace::Record* ts) {
  // Check if have received a chunk from left
  // Data receive doesn't need specific post, thus updating post & done
  // together
  if (opReadyToPost<Op::kRecvTrans>(algoCtx) &&
      progressRecvCheckTrans(args, resource, algoCtx, ts)) {
    opUpdatePost<Op::kRecvTrans>(algoCtx);
    opUpdateDone<Op::kRecvTrans>(algoCtx);
  }

  // Check if any received chunk is ready to flush
  if (opReadyToPost<Op::kRecvFlush>(algoCtx)) {
    progressRecvPostFlush(args, resource, algoCtx, flushResps, ts);
    opUpdatePost<Op::kRecvFlush>(algoCtx);
  }

  // Check if any outstanding flush is done
  if (opHasPosted<Op::kRecvFlush>(algoCtx)) {
    if (progressRecvCheckFlush(args, resource, algoCtx, flushResps, ts)) {
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
      progressRecvPostRedCopyKern(args, resource, algoCtx, ts);
      opUpdatePost<Op::kRecvRedCopy>(algoCtx);
    }
  }

  // Check if any outstanding reduceCopy is done
  if (opHasPosted<Op::kRecvRedCopy>(algoCtx)) {
    if (progressRecvCheckRedCopyKern(args, resource, algoCtx, ts)) {
      // Post buffer-ready sync after local reduce used the data.
      progressRecvPostRecvBuf(args, resource, algoCtx, bufSyncSResps);
      opUpdateDone<Op::kRecvRedCopy>(algoCtx);
    }
  }
}

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
  updatePartitionCtx(algoCtx);
  if (algoCtx.partition > 0) {
    // Sync with kernel to start the new partition if not the first one.
    resource.partitionSync->post(algoCtx.partition);
  }
}

inline void exchangePeerTmpBufs(
    CtranComm* comm,
    ctran::allreduce::ring::HostArgs& args) {
  // complete resource setup for ones needing the EpochLock
  if (comm->statex_->rank() % 2 == 0) {
    std::tie(args.rightRemBuf, args.rightRemKey) =
        comm->ctran_->algo->getRemoteTmpBufInfo(args.rightRank);
    comm->ctran_->algo->getRemoteTmpBufInfo(args.leftRank);
  } else {
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

  args.leftNotify.reset(new CtranMapperNotify());
  FB_COMMCHECK(comm->ctran_->mapper->initNotify(
      args.leftRank, resource.tmpRecvBufHdl, args.leftNotify.get()));

  size_t offsetRingTmpRecv = comm->ctran_->algo->getTmpBufOffset(
      CtranAlgo::TmpbufType::RING_TMP_RECV_BUF);
  args.rightRemBuf = (char*)args.rightRemBuf + offsetRingTmpRecv;

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

  // Perftrace: conditionally create tracer/record based on CVARs
  const bool shouldTrace = NCCL_CTRAN_ENABLE_PERFTRACE &&
      op->opCount >=
          static_cast<uint64_t>(NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_SKIP_OPS) &&
      (NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_NUM_OPS == 0 ||
       op->opCount < static_cast<uint64_t>(
                         NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_SKIP_OPS +
                         NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_NUM_OPS));

  std::unique_ptr<perftrace::Tracer> tracer;
  std::unique_ptr<perftrace::Record> ts;
  if (shouldTrace) {
    tracer = std::make_unique<perftrace::Tracer>(algoCtx.rank);
    ts = std::make_unique<perftrace::Record>("allReduceRing", algoCtx.rank);
    ts->addMetadata("opCount", std::to_string(op->opCount));
    ts->addMetadata("count", std::to_string(op->allreduce.count));
    ts->addMetadata(
        "datatype", std::to_string(static_cast<int>(op->allreduce.datatype)));
    ts->addMetadata("nRanks", std::to_string(algoCtx.nRanks));
    ts->addMetadata("rank", std::to_string(algoCtx.rank));
    ts->addMetadata("leftRank", std::to_string(args.leftRank));
    ts->addMetadata("rightRank", std::to_string(args.rightRank));
    ts->addMetadata("chunkSize", std::to_string(resource.chunkSize));
    ts->addMetadata("numChunks", std::to_string(resource.numChunks));
  }
  perftrace::Record* tsPtr = ts.get();

  std::vector<std::unique_ptr<CtranMapperRequest>> dataSResps;
  // - Responses for sync control send to left neighbor. Need wait for a
  // previous response to finish when want to post the same recvBuf chunk.
  std::vector<std::unique_ptr<CtranMapperRequest>> bufSyncSResps;
  // - Responses for sync control recv from right neighbor. Need wait for a
  // previous recvCtrl to finish when want to post irecvCtrl for the same
  // remote recvBuf chunk.
  std::vector<std::unique_ptr<CtranMapperRequest>> bufSyncRResps;
  // - Responses for local flush received data.
  std::vector<std::unique_ptr<CtranMapperRequest>> flushResps;

  // Split data into 1~N partitions, each partition is up to chunkSize *
  // numChunks * contextSize bytes. For each partition, we perform a
  // ReduceScatter phase and an AllGather phase to complete the transfer before
  // we move to next partition.
  while (algoCtx.partitionOffset < algoCtx.numElements) {
    updatePartitionCtxHost(args, resource, algoCtx);
    if (tsPtr) {
      tsPtr->startInterval("partition", algoCtx.partition);
    }
    CLOGF_TRACE(
        COLL,
        ALGO_CXT_LOG_FMT_HOST,
        ALGO_CXT_LOG_FIELDS(algoCtx, args.numBlocks));

    int totalSendTrans = algoCtx.opRounds[Op::kSendTrans].totalRounds;
    int totalRecvTrans = algoCtx.opRounds[Op::kRecvTrans].totalRounds;
    dataSResps.resize(totalSendTrans);
    // - Responses for sync control send to left neighbor. Need wait for a
    // previous response to finish when want to post the same recvBuf chunk.
    bufSyncSResps.resize(totalRecvTrans);
    // - Responses for sync control recv from right neighbor. Need wait for a
    // previous recvCtrl to finish when want to post irecvCtrl for the same
    // remote recvBuf chunk.
    bufSyncRResps.resize(totalSendTrans);
    // - Responses for local flush received data.
    flushResps.resize(totalRecvTrans);

    prePostRecvRemRecvBuf(args, resource, algoCtx, bufSyncRResps);

    // Ring main loop
    while (algoCtx.opRounds[Op::kSendTrans].done <
               algoCtx.opRounds[Op::kSendTrans].totalRounds ||
           algoCtx.opRounds[Op::kRecvRedCopy].done <
               algoCtx.opRounds[Op::kRecvRedCopy].totalRounds) {
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
      progressSend(args, resource, algoCtx, dataSResps, bufSyncRResps, tsPtr);
      HOST_ABORT();
      progressRecv(args, resource, algoCtx, bufSyncSResps, flushResps, tsPtr);
      HOST_ABORT();
    }

    // Release any remaining resps before moving to next partition
    int numDataSResps = waitAllResps(dataSResps, comm, "wait final dataSResps");
    int numSyncSResps =
        waitAllResps(bufSyncSResps, comm, "wait final bufSyncSResps");
    int numSyncRResps =
        waitAllResps(bufSyncRResps, comm, "wait final bufSyncRResps");

    CLOGF_TRACE(
        COLL,
        "Partition {} offset {} numel {} finished with {} syncSResps {} syncRResps {} dataSResps",
        algoCtx.partition,
        algoCtx.partitionOffset,
        algoCtx.partitionNumel,
        numSyncSResps,
        numSyncRResps,
        numDataSResps);

    if (tsPtr) {
      tsPtr->endInterval("partition", algoCtx.partition);
    }

    // Reset flags for next partition to reuse.
    // Kernel will wait for partitionSync update before checking the sync flags
    // for the next partition.
    resource.sendCopySync->resetStatus();
    resource.recvRedCopySync->resetStatus();

    // update local context
    updatePartitionDone(algoCtx);
    HOST_ABORT();
  } // end of partition loop

  // Reset flags for next allreduce to reuse
  resource.sendCopySync->reset();
  resource.recvRedCopySync->reset();
  resource.partitionSync->reset();

  if (tracer && ts) {
    ts->end();
    tracer->addRecord(std::move(ts));
  }

  return commSuccess;
}

commResult_t getNumBlocksAndThreads(
    int* numBlocks,
    int* numThreads,
    const void* func,
    size_t messageBytes,
    int nRanks) {
  FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
      numBlocks,
      numThreads,
      func,
      0 /* dynamicSMemSize */,
      0 /* blockSizeLimit */));

  if (NCCL_CTRAN_ALLREDUCE_RING_AUTO_TUNE_MAX_BDP > 0) {
    GpuArch arch = GpuArch::Default;
    int cudaDev = 0;
    FB_CUDACHECK(cudaGetDevice(&cudaDev));
    int smMajor = 0;
    FB_CUDACHECK(cudaDeviceGetAttribute(
        &smMajor, cudaDevAttrComputeCapabilityMajor, cudaDev));
    if (smMajor < 10) {
      arch = GpuArch::Hopper;
    }
    *numBlocks = getAutoTunedNumBlocks(messageBytes, nRanks, *numBlocks, arch);
    *numThreads =
        getAutoTunedThreadBlockSize(messageBytes, nRanks, *numThreads, arch);
  }

  // if (*numBlocks > NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS) {
  if (NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS > 0) {
    *numBlocks = NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS;
  }
  if (NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE > 0) {
    *numThreads = NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE;
  }

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
  const size_t messageBytes = count * typeSize;
  FB_COMMCHECK(
      ctran::allreduce::ring::getNumBlocksAndThreads(
          &numBlocks, &numThreads, func, messageBytes, nRanks));

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
  constexpr size_t kAllReduceRingNumSyncs = 3;
  FB_COMMCHECK(comm->ctran_->gpe->allocGpeKernelSyncs(
      kAllReduceRingNumSyncs, numBlocks, gpeKernelSyncs));
  FB_CHECKTHROW_EX(
      gpeKernelSyncs.size() == kAllReduceRingNumSyncs,
      comm->logMetaData_,
      "Failed to allocate GpeKernelSync");
  hostResource->sendCopySync = gpeKernelSyncs[0];
  hostResource->recvRedCopySync = gpeKernelSyncs[1];
  hostResource->partitionSync = gpeKernelSyncs[2];
  const int kAutoTuneMaxBDP = NCCL_CTRAN_ALLREDUCE_RING_AUTO_TUNE_MAX_BDP;
  if (kAutoTuneMaxBDP > 0) {
    auto params = ctran::allreduce::ring::getAutoTunedPipeline(
        messageBytes, kAutoTuneMaxBDP, nRanks);
    CLOGF(
        INFO,
        "AutoTune: pipline ({}, {}, {}) = ({}, {}) ",
        messageBytes,
        kAutoTuneMaxBDP,
        nRanks,
        params.chunkSize,
        params.numChunks);
    hostResource->chunkSize = params.chunkSize;
    hostResource->numChunks = params.numChunks;

    // One-time log of auto-tune decisions across message sizes
    static bool autoTuneLogged = false;
    if (!autoTuneLogged) {
      autoTuneLogged = true;

      // 32GB max
      constexpr int kPow2MaxExponent = 25;
      constexpr size_t KB = 1024ULL;
      for (int i = 0; i <= kPow2MaxExponent; i++) {
        const size_t sz = (1 << i) * KB;

        const auto p = ctran::allreduce::ring::getAutoTunedPipeline(
            sz, kAutoTuneMaxBDP, nRanks);
        const int blks = ctran::allreduce::ring::getAutoTunedNumBlocks(
            sz, nRanks, numBlocks);
        CLOGF(
            INFO,
            "AutoTune ranks {}, msg {}B: blocks {}, chunks {} x {}B",
            nRanks,
            sz,
            blks,
            p.numChunks,
            p.chunkSize);

        if (i != kPow2MaxExponent) {
          const size_t sz_next = (1 << (i + 1)) * KB;
          const size_t mid = (sz + sz_next) / 2;
          auto mp = ctran::allreduce::ring::getAutoTunedPipeline(
              mid, kAutoTuneMaxBDP, nRanks);
          int mblks = ctran::allreduce::ring::getAutoTunedNumBlocks(
              mid, nRanks, numBlocks);
          CLOGF(
              INFO,
              "AutoTune ranks {}, msg {}B: blocks {}, chunks {} x {}B",
              nRanks,
              sz_next,
              mblks,
              mp.numChunks,
              mp.chunkSize);
        }
      }
    }
  }

  if (NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS > 0) {
    hostResource->numChunks = NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS;
  }
  if (NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE > 0) {
    hostResource->chunkSize = NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE;
  }
  std::tie(hostResource->tmpSendBuf, hostResource->tmpSendBufHdl) =
      comm->ctran_->algo->getTmpBufInfo(
          CtranAlgo::TmpbufType::RING_TMP_SEND_BUF);
  std::tie(hostResource->tmpRecvBuf, hostResource->tmpRecvBufHdl) =
      comm->ctran_->algo->getTmpBufInfo(
          CtranAlgo::TmpbufType::RING_TMP_RECV_BUF);
  CLOGF(
      INFO,
      "AutoTune: {} blocks of {} threads, tmpbuf {} x {} chunks",
      numBlocks,
      numThreads,
      hostResource->numChunks,
      hostResource->chunkSize);

  auto* hostArgs = new ctran::allreduce::ring::HostArgs();
  op->allreduce.args = hostArgs;
  hostArgs->rank = rank;
  hostArgs->leftRank = (rank - 1 + nRanks) % nRanks;
  hostArgs->rightRank = (rank + 1) % nRanks;
  hostArgs->minShardSize = NCCL_CTRAN_ALLREDUCE_RING_MIN_SHARD_SIZE;
  hostArgs->numBlocks = numBlocks;
  hostArgs->numThreads = numThreads;
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
