// Copyright (c) Meta Platforms, Inc. and affiliates.

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
#include "comms/ctran/algos/AllReduce/AllReduceRingCommon.cuh"
#include "comms/ctran/algos/CtranAlgo.h"
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
    FB_COMMCHECKTHROW(
        resource.comm->ctran_->mapper->irecvCtrl(args.rightRank, &req));
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
  FB_CHECKTHROW(
      resp != nullptr,
      "bufSyncRResps is not initialized at round {}",
      prevRound);

  if (resp) {
    bool isComplete = false;
    FB_COMMCHECKTHROW(
        resource.comm->ctran_->mapper->testRequest(resp.get(), &isComplete));
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
      FB_COMMCHECKTHROW(
          resource.comm->ctran_->mapper->testRequest(resp.get(), &isComplete));
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
  FB_CHECKTHROW(chunkArg.numel > 0, "Unexpected empty chunk");

  char* tmpRemoteRecvBuf = reinterpret_cast<char*>(args.rightRemBuf) +
      tmpChunkId * algoCtx.chunkSize;
  char* tmpSendBuf = reinterpret_cast<char*>(resource.tmpSendBuf) +
      tmpChunkId * algoCtx.chunkSize;

  CtranMapperRequest* req;
  FB_COMMCHECKTHROW(resource.comm->ctran_->mapper->iput(
      tmpSendBuf,
      tmpRemoteRecvBuf,
      chunkArg.numel * algoCtx.typeSize,
      args.rightRank,
      CtranMapperConfig{
          .memHdl_ = resource.tmpSendBufHdl,
          .remoteAccessKey_ = args.rightRemKey,
          .notify_ = true,
      },
      &req));
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
  FB_COMMCHECKTHROW(
      resource.comm->ctran_->mapper->checkNotify(args.leftNotify.get(), &done));
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
  FB_COMMCHECKTHROW(resource.comm->ctran_->mapper->iflush(
      tmpRecvBuf, resource.tmpRecvBufHdl, &req));
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

  FB_CHECKTHROW(
      flushResps.at(round) != nullptr,
      "Flush resp is not initialized at round {} step {} chunkId {}",
      round,
      step,
      chunkId);
  auto& resp = flushResps.at(round);

  bool isComplete = false;
  FB_COMMCHECKTHROW(
      resource.comm->ctran_->mapper->testRequest(resp.get(), &isComplete));
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
  FB_COMMCHECKTHROW(
      resource.comm->ctran_->mapper->isendCtrl(args.leftRank, &req));
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

inline int waitAllResps(
    std::vector<std::unique_ptr<CtranMapperRequest>>& reqs,
    CtranComm* comm,
    const std::string& ctx) {
  int numComplete = 0;
  for (auto& req : reqs) {
    if (req) {
      numComplete++;
      FB_COMMCHECKTHROW(comm->ctran_->mapper->waitRequest(req.get()));
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
  FB_CHECKTHROW(opGroup.size() == 1, "ctring opGroup expected exactly one op");
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
      // TODO: enable other data types
      switch (op->allreduce.datatype) {
        case commFloat32:
        case commUint64:
        case commInt32:
        case commInt8:
          break;
        default:
          throw ctran::utils::Exception(
              fmt::format("Unsupported data type {}", op->allreduce.datatype),
              commInvalidArgument);
      }
      progressSend(args, resource, algoCtx, dataSResps, bufSyncRResps);
      HOST_ABORT();
      progressRecv(args, resource, algoCtx, bufSyncSResps, flushResps);
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

  return commSuccess;
}

commResult_t
getNumBlocksAndThreads(int* numBlocks, int* numThreads, const void* func) {
  // Allow user to customize thread block size if specified
  FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
      numBlocks,
      numThreads,
      func,
      0 /* dynamicSMemSize */,
      0 /* blockSizeLimit */));
  if (*numBlocks > NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS) {
    *numBlocks = NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS;
  }
  if (*numThreads > NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE) {
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

  const auto& statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();
  const auto rank = statex->rank();

  FB_CHECKTHROW(
      typeToFunc.contains(std::make_pair(datatype, redOp)),
      "typeToFunc does not contain datatype {} with op {}",
      datatype,
      redOp);
  const void* func = typeToFunc.at(std::make_pair(datatype, redOp));

  int numBlocks = 0;
  int numThreads = 0;
  FB_COMMCHECK(ctran::allreduce::ring::getNumBlocksAndThreads(
      &numBlocks, &numThreads, func));

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
  FB_CHECKTHROW(
      gpeKernelSyncs.size() == kAllReduceRingNumSyncs,
      "Failed to allocate GpeKernelSync");
  hostResource->sendCopySync = gpeKernelSyncs[0];
  hostResource->recvRedCopySync = gpeKernelSyncs[1];
  hostResource->partitionSync = gpeKernelSyncs[2];
  hostResource->chunkSize = NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE;
  hostResource->numChunks = NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS;
  std::tie(hostResource->tmpSendBuf, hostResource->tmpSendBufHdl) =
      comm->ctran_->algo->getTmpBufInfo(
          CtranAlgo::TmpbufType::RING_TMP_SEND_BUF);
  std::tie(hostResource->tmpRecvBuf, hostResource->tmpRecvBufHdl) =
      comm->ctran_->algo->getTmpBufInfo(
          CtranAlgo::TmpbufType::RING_TMP_RECV_BUF);

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
