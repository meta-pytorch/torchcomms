// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_fp16.h>
#include <deque>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"

CTRAN_DATATYPE_REDOP_TO_FUNC_MAPPER(
    reduceScatterKerns,
    ncclKernelReduceScatterRing);

static const auto myAlgo = NCCL_REDUCESCATTER_ALGO::ctring;

static inline std::tuple<size_t, int, int> getRingConfig(
    CtranComm* comm,
    struct OpElem* op) {
  // Compute stepCount based on tmpBuf size
  // We use half for receive and half for intermediate local reduce result.
  // (We may use recvbuf in out-of-place case, but to keep code simple we skip
  // this minor memory optimization).
  size_t stepCount = std::min(
      NCCL_CTRAN_INTERNODE_TMPBUF_SIZE /
          commTypeSize(op->reducescatter.datatype) / 2,
      op->reducescatter.recvcount);

  // Compute nSteps to finish
  // - We need run multiple rounds of the ring if size of tmpBuf < recvcount
  //   (e.g., nRings = 2 if tmpBuf = 16MB and recvcount = 30MB)
  int nRounds = op->reducescatter.recvcount / stepCount;
  if (op->reducescatter.recvcount % stepCount) {
    nRounds++;
  }
  // - Each round of the ring has nRanks - 1 steps
  int nSteps = comm->statex_->nRanks() - 1;
  return std::make_tuple(stepCount, nRounds, nSteps);
}

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t recvSize =
      op->reducescatter.recvcount * commTypeSize(op->reducescatter.datatype);
  CtranComm* comm = opGroup.front()->comm_;

  const int localRank = comm->statex_->localRank();
  const int nRanks = comm->statex_->nRanks();
  const int rank = comm->statex_->rank();

  void *sendHdl = nullptr, *recvHdl = nullptr;
  bool localRegSend = false, localRegRecv = false;
  void* remoteTmpBuf;
  struct CtranMapperRemoteAccessKey remoteTmpAccessKey;

  CtranAlgoLogger logger(reduceScatterAlgoName(myAlgo), op->opCount, comm);

  CtranMapper* mapper = comm->ctran_->mapper.get();

  CtranMapperContext context(
      reduceScatterAlgoName(myAlgo), recvSize * nRanks, recvSize);
  mapper->setContext(std::move(context));

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(reduceScatterAlgoName(myAlgo)));

  int leftNode = (rank + nRanks - 1) % nRanks;
  int leftRank = comm->statex_->localRankToRank(localRank, leftNode);
  int rightNode = (rank + 1) % nRanks;
  int rightRank = comm->statex_->localRankToRank(localRank, rightNode);
  auto [stepCount, nRounds, nSteps] = getRingConfig(comm, op);
  size_t stepSize = stepCount * commTypeSize(op->reducescatter.datatype);
  bool inplace = (reinterpret_cast<uintptr_t>(op->reducescatter.sendbuff) +
                  rank * recvSize) ==
      reinterpret_cast<uintptr_t>(op->reducescatter.recvbuff);

  CLOGF_TRACE(
      COLL,
      "myRank {} (rank {}) left={} (rank {}) right={} (rank {}), stepCount {} (size {}) nRounds {} nSteps {}, inplace {}",
      comm->statex_->rank(),
      rank,
      leftRank,
      leftNode,
      rightRank,
      leftNode,
      stepCount,
      stepSize,
      nRounds,
      nSteps,
      inplace);

  std::unique_ptr<CtranMapperRequest> irecvReq;
  std::vector<std::unique_ptr<CtranMapperRequest>> isendReqs;
  std::unique_ptr<CtranMapperNotify> notifyLeft;

  isendReqs.reserve(nRounds * nSteps);

  FB_COMMCHECK(mapper->searchRegHandle(
      op->reducescatter.sendbuff, recvSize * nRanks, &sendHdl, &localRegSend));
  FB_COMMCHECK(mapper->searchRegHandle(
      op->reducescatter.recvbuff, recvSize, &recvHdl, &localRegRecv));

  // No need register recvBuf since we always receive data in tmp buffer.
  auto [tmpBuf, tmpRegHdl] = comm->ctran_->algo->getTmpBufInfo(
      CtranAlgo::TmpbufType::INTERNODE_TMPBUF);

  CtranMapperRequest* req = nullptr;
  FB_COMMCHECK(
      mapper->irecvCtrl(&remoteTmpBuf, &remoteTmpAccessKey, rightRank, &req));
  irecvReq = std::unique_ptr<CtranMapperRequest>(req);

  FB_COMMCHECK(mapper->isendCtrl(tmpBuf, tmpRegHdl, leftRank, &req));
  isendReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));

  // Initialize notify flag to receive from left
  notifyLeft = std::make_unique<CtranMapperNotify>();
  FB_COMMCHECK(mapper->initNotify(leftRank, tmpRegHdl, notifyLeft.get()));

  // Main loop of Ring
  auto kElem = op->reducescatter.interReduce;
  size_t recvCount = op->reducescatter.recvcount;

  for (int r = 0; r < nRounds; r++) {
    // Each round reduces different portion of data in recvBuf
    // E.g., in a 2-round Ring for 31MB data, first round reduces 0-15MB for
    // each rank, second round reduces 16-31MB. At end of each round, each rank
    // should have the final reduced result of the finsihed portion.
    const char* sendBuf =
        reinterpret_cast<const char*>(op->reducescatter.sendbuff) +
        r * stepSize;
    // Tmp buffers for intermediate steps
    // - First half of tmpBuf is used to receive data from left
    void* recvBuf = tmpBuf;
    // - Second half of tmpBuf is used to store local reduce result
    void* redBuf =
        reinterpret_cast<char*>(tmpBuf) + NCCL_CTRAN_INTERNODE_TMPBUF_SIZE / 2;

    // Final step updates directly to user recvbuff
    char* finalRedBuf =
        reinterpret_cast<char*>(op->reducescatter.recvbuff) + r * stepSize;

    // Compute num of elements for this round
    size_t count = recvCount > stepCount ? stepCount : recvCount;

    // Within a Ring, each rank starts the partial reduce of chunk belonging to
    // the 2nd left neighbor and shift to right after each step. Thus, after
    // nRanks-1 step, each rank just handles the final partial reduce of the
    // chunk belonging to itself, avoiding additional data shifting after final
    // reduce. E.g., nRanks = 4 requires 3 steps per round, the partial reduce
    // of each step is below:
    //   - rank of rank 0: chunk_2(0+3), chunk_1(3+2+0), chunk_0(1+2+3+0)
    //   - rank of rank 1: chunk_3(1+0), chunk_2(0+3+1), chunk_1(3+2+0+1)
    //   - rank of rank 2: chunk_0(1+2), chunk_3(1+0+2), chunk_2(0+3+1+2)
    //   - rank of rank 3: chunk_1(3+2), chunk_0(1+2+3), chunk_3(1+0+2+3)
    int redPos = (nRanks + rank - 2) % nRanks;
    // Each rank puts the chunk to be reduced by the right neighbor
    int putPos = (nRanks + rank - 1) % nRanks;
    for (int i = 0; i < nRanks - 1; i++) {
      CtranMapperRequest* req = nullptr;
      bool lastStep = i == nRanks - 2;
      bool lastRound = r == nRounds - 1;

      // Wait RTS from right (e.g., previous reduce is done, thus recvBuf is
      // ready to receive new data)
      FB_COMMCHECK(mapper->waitRequest(irecvReq.get()));
      timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(rightRank));

      // Post iRecv for next RTS from right
      if (!lastStep || !lastRound) {
        FB_COMMCHECK(mapper->irecvCtrl(rightRank, &req));
        irecvReq = std::unique_ptr<CtranMapperRequest>(req);
      }

      // Put to right
      // Only first step needs to put from sendBuf, remaining steps just forward
      // local reduce result in tmpBuf
      const void* putSrc = (i == 0) ? sendBuf + putPos * recvSize : redBuf;
      void* putHdl = (i == 0) ? sendHdl : tmpRegHdl;

      CtranMapperRequest* iputReq = nullptr;
      FB_COMMCHECK(mapper->iput(
          putSrc,
          remoteTmpBuf, // use first half of tmpbuf to receive data
          count * commTypeSize(op->reducescatter.datatype),
          rightRank,
          CtranMapperConfig{
              .memHdl_ = putHdl,
              .remoteAccessKey_ = remoteTmpAccessKey,
              .notify_ = true},
          &iputReq));

      FB_COMMCHECK(mapper->waitNotify(notifyLeft.get()));

      // Local reduce will update redBuf, let's ensure the previous put has
      // finished so redBuf can be updated
      FB_COMMCHECK(mapper->waitRequest(iputReq));
      delete iputReq;

      kElem->reduce.ndsts = 1;
      kElem->reduce.nsrcs = 2; // Always reduce from 2 srcs (tmpBuf, srcbuf)
      kElem->reduce.count = count;
      kElem->reduce.isFinal = lastStep;
      kElem->reduce.dsts[0] = lastStep ? finalRedBuf : redBuf;
      kElem->reduce.srcs[0] = recvBuf;
      kElem->reduce.srcs[1] = sendBuf + redPos * recvSize;
      // Flush memory so that we can forward to the next rank
      kElem->reduce.flushMem = true;

      // Post to kernel to reduce
      kElem->post();

      // Wait local reduce to finish
      // TODO: pipeline local reduce and data transfer
      kElem->wait();

      if (!lastStep || !lastRound) {
        // Notify left to send the next chunk to my tmpBuf
        FB_COMMCHECK(mapper->isendCtrl(leftRank, &req));
        isendReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
      }

      redPos = (redPos + nRanks - 1) % nRanks;
    }
    recvCount -= count;
  }

  while (!isendReqs.empty()) {
    FB_COMMCHECK(mapper->testSomeRequests(isendReqs));
  }

  if (localRegSend == true) {
    FB_COMMCHECK(mapper->deregDynamic(sendHdl));
  }
  if (localRegRecv == true) {
    FB_COMMCHECK(mapper->deregDynamic(recvHdl));
  }

  mapper->timestamps.push_back(std::move(timestamp));
  mapper->reportProfiling();

  return commSuccess;
}

static inline int getNumGroups(size_t count, int nvectors) {
  // compute needed thread blocks for given bytes
  int nGroups = (count * nvectors) /
      NCCL_CTRAN_REDUCESCATTER_REDUCE_NELEM_PER_THREAD_BLOCK;
  return std::min(
      std::max(1, nGroups), // at least 1 thread block
      // not exceed max theshold
      NCCL_CTRAN_REDUCESCATTER_RING_MAX_NUM_THREAD_BLOCKS);
}

static unsigned int bestThreadBlockSize = 0;

static inline unsigned int getThreadBlockSize() {
  // If first time call, query cuda recommended blockSize
  if (bestThreadBlockSize == 0) {
    int minGridSize = 0;
    FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        (int*)&bestThreadBlockSize,
        reinterpret_cast<const void*>(
            ncclKernelReduceScatterRing<int, commSum>),
        0 /* dynamicSMemSize */,
        0 /* blockSizeLimit */));
  }

  return NCCL_CTRAN_REDUCESCATTER_THREAD_BLOCK_SIZE == -1
      ? bestThreadBlockSize
      : NCCL_CTRAN_REDUCESCATTER_THREAD_BLOCK_SIZE;
}

static inline commResult_t setupPlan(
    CtranComm* comm,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    KernelConfig& config) {
  auto op = opGroup.front().get();
  config.args.devState_d = comm->ctran_->algo->getDevState();

  auto [stepCount, nRounds, nSteps] = getRingConfig(comm, op);

  // Local reduce for inter-rank potion handles only at most 2 srcs (i.e., local
  // and received data in tmpbuf) at a time
  int nGroups = getNumGroups(stepCount, 2);

  KernelElem* elem = nullptr;
  FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, nGroups, &elem));

  elem->reduce.ndsts = 1;
  elem->reduce.nsrcs = 2; // Always reduce from 2 srcs (tmpBuf, srcbuf)
  op->reducescatter.interReduce = elem;

  // Kernel doesn't use them, but for colltrace to record only
  config.args.collective.reducescatter.sendbuff = op->reducescatter.sendbuff;
  config.args.collective.reducescatter.recvbuff = op->reducescatter.recvbuff;
  config.args.collective.reducescatter.datatype = op->reducescatter.datatype;
  config.args.collective.reducescatter.recvcount = op->reducescatter.recvcount;
  config.args.collective.reducescatter.interReduce = elem;
  config.args.collective.reducescatter.nStepsInterReduce = nRounds * nSteps;

  config.numBlocks = nGroups;
  config.numThreads = getThreadBlockSize();
  return commSuccess;
}

commResult_t ctranReduceScatterRing(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream) {
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_REDCOLL_INFO(
      reduceScatterAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      recvcount,
      datatype,
      redOp,
      -1,
      comm,
      stream);
  // make sure tmpbuf is allocated and registered on main thread
  FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());

  op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::REDUCESCATTER, comm, opCount));
  op->reducescatter.sendbuff = sendbuff;
  op->reducescatter.recvbuff = recvbuff;
  op->reducescatter.recvcount = recvcount;
  op->reducescatter.datatype = datatype;
  op->reducescatter.redOp = redOp;
  opGroup.push_back(std::move(op));

  auto config = KernelConfig(
      KernelConfig::KernelType::REDUCESCATTER,
      stream,
      reduceScatterAlgoName(myAlgo),
      opCount);
  FB_COMMCHECK(setupPlan(comm, opGroup, config));

  const void* func = reduceScatterKerns.at(std::make_pair(datatype, redOp));
  FB_COMMCHECK(
      comm->ctran_->gpe->submit(std::move(opGroup), impl, config, func));

  return commSuccess;
}
