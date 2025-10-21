// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cuda_fp16.h>
#include <deque>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/DevUtils.cuh"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

CTRAN_DATATYPE_REDOP_TO_FUNC_MAPPER(
    reduceScatterKerns,
    ncclKernelReduceScatterRHD);

static const auto myAlgo = NCCL_REDUCESCATTER_ALGO::ctrhd;

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t recvCount = op->reducescatter.recvcount;
  size_t recvSize = recvCount * commTypeSize(op->reducescatter.datatype);
  CtranComm* comm = opGroup.front()->comm_;
  const int rank = comm->statex_->rank();
  const int nRanks = comm->statex_->nRanks();
  int nSteps = ctran::utils::log2i(nRanks);
  void* sendbuff = (void*)op->reducescatter.sendbuff;
  void *tmpBuf, *tmpRecvBuf, *tmpRedBuf, *finalRedBuf;

  void *sendHdl{nullptr}, *recvHdl{nullptr}, *tmpHdl{nullptr};
  bool localRegSend{false}, localRegRecv{false};
  std::vector<size_t> peers(nSteps);
  std::vector<void*> remoteTmpBuffs(nSteps);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nSteps);
  std::vector<std::unique_ptr<CtranMapperRequest>> irecvReq(nSteps);
  std::vector<std::unique_ptr<CtranMapperRequest>> isendReq(nSteps);
  // these are the final iput requests per step for which we will need to wait
  std::vector<std::unique_ptr<CtranMapperRequest>> iputReq(nSteps);
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec(nSteps);
  CtranAlgoLogger logger(reduceScatterAlgoName(myAlgo), op->opCount, comm);
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("ctranReduceScatterRHD"));
  for (int i = 0; i < nSteps; i++) {
    peers[i] = rank ^ (1 << i); // flip the i-th bit from the end
  }

  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      sendbuff, nRanks * recvSize, &sendHdl, &localRegSend));
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      op->reducescatter.recvbuff, recvSize, &recvHdl, &localRegRecv));

  CtranMapperContext context(
      reduceScatterAlgoName(myAlgo), recvSize * nRanks, recvSize);
  comm->ctran_->mapper->setContext(std::move(context));

  // No need register recvBuf since we always receive data in tmp buffer.
  std::tie(tmpBuf, tmpHdl) = comm->ctran_->algo->getTmpBufInfo(
      CtranAlgo::TmpbufType::INTERNODE_TMPBUF);

  // Tmp buffers for intermediate steps:
  // - First half of tmpBuf is used to receive data from peer
  tmpRecvBuf = tmpBuf;
  // - Second half of tmpBuf is used to store local reduce result and
  //   to send to peer
  tmpRedBuf =
      reinterpret_cast<char*>(tmpBuf) + NCCL_CTRAN_INTERNODE_TMPBUF_SIZE / 2;

  // Final step updates directly to user recvbuff
  finalRedBuf = reinterpret_cast<char*>(op->reducescatter.recvbuff);

  auto kElem = op->reducescatter.interReduce;

  // Exchange memory handles with relevant peers
  for (size_t i = 0; i < nSteps; i++) {
    auto peer = peers[i];
    CtranMapperRequest* recvReq = nullptr;
    FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
        &remoteTmpBuffs[i], &remoteAccessKeys[i], peer, &recvReq));
    irecvReq[i] = std::unique_ptr<CtranMapperRequest>(recvReq);
  }

  for (size_t i = 0; i < nSteps; i++) {
    auto peer = peers[i];
    bool lastStep = i == nSteps - 1;

    CLOGF_TRACE(COLL, "rank {} peer {} step {}", rank, peer, i);

    // Send buffer information to peer, which also signals that this rank
    // is ready to receive
    CtranMapperRequest* sendReq = nullptr;
    FB_COMMCHECK(
        comm->ctran_->mapper->isendCtrl(tmpBuf, tmpHdl, peer, &sendReq));
    isendReq[i] = std::unique_ptr<CtranMapperRequest>(sendReq);

    // Initialize notify to receive notification from peer
    notifyVec[i] = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(
        comm->ctran_->mapper->initNotify(peer, tmpHdl, notifyVec[i].get()));

    // Block until we have handle for this peer
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(irecvReq[i].get()));
    timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));
    for (size_t j = 0; j < (nRanks >> (i + 1)); j++) {
      bool lastChunkPerStep = (j == (nRanks >> (i + 1)) - 1) ? 1 : 0;
      size_t sendBufOffset = j * (2 << i) + peer % (2 << i);
      size_t tmpBufOffset = j * 2 + (peer >> i) % 2;
      CtranMapperRequest* putReqPtr = nullptr;

      // Put to peer
      // Only first step needs to put from sendbuff, remaining steps just
      // forward local reduce result in tmpBuf
      const void* putSrc = (i == 0)
          ? (char*)sendbuff + sendBufOffset * recvSize
          : (char*)tmpRedBuf + tmpBufOffset * recvSize;
      void* putHdl = (i == 0) ? sendHdl : tmpHdl;

      CLOGF_TRACE(
          COLL,
          "iput rank {} peer {} sendBufOffset: {} tmpBufOffset: {} lastChunkPerStep: {}",
          rank,
          peer,
          sendBufOffset,
          tmpBufOffset,
          lastChunkPerStep);
      if (lastChunkPerStep) {
        FB_COMMCHECK(comm->ctran_->mapper->iput(
            putSrc,
            (char*)remoteTmpBuffs[i] + j * recvSize,
            recvSize,
            peer,
            CtranMapperConfig{
                .memHdl_ = putHdl,
                .remoteAccessKey_ = remoteAccessKeys[i],
                .notify_ = lastChunkPerStep ? true : false},
            &putReqPtr));
      } else {
        FB_COMMCHECK(comm->ctran_->mapper->iput(
            putSrc,
            (char*)remoteTmpBuffs[i] + j * recvSize,
            recvSize,
            peer,
            CtranMapperConfig{
                .memHdl_ = putHdl,
                .remoteAccessKey_ = remoteAccessKeys[i],
                .notify_ = lastChunkPerStep ? true : false},
            static_cast<CtranMapperRequest*>(nullptr)));
      }
      // Capture duration started from first put
      if (j == 0) {
        timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
      }
      if (lastChunkPerStep) {
        iputReq[i] = std::unique_ptr<CtranMapperRequest>(putReqPtr);
        // Local reduce will update tmpRedBuf, let's ensure the previous put has
        // finished so tmpRedBuf can be updated
        CLOGF_TRACE(
            COLL, "rank {} peer {} iputReq: {}", rank, peer, (void*)putReqPtr);
        FB_COMMCHECK(comm->ctran_->mapper->waitRequest(iputReq[i].get()));
        timestamp->putComplete.push_back(CtranMapperTimestampPoint(peer));
      }
    }
    // make sure send from peer is completed
    FB_COMMCHECK(comm->ctran_->mapper->waitNotify(notifyVec[i].get()));

    // Local reduce
    for (size_t j = 0; j < (nRanks >> (i + 1)); j++) {
      bool lastChunkPerStep = (j == (nRanks >> (i + 1)) - 1) ? 1 : 0;
      // offsets for local reduce are analogous to the offsets in the
      // send phase above, except using this rank's id rather than the peer's id
      size_t tmpRedBufOffset = j * 2 + (rank >> i) % 2;
      size_t sendBufRedOffset = j * (2 << i) + rank % (2 << i);

      kElem->reduce.ndsts = 1;
      kElem->reduce.nsrcs = 2; // Always reduce from 2 srcs (tmpBuf, srcbuf)
      kElem->reduce.count = recvCount;
      kElem->reduce.isFinal = lastStep;
      kElem->reduce.dsts[0] = lastStep ? (char*)finalRedBuf + j * recvSize
                                       : (char*)tmpRedBuf + j * recvSize;
      kElem->reduce.srcs[0] = (char*)tmpRecvBuf + j * recvSize;
      kElem->reduce.srcs[1] = (i == 0)
          ? (char*)sendbuff + sendBufRedOffset * recvSize
          : (char*)tmpRedBuf + tmpRedBufOffset * recvSize;

      CLOGF_TRACE(
          COLL,
          "Elem post rank {} peer {} tmpRedBufOffset: {} sendBufRedOffset: {} lastChunkPerStep: {}",
          rank,
          peer,
          tmpRedBufOffset,
          sendBufRedOffset,
          lastChunkPerStep);
      // Post to kernel to reduce
      timestamp->kernelPost.push_back(CtranMapperTimestampPoint(peer));
      kElem->post();

      // Wait for local reduce to finish
      // TODO: pipeline local reduce and data transfer
      timestamp->kernelWait.push_back(CtranMapperTimestampPoint(peer));
      kElem->wait();
      timestamp->kernelWaitComplete.push_back(CtranMapperTimestampPoint(peer));
    }
  }

  for (int i = 0; i < nSteps; i++) {
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(isendReq[i].get()));
  }

  if (localRegSend == true) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(sendHdl));
  }
  if (localRegRecv == true) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(recvHdl));
  }
  comm->ctran_->mapper->timestamps.push_back(std::move(timestamp));
  comm->ctran_->mapper->reportProfiling();

  CLOGF_TRACE(COLL, "rank {} reached end of ReduceScatter", rank);
  return commSuccess;
}

static inline int getNumGroups(size_t count, int nvectors) {
  // compute needed thread blocks for given bytes
  int nGroups = (count * nvectors) /
      NCCL_CTRAN_REDUCESCATTER_REDUCE_NELEM_PER_THREAD_BLOCK;
  return std::min(
      std::max(1, nGroups), // at least 1 thread block
      // not exceed max theshold
      NCCL_CTRAN_REDUCESCATTER_RHD_MAX_NUM_THREAD_BLOCKS);
}

static unsigned int bestThreadBlockSize = 0;

static inline unsigned int getThreadBlockSize() {
  // If first time call, query cuda recommended blockSize
  if (bestThreadBlockSize == 0) {
    int minGridSize = 0;
    FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        (int*)&bestThreadBlockSize,
        reinterpret_cast<const void*>(ncclKernelReduceScatterRHD<int, commSum>),
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
  size_t nRanks = comm->statex_->nRanks();
  size_t recvcount = op->reducescatter.recvcount;
  // Local reduce for inter-rank portion handles only at most 2 srcs (i.e.,
  // local and received data in tmpbuf) at a time
  int nGroups = getNumGroups(recvcount, 2);

  KernelElem* elem = nullptr;
  FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, nGroups, &elem));
  elem->reduce.ndsts = 1;
  elem->reduce.nsrcs = 2;
  elem->reduce.count = recvcount;
  elem->reduce.flushMem = true;
  elem->reduce.barrier = false;
  op->reducescatter.interReduce = elem;
  config.args.collective.reducescatter.sendbuff = op->reducescatter.sendbuff;
  config.args.collective.reducescatter.recvbuff = op->reducescatter.recvbuff;
  config.args.collective.reducescatter.datatype = op->reducescatter.datatype;
  config.args.collective.reducescatter.recvcount = op->reducescatter.recvcount;
  config.args.collective.reducescatter.interReduce = elem;
  config.args.collective.reducescatter.intraReduce = nullptr;
  // number of blocks == nRanks/2 + nRanks/4 + ... + 2 + 1 = nRanks -1
  config.args.collective.reducescatter.nStepsInterReduce = nRanks - 1;

  config.numBlocks = nGroups;
  config.numThreads = getThreadBlockSize();
  return commSuccess;
}

// Recursive vector-halving distance-doubling ReduceScatter
//
// The algorithm consists of log P steps (when P is a power of 2). In each
// step the ranks are partitioned into pairs, with each member of the pair
// both sending and receiving. In step i, each rank communicates with the
// rank obtained by flipping the i-th bit from the end in its binary
// representation. This results in neighbor ranks communicating in iteration
// 0 and the distance between communicating ranks doubling in each
// subsequent iteration (hence distance-doubling).
// Given a send buffer of size M, each rank sends/receives M/2 bytes in
// iteration 0. The amount of data sent/received per step is then halved in
// each subsequent iteration (hence vector-halving).
// While the algorithm can also be done in distance-halving fashion, the
// distance-doubling variant has the advantage that the largest messages
// are sent in the initial steps between nearby ranks, which can
// be advantegeous when nearby ranks are intranode.
// In the classic version of the algorithm, in step 0
// odd ranks send the first half of the data array to their
// left neighbor while even ranks send the second half of the data
// array to their right neighbor. The received data is then reduced
// with the corresponding part of the array at the destination. In
// subsequent iterations, each rank sends/receives halves of the portion
// of the array received/reduced in the prior step. This pattern has the
// effect that in the final step each rank will have a block of the array
// reduced over all ranks. However, the requirement that for all ranks,
// the i-th rank will have the i-th block of the array is not
// satisfied.
// In order to satisfy the above requirement while preserving the
// locality advantages of distance-doubling, we modify
// which data are sent/received in each step, while keeping
// the amount of data sent/received in each step same as in the
// classic algorithm. The array is partitioned into P blocks and in
// iteration 0, even ranks will send all the odd blocks in their array
// to their right neighbors, while odd ranks will send all their even
// blocks to their left neighbors. As in the classic algorithm, in
// each subsequent step, ranks send/receive halves of the portion
// of the array reduced in the prior step, using the same strided
// pattern as in iteration 0.
// This initial implementation assumes tmpBuf is at least as large
// as the send buffer (M) and that P is a power of 2.
// These constraints will be removed in follow up diffs.
commResult_t ctranReduceScatterRHD(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream) {
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;
  const auto& statex = comm->statex_.get();
  auto opCount = comm->ctran_->getOpCount();
  size_t typeSize = commTypeSize(datatype);
  void* sbuf = const_cast<void*>(sendbuff);
  void* dbuf = recvbuff;

  CTRAN_COLL_INFO(
      reduceScatterAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      recvcount,
      datatype,
      -1,
      comm,
      stream);
  const size_t totalBufSize = recvcount * typeSize * statex->nRanks();
  if (NCCL_CTRAN_INTERNODE_TMPBUF_SIZE < totalBufSize) {
    CLOGF(
        WARN,
        "ctranReduceScatterRHD: data buffer of size {} bytes "
        "is too large to fit in tmpBuf",
        totalBufSize);
    return commInternalError;
  } else if ((statex->nRanks() & (statex->nRanks() - 1)) != 0) {
    CLOGF(
        WARN,
        "ctranReduceScatterRHD: current implementation requires"
        "number of ranks to be a power of 2, nRanks: {}",
        statex->nRanks());
    return commInternalError;
  }

  // make sure tmpbuf is allocated and registered
  FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());

  if (totalBufSize < CTRAN_MIN_REGISTRATION_SIZE) {
    sbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_SRC_TMPBUF);
    FB_CUDACHECK(cudaMemcpyAsync(
        sbuf, sendbuff, totalBufSize, cudaMemcpyDefault, stream));
  }

  if (recvcount * typeSize < CTRAN_MIN_REGISTRATION_SIZE) {
    dbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_DST_TMPBUF);
  }

  op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::REDUCESCATTER, comm, opCount));
  op->reducescatter.sendbuff = reinterpret_cast<const void*>(sbuf);
  op->reducescatter.recvbuff = dbuf;
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

  // copy result out of tmpbufSegments[MIN_REG_DST_TMPBUF] if recvbuff was
  // switched due to the original not satisfying the min registration size
  // requirement
  if (dbuf ==
      comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::MIN_REG_DST_TMPBUF)) {
    FB_CUDACHECK(cudaMemcpyAsync(
        recvbuff, dbuf, recvcount * typeSize, cudaMemcpyDefault, stream));
  }

  return commSuccess;
}
