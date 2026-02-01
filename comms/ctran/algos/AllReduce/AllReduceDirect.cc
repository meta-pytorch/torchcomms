// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_fp16.h>
#include <iostream>

#include "comms/ctran/CtranComm.h"

#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

CTRAN_DATATYPE_REDOP_TO_FUNC_MAPPER(typeToFunc, ncclKernelAllReduceCtranDirect);

static const auto myAlgo = NCCL_ALLREDUCE_ALGO::ctdirect;

/*
 * Algorithm:
 *   - We split the data to be allreduced into two segments.  The
 *     first segment count is a multiple of ranks, so all processes
 *     are involved and it can use a symmetric algorithm between all
 *     the processes.  The second segment is the remaining count that
 *     is not divisible by the number of ranks, and is handled as a
 *     special case.
 *   - For the first segment, we perform four steps:
 *     - Intra-node reduce-scatter
 *       - Directly read data over NVLink from other GPUs into
 *         registers and reduce into the receive buffer.
 *     - Inter-node reduce-scatter (rail-based)
 *       - Each process scatters its reduced data into the other
 *         processes temporary buffer in its rail.
 *       - Each process reads its chunk of the already reduced data
 *         and the data it received from other processes into
 *         registers and reduces them into the receive buffer.
 *     - Inter-node allgather (rail-based)
 *       - Each process broadcasts its chunk of reduced data into the
 *         receive buffers of the other processes on its rail.
 *     - Intra-node allgather
 *       - Each process broadcasts the data from its rail into the
 *         receive buffers of the other processes within the node.
 *   - For the second segment, we perform two steps:
 *     - Intra-node flat allreduce
 *       - Each process reads data from the other processes into
 *         registers, and reduces it into the receive buffer.
 *     - Inter-node flat allreduce
 *       - Each process broadcasts its reduced data into the other
 *         processes temporary buffer in its rail.
 *       - Each process reads its chunk of the already reduced data
 *         and the data it received from other processes into
 *         registers and reduces them into the receive buffer.
 */

#define THROW_IF_ABORTED(code)                                        \
  do {                                                                \
    code;                                                             \
    if (comm->testAbort()) {                                          \
      throw ctran::utils::Exception("comm aborted", commRemoteError); \
    }                                                                 \
  } while (0)

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t typeSize = commTypeSize(op->allreduce.datatype);
  size_t size = op->allreduce.count * typeSize;
  CtranComm* comm = opGroup.front()->comm_;
  const auto& statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  const int node = statex->node();
  const int nNodes = statex->nNodes();
  const int nLocalRanks = statex->nLocalRanks();
  const int localRank = statex->localRank();
  void *sendHdl, *recvHdl;
  KernelElem* elem = nullptr;

  CtranAlgoLogger logger(allReduceAlgoName(myAlgo), op->opCount, comm);

  /* intra-node */
  std::vector<void*> intraNodeRemoteSendBuffs(nLocalRanks);
  std::vector<void*> intraNodeRemoteRecvBuffs(nLocalRanks);
  std::vector<struct CtranMapperRemoteAccessKey> intraNodeRemoteSendAccessKeys(
      nLocalRanks);
  std::vector<struct CtranMapperRemoteAccessKey> intraNodeRemoteRecvAccessKeys(
      nLocalRanks);
  std::vector<std::unique_ptr<CtranMapperRequest>> intraNodeRemoteSendbuffReq(
      nLocalRanks);
  std::vector<std::unique_ptr<CtranMapperRequest>> intraNodeRemoteRecvbuffReq(
      nLocalRanks);
  std::vector<std::unique_ptr<CtranMapperRequest>> intraNodeLocalSendbuffReq(
      nLocalRanks);
  std::vector<std::unique_ptr<CtranMapperRequest>> intraNodeLocalRecvbuffReq(
      nLocalRanks);

  /* inter-node */
  std::vector<void*> interNodeRemoteRecvBuffs(nNodes);
  std::vector<struct CtranMapperRemoteAccessKey> interNodeRemoteRecvAccessKeys(
      nNodes);
  std::vector<std::unique_ptr<CtranMapperRequest>> interNodeRemoteRecvbuffReq(
      nNodes);
  std::vector<std::unique_ptr<CtranMapperRequest>> interNodeLocalRecvbuffReq(
      nNodes);
  std::vector<std::unique_ptr<CtranMapperRequest>> interNodePutReq(nNodes);

  bool localRegSend, localRegRecv;
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allReduceAlgoName(myAlgo)));

  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      op->allreduce.sendbuff, size, &sendHdl, &localRegSend));
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      op->allreduce.recvbuff, size, &recvHdl, &localRegRecv));

  CtranMapperContext context(allReduceAlgoName(myAlgo), size, size);
  comm->ctran_->mapper->setContext(std::move(context));

  // FIXME: There seems to be a bug in our control message exchange,
  // so we first exchange control messages for the sendbuff, wait for
  // it to complete, and then exchange control messages for the
  // recvbuff.

  // Issue sendbuff control messages within the node
  for (int lr = 0; lr < nLocalRanks; lr++) {
    if (lr != localRank) {
      int p = statex->localRankToRank(lr);

      CtranMapperRequest* req = nullptr;
      FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
          &intraNodeRemoteSendBuffs[lr],
          &intraNodeRemoteSendAccessKeys[lr],
          p,
          &req));
      intraNodeRemoteSendbuffReq[lr] = std::unique_ptr<CtranMapperRequest>(req);

      FB_COMMCHECK(comm->ctran_->mapper->isendCtrl(
          op->allreduce.sendbuff, sendHdl, p, &req));
      intraNodeLocalSendbuffReq[lr] = std::unique_ptr<CtranMapperRequest>(req);
    }
  }

  // Wait for all sendbuff control messages to complete
  for (int lr = 0; lr < nLocalRanks; lr++) {
    if (lr != localRank) {
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest(
          intraNodeRemoteSendbuffReq[lr].get()));
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest(
          intraNodeLocalSendbuffReq[lr].get()));
      if (intraNodeRemoteSendAccessKeys[lr].backend !=
          CtranMapperBackend::NVL) {
        CLOGF(
            WARN,
            "NVLink backend not available between rank {} and {}",
            rank,
            statex->localRankToRank(lr));
      }
    }
  }

  // Issue recvbuff control messages within the node
  for (int lr = 0; lr < nLocalRanks; lr++) {
    if (lr != localRank) {
      int p = statex->localRankToRank(lr);

      CtranMapperRequest* req = nullptr;
      FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
          &intraNodeRemoteRecvBuffs[lr],
          &intraNodeRemoteRecvAccessKeys[lr],
          p,
          &req));
      intraNodeRemoteRecvbuffReq[lr] = std::unique_ptr<CtranMapperRequest>(req);

      FB_COMMCHECK(comm->ctran_->mapper->isendCtrl(
          op->allreduce.recvbuff, recvHdl, p, &req));
      intraNodeLocalRecvbuffReq[lr] = std::unique_ptr<CtranMapperRequest>(req);
    }
  }

  // Wait for all recvbuff control messages to complete
  for (int lr = 0; lr < nLocalRanks; lr++) {
    if (lr != localRank) {
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest(
          intraNodeRemoteRecvbuffReq[lr].get()));
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest(
          intraNodeLocalRecvbuffReq[lr].get()));
      if (intraNodeRemoteRecvAccessKeys[lr].backend !=
          CtranMapperBackend::NVL) {
        CLOGF(
            ERR,
            "NVLink backend not available between rank {} and {}",
            rank,
            statex->localRankToRank(lr));
        return commInternalError;
      }
    }
  }

  // Issue recvbuff control messages for rail-based inter-node
  // communication
  for (int n = 0; n < nNodes; n++) {
    if (n != node) {
      int p = statex->localRankToRank(localRank, n);

      CtranMapperRequest* req = nullptr;
      FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
          &interNodeRemoteRecvBuffs[n],
          &interNodeRemoteRecvAccessKeys[n],
          p,
          &req));
      interNodeRemoteRecvbuffReq[n] = std::unique_ptr<CtranMapperRequest>(req);

      FB_COMMCHECK(comm->ctran_->mapper->isendCtrl(
          op->allreduce.recvbuff, recvHdl, p, &req));
      interNodeLocalRecvbuffReq[n] = std::unique_ptr<CtranMapperRequest>(req);
    }
  }

  // Wait for all tmpbuff control messages to complete
  for (int n = 0; n < nNodes; n++) {
    if (n != node) {
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest(
          interNodeRemoteRecvbuffReq[n].get()));
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest(
          interNodeLocalRecvbuffReq[n].get()));
    }
  }

  auto [tmpBuf, tmpbufRegHdl] = comm->ctran_->algo->getTmpBufInfo(
      CtranAlgo::TmpbufType::INTERNODE_TMPBUF);

  size_t totalStepCount = (op->allreduce.count / nRanks) * nRanks;
  size_t stepCount = NCCL_CTRAN_INTERNODE_TMPBUF_SIZE * nLocalRanks / typeSize;
  if (stepCount > totalStepCount) {
    stepCount = totalStepCount;
  }
  // Get allreduce specific IB config
  static thread_local auto allReduceConfig =
      comm->ctran_->algo->getCollToVcConfig(CollType::ALLREDUCE);

  while (totalStepCount) {
    size_t chunkCount = stepCount / nRanks;
    size_t chunk = chunkCount * typeSize;
    size_t localOffset = localRank * stepCount * typeSize / nLocalRanks;
    size_t nodeOffset = node * chunk;

    /* Step 1: Intra-node reduce-scatter */
    elem = op->allreduce.kElemStepMap.at(
        static_cast<int>(ctran::allreduce::KernElemRole::kIntraReduceScatter));
    elem->localReduce.count = stepCount / nLocalRanks;
    elem->localReduce.dst = BUFOFFSET(op->allreduce.recvbuff, localOffset);
    for (int lr = 0; lr < nLocalRanks; lr++) {
      if (lr == localRank) {
        elem->localReduce.srcs[lr] =
            BUFOFFSET(op->allreduce.sendbuff, localOffset);
      } else {
        elem->localReduce.srcs[lr] =
            BUFOFFSET(intraNodeRemoteSendBuffs[lr], localOffset);
      }
    }

    elem->post();
    THROW_IF_ABORTED(elem->wait(comm->getAbort()));

    /* Step 2: Inter-node Reduce-scatter */
    /* wait for inter-node data transfer to perform local reduction */
    if (chunk) {
      for (int n = 0; n < nNodes; n++) {
        if (n != node) {
          int p = statex->localRankToRank(localRank, n);

          void* src =
              BUFOFFSET(op->allreduce.recvbuff, localOffset + n * chunk);
          auto [interNodeRemoteTmpbuff, interNodeRemoteTmpAccessKey] =
              comm->ctran_->algo->getRemoteTmpBufInfo(p);
          void* dst = BUFOFFSET(interNodeRemoteTmpbuff, node * chunk);

          CtranMapperRequest* req = nullptr;
          FB_COMMCHECK(comm->ctran_->mapper->iput(
              src,
              dst,
              chunk,
              p,
              CtranMapperConfig{
                  .memHdl_ = recvHdl,
                  .remoteAccessKey_ = interNodeRemoteTmpAccessKey,
                  .notify_ = true,
                  .ibConfig_ = allReduceConfig},
              &req));
          interNodePutReq[n] = std::unique_ptr<CtranMapperRequest>(req);
        }
      }

      for (int n = 0; n < nNodes; n++) {
        if (n != node) {
          int p = statex->localRankToRank(localRank, n);

          FB_COMMCHECK(
              comm->ctran_->mapper->waitRequest(interNodePutReq[n].get()));

          CtranMapperNotify notify;
          FB_COMMCHECK(
              comm->ctran_->mapper->initNotify(p, tmpbufRegHdl, &notify));
          FB_COMMCHECK(comm->ctran_->mapper->waitNotify(&notify));
        }
      }
    }

    /* local reduction from tmpbuf */
    elem = op->allreduce.kElemStepMap.at(
        static_cast<int>(ctran::allreduce::KernElemRole::kInterReduceScatter));
    elem->stridedReduce.dst =
        BUFOFFSET(op->allreduce.recvbuff, localOffset + nodeOffset);
    elem->stridedReduce.stridedSrc = tmpBuf;
    elem->stridedReduce.blockCount = chunkCount;
    elem->stridedReduce.stride = chunkCount;
    /* poke kernel to start the local reduction */
    elem->post();
    THROW_IF_ABORTED(elem->wait(comm->getAbort()));

    /* Step 3: Inter-node Allgather */
    /* wait for inter-node data transfer to perform local reduction */
    if (chunk) {
      for (int n = 0; n < nNodes; n++) {
        if (n != node) {
          int p = statex->localRankToRank(localRank, n);
          void* src =
              BUFOFFSET(op->allreduce.recvbuff, localOffset + nodeOffset);
          void* dst =
              BUFOFFSET(interNodeRemoteRecvBuffs[n], localOffset + nodeOffset);

          CtranMapperRequest* req = nullptr;
          FB_COMMCHECK(comm->ctran_->mapper->iput(
              src,
              dst,
              chunk,
              p,
              CtranMapperConfig{
                  .memHdl_ = recvHdl,
                  .remoteAccessKey_ = interNodeRemoteRecvAccessKeys[n],
                  .notify_ = true},
              &req));
          interNodePutReq[n] = std::unique_ptr<CtranMapperRequest>(req);
        }
      }

      for (int n = 0; n < nNodes; n++) {
        if (n != node) {
          int p = statex->localRankToRank(localRank, n);

          FB_COMMCHECK(
              comm->ctran_->mapper->waitRequest(interNodePutReq[n].get()));

          CtranMapperNotify notify;
          FB_COMMCHECK(comm->ctran_->mapper->initNotify(p, recvHdl, &notify));
          FB_COMMCHECK(comm->ctran_->mapper->waitNotify(&notify));
        }
      }
    }
    THROW_IF_ABORTED();

    /* Step 4: Intra-node Allgather */
    elem = op->allreduce.kElemStepMap.at(
        static_cast<int>(ctran::allreduce::KernElemRole::kIntraAllGather));
    elem->bcast.count = stepCount / nLocalRanks;
    elem->bcast.src = BUFOFFSET(op->allreduce.recvbuff, localOffset);
    for (int lr = 0; lr < nLocalRanks; lr++) {
      if (lr == localRank) {
        elem->bcast.dsts[lr] = BUFOFFSET(op->allreduce.recvbuff, localOffset);
      } else {
        elem->bcast.dsts[lr] =
            BUFOFFSET(intraNodeRemoteRecvBuffs[lr], localOffset);
      }
    }

    /* poke kernel to start the allgather */
    elem->post();
    THROW_IF_ABORTED(elem->wait(comm->getAbort()));

    op->allreduce.sendbuff =
        BUFOFFSET(op->allreduce.sendbuff, stepCount * typeSize);
    op->allreduce.recvbuff =
        BUFOFFSET(op->allreduce.recvbuff, stepCount * typeSize);
    for (int lr = 0; lr < nLocalRanks; lr++) {
      if (lr != localRank) {
        intraNodeRemoteSendBuffs[lr] =
            BUFOFFSET(intraNodeRemoteSendBuffs[lr], stepCount * typeSize);
        intraNodeRemoteRecvBuffs[lr] =
            BUFOFFSET(intraNodeRemoteRecvBuffs[lr], stepCount * typeSize);
      }
    }
    for (int n = 0; n < nNodes; n++) {
      if (n != node) {
        interNodeRemoteRecvBuffs[n] =
            BUFOFFSET(interNodeRemoteRecvBuffs[n], stepCount * typeSize);
      }
    }
    totalStepCount -= stepCount;
    if (stepCount > totalStepCount) {
      stepCount = totalStepCount;
    }
  }

  size_t remCount = op->allreduce.count % nRanks;

  if (remCount) {
    // NOTE: we have to decouple intra-node allreduce to (1) reduce to localRank
    // 0 and (2) bcast for in-place case

    /* Step 5: Intra-node reduce */
    elem = op->allreduce.kElemStepMap.at(
        static_cast<int>(ctran::allreduce::KernElemRole::kRemIntraReduce));
    elem->localReduce.dst = op->allreduce.recvbuff;
    for (int lr = 0; lr < nLocalRanks; lr++) {
      if (lr == localRank) {
        elem->localReduce.srcs[lr] = op->allreduce.sendbuff;
      } else {
        elem->localReduce.srcs[lr] = intraNodeRemoteSendBuffs[lr];
      }
    }
    elem->post();
    THROW_IF_ABORTED(elem->wait(comm->getAbort()));

    /* Step 6: Intra-node bcast */
    elem = op->allreduce.kElemStepMap.at(
        static_cast<int>(ctran::allreduce::KernElemRole::kRemIntraBcast));
    elem->bcast.src = op->allreduce.recvbuff;
    for (int lr = 0; lr < nLocalRanks; lr++) {
      if (lr == localRank) {
        elem->bcast.dsts[lr] = op->allreduce.recvbuff;
      } else {
        elem->bcast.dsts[lr] = intraNodeRemoteRecvBuffs[lr];
      }
    }
    elem->post();
    THROW_IF_ABORTED(elem->wait(comm->getAbort()));

    /* Step 7: Inter-node allreduce */
    /* wait for inter-node data transfer to perform local reduction */
    for (int n = 0; n < nNodes; n++) {
      if (n != node) {
        int p = statex->localRankToRank(localRank, n);

        void* src = op->allreduce.recvbuff;
        auto [interNodeRemoteTmpbuff, interNodeRemoteTmpAccessKey] =
            comm->ctran_->algo->getRemoteTmpBufInfo(p);
        void* dst =
            BUFOFFSET(interNodeRemoteTmpbuff, node * remCount * typeSize);

        CtranMapperRequest* req = nullptr;
        FB_COMMCHECK(comm->ctran_->mapper->iput(
            src,
            dst,
            remCount * typeSize,
            p,
            CtranMapperConfig{
                .memHdl_ = recvHdl,
                .remoteAccessKey_ = interNodeRemoteTmpAccessKey,
                .notify_ = true},
            &req));
        interNodePutReq[n] = std::unique_ptr<CtranMapperRequest>(req);
      }
    }

    for (int n = 0; n < nNodes; n++) {
      if (n != node) {
        int p = statex->localRankToRank(localRank, n);

        FB_COMMCHECK(
            comm->ctran_->mapper->waitRequest(interNodePutReq[n].get()));

        CtranMapperNotify notify;
        FB_COMMCHECK(
            comm->ctran_->mapper->initNotify(p, tmpbufRegHdl, &notify));
        FB_COMMCHECK(comm->ctran_->mapper->waitNotify(&notify));
      }
    }

    /* local reduction from tmpbuf */
    elem = op->allreduce.kElemStepMap.at(
        static_cast<int>(ctran::allreduce::KernElemRole::kRemInterReduce));
    elem->stridedReduce.dst = op->allreduce.recvbuff;
    elem->stridedReduce.stridedSrc = tmpBuf;
    elem->stridedReduce.blockCount = remCount;
    elem->stridedReduce.stride = remCount;
    elem->post();
    THROW_IF_ABORTED(elem->wait(comm->getAbort()));
  }

  if (localRegSend == true) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(sendHdl));
  }
  if (localRegRecv == true) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(recvHdl));
  }

  comm->ctran_->mapper->timestamps.emplace_back(std::move(timestamp));
  comm->ctran_->mapper->reportProfiling();

  return commSuccess;
}

commResult_t ctranAllReduceDirect(
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
      allReduceAlgoName(myAlgo).c_str(),
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
  int nLocalRanks = statex->nLocalRanks();
  int localRank = statex->localRank();
  int nRanks = statex->nRanks();
  int nNodes = statex->nNodes();
  int node = statex->node();

  if (nRanks == 1) {
    // For single-rank comm, allreduce is a no-op if sendbuff == recvbuff
    if (sendbuff == recvbuff) {
      return commSuccess;
    }

    // otherwise, allreduce is just a copy
    size_t size = count * commTypeSize(datatype);
    FB_CUDACHECK(
        cudaMemcpyAsync(recvbuff, sendbuff, size, cudaMemcpyDefault, stream));
    return commSuccess;
  }

  // Prevent buffer overflow in localReduce.srcs array
  if (nLocalRanks > CTRAN_MAX_NVL_PEERS) {
    CLOGF(
        ERR,
        "nLocalRanks ({}) exceeds CTRAN_MAX_NVL_PEERS ({}). This will cause buffer overflow in localReduce.srcs array! ",
        nLocalRanks,
        CTRAN_MAX_NVL_PEERS);
    return commInvalidUsage;
  }
  size_t typeSize = commTypeSize(datatype);
  void* sbuf = const_cast<void*>(sendbuff);
  void* dbuf = recvbuff;

  size_t totalStepCount = (count / nRanks) * nRanks;
  size_t stepCount = NCCL_CTRAN_INTERNODE_TMPBUF_SIZE * nLocalRanks / typeSize;
  if (stepCount > totalStepCount) {
    stepCount = totalStepCount;
  }
  size_t nSteps = stepCount ? (totalStepCount + stepCount - 1) / stepCount : 0;
  size_t remCount = count % nRanks;

  FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());

  // FIXME: We perform an extra copy here before we submit to the GPE
  // thread.  Ideally we should be doing this copy inside the GPE
  // thread, but that requires two changes first: (1) our
  // searchRegHandle cannot try to dynamically register the buffer (as
  // that will fail); and (2) we need a copy kernel which does not
  // currently exist.
  if (count * typeSize < CTRAN_MIN_REGISTRATION_SIZE) {
    sbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_SRC_TMPBUF);
    dbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_DST_TMPBUF);
    FB_CUDACHECK(cudaMemcpyAsync(
        sbuf, sendbuff, count * typeSize, cudaMemcpyDefault, stream));
  }

  op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::ALLREDUCE, comm, opCount));
  op->allreduce.sendbuff = reinterpret_cast<const void*>(sbuf);
  op->allreduce.recvbuff = dbuf;
  op->allreduce.count = count;
  op->allreduce.datatype = datatype;
  op->allreduce.op = redOp;

  XCHECK(typeToFunc.contains(std::make_pair(datatype, redOp)))
      << "typeToFunc does not contain datatype " << datatype << " with op "
      << redOp;
  const void* func = typeToFunc.at(std::make_pair(datatype, redOp));

  auto config = KernelConfig(
      KernelConfig::KernelType::ALLREDUCE,
      stream,
      allReduceAlgoName(myAlgo),
      opCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.args.collective.allreduce.sendbuff =
      reinterpret_cast<const void*>(sbuf);
  config.args.collective.allreduce.recvbuff = dbuf;
  config.args.collective.allreduce.redOp = redOp;
  config.args.collective.allreduce.count = count;
  config.args.collective.allreduce.nSteps = nSteps;
  config.args.collective.allreduce.datatype = datatype;

  // Reset kernel elements
  config.args.collective.allreduce.kernelElems[static_cast<int>(
      ctran::allreduce::KernElemRole::kIntraReduceScatter)] = nullptr;
  config.args.collective.allreduce.kernelElems[static_cast<int>(
      ctran::allreduce::KernElemRole::kInterReduceScatter)] = nullptr;
  config.args.collective.allreduce.kernelElems[static_cast<int>(
      ctran::allreduce::KernElemRole::kIntraAllGather)] = nullptr;
  config.args.collective.allreduce.kernelElems[static_cast<int>(
      ctran::allreduce::KernElemRole::kRemIntraReduce)] = nullptr;
  config.args.collective.allreduce.kernelElems[static_cast<int>(
      ctran::allreduce::KernElemRole::kRemIntraBcast)] = nullptr;
  config.args.collective.allreduce.kernelElems[static_cast<int>(
      ctran::allreduce::KernElemRole::kRemInterReduce)] = nullptr;

  FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
      (int*)&config.numBlocks,
      (int*)&config.numThreads,
      func,
      0 /* dynamicSMemSize */,
      0 /* blockSizeLimit */));
  if (config.numBlocks > NCCL_CTRAN_ALLREDUCE_DIRECT_MAX_NUM_THREAD_BLOCKS) {
    config.numBlocks = NCCL_CTRAN_ALLREDUCE_DIRECT_MAX_NUM_THREAD_BLOCKS;
  }
  if (config.numThreads > NCCL_CTRAN_ALLREDUCE_DIRECT_THREAD_BLOCK_SIZE) {
    config.numThreads = NCCL_CTRAN_ALLREDUCE_DIRECT_THREAD_BLOCK_SIZE;
  }

  // Prepare kElems for first segment symmetrically handled by each rank
  if (nSteps > 0) {
    KernelElem* elem = nullptr;
    FB_COMMCHECK(
        comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, &elem));
    elem->localReduce.nvectors = nLocalRanks;
    config.args.collective.allreduce.kernelElems[static_cast<int>(
        ctran::allreduce::KernElemRole::kIntraReduceScatter)] = elem;
    op->allreduce.kElemStepMap[static_cast<int>(
        ctran::allreduce::KernElemRole::kIntraReduceScatter)] = elem;

    FB_COMMCHECK(
        comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, &elem));
    // local node's block is in recvbuf, specify it as an "inplace" block in
    // stridedSrc.
    elem->stridedReduce.inplaceBlockIdx = node;
    elem->stridedReduce.numBlocks = nNodes;
    // ensure result is visible to step-3 inter-node allgather
    elem->stridedReduce.flushMem = true;
    config.args.collective.allreduce.kernelElems[static_cast<int>(
        ctran::allreduce::KernElemRole::kInterReduceScatter)] = elem;
    op->allreduce.kElemStepMap[static_cast<int>(
        ctran::allreduce::KernElemRole::kInterReduceScatter)] = elem;

    FB_COMMCHECK(
        comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, &elem));
    config.args.collective.allreduce.kernelElems[static_cast<int>(
        ctran::allreduce::KernElemRole::kIntraAllGather)] = elem;
    op->allreduce.kElemStepMap[static_cast<int>(
        ctran::allreduce::KernElemRole::kIntraAllGather)] = elem;
  }

  if (remCount > 0) {
    KernelElem* elem = nullptr;
    // NOTE: we have to decouple intra-node allreduce to (1) reduce to localRank
    // 0 and (2) bcast for in-place case

    /* fifth step: intra-node reduce */
    FB_COMMCHECK(
        comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, &elem));
    elem->localReduce.nvectors = nLocalRanks;
    // Reduce with root = 0
    elem->localReduce.count = localRank ? 0 : remCount;
    config.args.collective.allreduce.kernelElems[static_cast<int>(
        ctran::allreduce::KernElemRole::kRemIntraReduce)] = elem;
    op->allreduce.kElemStepMap[static_cast<int>(
        ctran::allreduce::KernElemRole::kRemIntraReduce)] = elem;

    /* sixth step: intra-node bcast */
    FB_COMMCHECK(
        comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, &elem));
    // Bcast with root = 0
    elem->bcast.count = localRank ? 0 : remCount;
    config.args.collective.allreduce.kernelElems[static_cast<int>(
        ctran::allreduce::KernElemRole::kRemIntraBcast)] = elem;
    op->allreduce.kElemStepMap[static_cast<int>(
        ctran::allreduce::KernElemRole::kRemIntraBcast)] = elem;

    /* seventh step: inter-node allreduce */
    FB_COMMCHECK(
        comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, &elem));
    // local node's block is in recvbuf, specify it as an "inplace" block in
    // stridedSrc
    elem->stridedReduce.inplaceBlockIdx = node;
    elem->stridedReduce.numBlocks = nNodes;
    config.args.collective.allreduce.kernelElems[static_cast<int>(
        ctran::allreduce::KernElemRole::kRemInterReduce)] = elem;
    op->allreduce.kElemStepMap[static_cast<int>(
        ctran::allreduce::KernElemRole::kRemInterReduce)] = elem;
  }

  opGroup.push_back(std::move(op));

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup), impl, config, func, timeout));

  if (count * typeSize < CTRAN_MIN_REGISTRATION_SIZE) {
    FB_CUDACHECK(cudaMemcpyAsync(
        recvbuff, dbuf, count * typeSize, cudaMemcpyDefault, stream));
  }

  return commSuccess;
}
