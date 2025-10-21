// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_fp16.h>
#include <deque>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/logger/LogUtils.h"

CTRAN_DATATYPE_REDOP_TO_FUNC_MAPPER(
    reduceScatterKerns,
    ncclKernelReduceScatterDirect);

static const auto myAlgo = NCCL_REDUCESCATTER_ALGO::ctdirect;

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t size =
      op->reducescatter.recvcount * commTypeSize(op->reducescatter.datatype);
  CtranComm* comm = opGroup.front()->comm_;
  auto mapper = comm->ctran_->mapper.get();

  const auto& statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  const int nLocalRanks = statex->nLocalRanks();

  void* sendHdl;
  std::vector<void*> remoteSendBuffs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteSendAccessKeys(nRanks);

  CtranAlgoLogger logger(reduceScatterAlgoName(myAlgo), op->opCount, comm);

  bool localRegSend;
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(reduceScatterAlgoName(myAlgo)));

  FB_COMMCHECK(mapper->searchRegHandle(
      op->reducescatter.sendbuff, size * nRanks, &sendHdl, &localRegSend));

  CtranMapperContext context(
      reduceScatterAlgoName(myAlgo), size * nRanks, size);
  mapper->setContext(std::move(context));

  // Intra-node reduce
  // Issue control messages within the node and wait for completion
  // - Exchange for remote access to send buffer
  FB_COMMCHECK(mapper->intraAllGatherCtrl(
      op->reducescatter.sendbuff,
      sendHdl,
      remoteSendBuffs,
      remoteSendAccessKeys));

  // Set src from other local ranks and kickoff local reduce
  auto elem = op->reducescatter.intraReduce.at(0);
  size_t srcOffset = rank * size;
  for (int r = 0; r < nLocalRanks; r++) {
    elem->reduce.srcs[r] =
        reinterpret_cast<const char*>(remoteSendBuffs[r]) + srcOffset;
  }
  // Post to kernel
  elem->post();

  // Wait last elem with barrier to complete, indicating all local ranks have
  // done. It ensures safe deregistration or buffer release after this kernel
  elem->wait();

  if (localRegSend == true) {
    FB_COMMCHECK(mapper->deregDynamic(sendHdl));
  }

  mapper->timestamps.emplace_back(std::move(timestamp));
  mapper->reportProfiling();

  return commSuccess;
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
            ncclKernelReduceScatterDirect<int, commSum>),
        0 /* dynamicSMemSize */,
        0 /* blockSizeLimit */));
  }

  return NCCL_CTRAN_REDUCESCATTER_THREAD_BLOCK_SIZE == -1
      ? bestThreadBlockSize
      : NCCL_CTRAN_REDUCESCATTER_THREAD_BLOCK_SIZE;
}

static inline int getNumGroups(size_t count, int nvectors) {
  // compute needed thread blocks for given bytes
  int nGroups = (count * nvectors) /
      NCCL_CTRAN_REDUCESCATTER_REDUCE_NELEM_PER_THREAD_BLOCK;
  return std::min(
      std::max(1, nGroups), // at least 1 thread block
      // not exceed max theshold
      NCCL_CTRAN_REDUCESCATTER_DIRECT_MAX_NUM_THREAD_BLOCKS);
}

static inline bool
useStageCopy(size_t recvcount, commDataType_t datatype, CtranComm* comm) {
  size_t maxSendBytes = std::min(
      NCCL_CTRAN_REDUCESCATTER_DIRECT_MIN_SIZE * comm->statex_->nRanks(),
      NCCL_CTRAN_BCAST_NVL_SHARED_DEVBUF_SIZE);
  size_t sendBytes =
      recvcount * commTypeSize(datatype) * comm->statex_->nRanks();
  return sendBytes <= maxSendBytes;
}

static inline commResult_t setupPlan(
    CtranComm* comm,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    KernelConfig& config) {
  auto op = opGroup.front().get();
  const auto statex = comm->statex_.get();
  config.args.devState_d = comm->ctran_->algo->getDevState();

  const int nLocalRanks = statex->nLocalRanks();

  // Intra-node reduce.
  // Each rank handles the reduce of its portion.
  int nGroups = getNumGroups(op->reducescatter.recvcount, nLocalRanks);

  if (useStageCopy(
          op->reducescatter.recvcount, op->reducescatter.datatype, comm)) {
    // For small msg, skip GPE side submit and launches only kernel
    // Thus, no need to setup kernel args
    config.args.collective.reducescatter.intraReduce = nullptr;
    config.args.collective.reducescatter.stageCopy = true;
  } else {
    // For large msg,  directly reduce from peer sendbuff which requires
    // GPE thread to exchange the buffer and pass to kernel
    KernelElem* elem = nullptr;
    FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, nGroups, &elem));
    // skip redElem argument setup since it is already known in kernel
    op->reducescatter.intraReduce[0] = elem;
    config.args.collective.reducescatter.intraReduce = elem;
    config.args.collective.reducescatter.stageCopy = false;
  }

  // Kernel doesn't use them, but for colltrace to record only
  config.args.collective.reducescatter.sendbuff = op->reducescatter.sendbuff;
  config.args.collective.reducescatter.recvbuff = op->reducescatter.recvbuff;
  config.args.collective.reducescatter.datatype = op->reducescatter.datatype;
  config.args.collective.reducescatter.recvcount = op->reducescatter.recvcount;

  config.numBlocks = nGroups;
  config.numThreads = getThreadBlockSize();
  return commSuccess;
}

commResult_t ctranReduceScatterDirect(
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

  if (comm->statex_->nNodes() > 1) {
    CLOGF(
        ERR,
        "ctranReduceScatterDirect supports only single node, but nNodes={}",
        comm->statex_->nNodes());
    return commInternalError;
  }

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

  // For small msg, skip GPE side submit and launches only kernel.
  // We copy entire sendbuff to pre-IPC shared tmpBuf before kernel launch. It
  // avoids overhead from GPE-kernel sync and ctrl msg exchange, which can be
  // more expensive than D2D stage copy.
  if (useStageCopy(recvcount, datatype, comm)) {
    opGroup.clear();
  }

  const void* func = reduceScatterKerns.at(std::make_pair(datatype, redOp));
  FB_COMMCHECK(
      comm->ctran_->gpe->submit(std::move(opGroup), impl, config, func));

  return commSuccess;
}
