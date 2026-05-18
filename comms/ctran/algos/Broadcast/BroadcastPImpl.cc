// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/Broadcast/BroadcastPImpl.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/Broadcast/BroadcastImpl.h"
#include "comms/ctran/algos/Broadcast/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/logger/LogUtils.h"

using ctran::broadcastp::AlgoImpl;
using ctran::broadcastp::PersistArgs;

namespace {

static unsigned int bestThreadBlockSize = 0;

static inline int getNumGroups(size_t nbytes) {
  int nGroups = nbytes / NCCL_CTRAN_NVL_BROADCAST_CHUNK_SIZE;
  return std::min(
      std::max(1, nGroups), NCCL_CTRAN_NVL_BROADCAST_MAX_NUM_THREAD_BLOCKS);
}

static inline unsigned int getThreadBlockSize() {
  if (bestThreadBlockSize == 0) {
    int minGridSize = 0;
    FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        (int*)&bestThreadBlockSize,
        reinterpret_cast<const void*>(ncclKernelBroadcast</*UNPACK=*/false>),
        0,
        0));
  }
  return NCCL_CTRAN_NVL_BROADCAST_THREAD_BLOCK_SIZE == -1
      ? bestThreadBlockSize
      : NCCL_CTRAN_NVL_BROADCAST_THREAD_BLOCK_SIZE;
}

commResult_t gpeFn(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = op->comm_;
  const auto statex = comm->statex_.get();
  const int nRanks = statex->nRanks();
  const int myRank = statex->rank();
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->broadcastP.pArgs);
  const int root = op->broadcastP.root;
  const size_t sendSize = op->broadcastP.count * commTypeSize(pArgs->datatype);
  CtranMapper* mapper = comm->ctran_->mapper.get();

  auto& putNotifyMap = op->broadcastP.putNotifyMap;
  auto& waitNotifyMap = op->broadcastP.waitNotifyMap;

  if (nRanks == 1 || sendSize == 0) {
    return commSuccess;
  }

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::make_unique<CtranMapperTimestamp>("CtranBroadcastP");

  CtranMapperContext context("CtranBroadcastP", sendSize, sendSize);
  mapper->setContext(std::move(context));

  if (myRank == root) {
    void* sendHdl = nullptr;
    bool localRegSend = false;
    FB_COMMCHECK(mapper->searchRegHandle(
        op->broadcastP.sendbuff, sendSize, &sendHdl, &localRegSend));

    std::vector<std::unique_ptr<CtranMapperRequest>> iputReqs;

    for (int p = 0; p < nRanks; p++) {
      if (p == root) {
        continue;
      }

      KernelElem* elem = nullptr;
      auto it = putNotifyMap.find(p);
      if (it != putNotifyMap.end()) {
        elem = it->second;
      }

      auto req = std::make_unique<CtranMapperRequest>();
      FB_COMMCHECK(mapper->iput(
          op->broadcastP.sendbuff,
          pArgs->remoteRecvBuffs[p],
          sendSize,
          p,
          CtranMapperConfig{
              .memHdl_ = sendHdl,
              .remoteAccessKey_ = pArgs->remoteAccessKeys[p],
              .notify_ = true,
              .kernElem_ = elem},
          req.get()));
      iputReqs.push_back(std::move(req));
      timestamp->putIssued.push_back(CtranMapperTimestampPoint(p));
    }

    while (!iputReqs.empty()) {
      FB_COMMCHECK(mapper->testSomeRequests(iputReqs, timestamp->putComplete));
    }

    if (localRegSend) {
      FB_COMMCHECK(mapper->deregDynamic(sendHdl));
    }
  } else {
    KernelElem* elem = nullptr;
    auto it = waitNotifyMap.find(root);
    if (it != waitNotifyMap.end()) {
      elem = it->second;
    }

    auto notifyRoot = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(
        mapper->initNotify(root, pArgs->recvHdl, elem, notifyRoot.get()));

    if (!pArgs->skipCtrlMsg) {
      CtranMapperRequest sendReq;
      FB_COMMCHECK(
          mapper->isendCtrl(pArgs->recvbuff, pArgs->recvHdl, root, &sendReq));
      FB_COMMCHECK(mapper->waitRequest(&sendReq));
    }

    FB_COMMCHECK(mapper->waitNotify(notifyRoot.get()));
  }

  mapper->timestamps.emplace_back(std::move(timestamp));
  mapper->reportProfiling();
  return commSuccess;
}

} // namespace

namespace ctran::broadcastp {

commResult_t
AlgoImpl::exec(const void* sendbuff, const size_t count, const int root) {
  const auto statex = comm_->statex_.get();
  const int nRanks = statex->nRanks();
  const int myRank = statex->rank();
  const size_t sendSize = count * commTypeSize(pArgs.datatype);
  auto opCount = comm_->ctran_->getOpCount();

  CTRAN_COLL_INFO(
      "CtranBroadcastP",
      sendbuff,
      pArgs.recvbuff,
      count,
      pArgs.datatype,
      root,
      comm_,
      stream_);

  if (count == 0 || nRanks == 1) {
    return commSuccess;
  }
  if (count > pArgs.maxRecvCount) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "BroadcastP count {} exceeds maximum count {}.",
        count,
        pArgs.maxRecvCount);
  }

  if (myRank == root && sendbuff != pArgs.recvbuff) {
    FB_COMMCHECK(comm_->ctran_->mapper->icopy(
        pArgs.recvbuff, sendbuff, sendSize, stream_));
  }

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::BROADCAST, stream_, "CtranBroadcastP", opCount);
  config.args.devState_d = comm_->ctran_->algo->getDevState();
  config.args.collective.broadcast.sendbuff = sendbuff;
  config.args.collective.broadcast.recvbuff = pArgs.recvbuff;
  config.args.collective.broadcast.datatype = pArgs.datatype;
  config.args.collective.broadcast.count = count;

  auto putNotifyList = CommonList<KernelElem>();
  auto waitNotifyList = CommonList<KernelElem>();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto op = std::make_unique<OpElem>(
      OpElem::opType::BROADCASTP, stream_, comm_, opCount);
  op->broadcastP.pArgs = &pArgs;
  op->broadcastP.sendbuff = sendbuff;
  op->broadcastP.count = count;
  op->broadcastP.root = root;

  int nGroups = getNumGroups(sendSize);
  int maxNumBlocks = 1;

  if (myRank == root) {
    for (int p = 0; p < nRanks; p++) {
      if (comm_->ctran_->mapper->getBackend(p) == CtranMapperBackend::NVL &&
          p != myRank) {
        KernelElem* elem = nullptr;
        FB_COMMCHECK(comm_->ctran_->gpe->allocKernelElems(1, nGroups, &elem));
        elem->putNotify.sendbuff = sendbuff;
        elem->putNotify.nbytes = sendSize;
        elem->putNotify.peerLocalRank = statex->localRank(p);
        elem->putNotify.ngroups = nGroups;
        elem->putNotify.notify = true;
        putNotifyList.enqueue(elem);
        op->broadcastP.putNotifyMap.insert({p, elem});
      }
    }
    if (putNotifyList.count > 0) {
      maxNumBlocks = std::max(maxNumBlocks, nGroups);
      config.numBlocks = maxNumBlocks;
      config.numThreads = getThreadBlockSize();
    }
  } else {
    if (comm_->ctran_->mapper->getBackend(root) == CtranMapperBackend::NVL) {
      KernelElem* elem = nullptr;
      FB_COMMCHECK(comm_->ctran_->gpe->allocKernelElems(1, 1, &elem));
      elem->waitNotify.peerLocalRank = statex->localRank(root);
      elem->waitNotify.ngroups = getNumGroups(sendSize);
      waitNotifyList.enqueue(elem);
      op->broadcastP.waitNotifyMap.insert({root, elem});
    }
  }

  config.args.collective.broadcast.putNotifyList = putNotifyList.head;
  config.args.collective.broadcast.waitNotifyList = waitNotifyList.head;

  opGroup.push_back(std::move(op));
  FB_COMMCHECK(comm_->ctran_->gpe->submit(
      std::move(opGroup),
      gpeFn,
      config,
      reinterpret_cast<void*>(ncclKernelBroadcast</*UNPACK=*/false>)));

  return commSuccess;
}

} // namespace ctran::broadcastp
