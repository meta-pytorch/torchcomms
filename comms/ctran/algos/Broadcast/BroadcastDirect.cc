// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>
#include <memory>
#include <vector>

#include "comms/ctran/algos/Broadcast/BroadcastImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/logger/LogUtils.h"

static unsigned int bestThreadBlockSize = 0;
static const auto myAlgo = NCCL_BROADCAST_ALGO::ctdirect;

static inline int getNumGroups(size_t nbytes) {
  // compute needed thread blocks for given bytes
  int nGroups = nbytes / NCCL_CTRAN_NVL_BROADCAST_CHUNK_SIZE;
  return std::min(
      std::max(1, nGroups), // at least 1 thread block
      // not exceed max theshold
      NCCL_CTRAN_NVL_BROADCAST_MAX_NUM_THREAD_BLOCKS);
}

static inline unsigned int getThreadBlockSize() {
  // If first time call, query cuda recommended blockSize
  if (bestThreadBlockSize == 0) {
    int minGridSize = 0;
    FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        (int*)&bestThreadBlockSize,
        reinterpret_cast<const void*>(ncclKernelBroadcast</*UNPACK=*/false>),
        0 /* dynamicSMemSize */,
        0 /* blockSizeLimit */));
  }

  return NCCL_CTRAN_NVL_BROADCAST_THREAD_BLOCK_SIZE == -1
      ? bestThreadBlockSize
      : NCCL_CTRAN_NVL_BROADCAST_THREAD_BLOCK_SIZE;
}

static inline commResult_t setupPlan(
    CtranComm* comm,
    std::vector<std::unique_ptr<OpElem>>& opGroup,
    KernelConfig& config) {
  const auto statex = comm->statex_.get();
  struct OpElem* op = opGroup.front().get();
  size_t sendSize = op->broadcast.count * commTypeSize(op->broadcast.datatype);
  int root = op->broadcast.root;
  int nRanks = statex->nRanks();
  int myRank = statex->rank();
  int maxNumBlocks = 1;

  auto putNotifyList = CommonList<KernelElem>();
  auto waitNotifyList = CommonList<KernelElem>();

  if (sendSize == 0 || nRanks == 1) {
    return commSuccess;
  }

  KernelElem* elem = nullptr;
  int nGroups = getNumGroups(sendSize);
  // record the max number of thread blocks as final launching grid size
  maxNumBlocks = std::max(maxNumBlocks, nGroups);

  if (myRank == root) {
    for (int p = 0; p < nRanks; p++) {
      if (comm->ctran_->mapper->getBackend(p) == CtranMapperBackend::NVL &&
          p != myRank) {
        FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, nGroups, &elem));

        elem->putNotify.sendbuff = op->broadcast.sendbuff;
        elem->putNotify.nbytes = sendSize;
        elem->putNotify.peerLocalRank = statex->localRank(p);
        elem->putNotify.ngroups = nGroups;
        elem->putNotify.notify =
            true; // each put will be notified to remote peer
        putNotifyList.enqueue(elem);
        op->broadcast.putNotifyMap.insert({p, elem});
      }
    }
  } else {
    if (comm->ctran_->mapper->getBackend(root) == CtranMapperBackend::NVL) {
      // only 1 group handles waitNotify elem
      FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, 1, &elem));
      elem->waitNotify.peerLocalRank = statex->localRank(root);

      // pass the ngroups used by remote put
      elem->waitNotify.ngroups = getNumGroups(sendSize);

      waitNotifyList.enqueue(elem);
      op->broadcast.waitNotifyMap.insert({root, elem});
    }
  }

  if (putNotifyList.count > 0) {
    // Allow user to increase SM usuage for putNotify involved kernel
    config.numBlocks = maxNumBlocks;
    config.numThreads = getThreadBlockSize();
  }

  config.args.collective.broadcast.putNotifyList = putNotifyList.head;
  config.args.collective.broadcast.waitNotifyList = waitNotifyList.head;

  return commSuccess;
}

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t sendSize = op->broadcast.count * commTypeSize(op->broadcast.datatype);
  CtranComm* comm = opGroup.front()->comm_;
  const auto statex = comm->statex_.get();
  int root = op->broadcast.root;
  int nRanks = statex->nRanks();
  int myRank = statex->rank();
  void *sendHdl, *recvHdl;
  std::vector<void*> remoteRecvBuffs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;
  std::vector<std::unique_ptr<CtranMapperRequest>> irecvReq;
  std::vector<std::unique_ptr<CtranMapperRequest>> iputReq;
  bool localRegSend, localRegRecv;
  CtranMapper* mapper = comm->ctran_->mapper.get();
  CtranMapperRequest* req = nullptr;

  CtranAlgoLogger logger(broadcastAlgoName(myAlgo), op->opCount, comm);

  auto& putNotifyMap = op->broadcast.putNotifyMap;
  auto& waitNotifyMap = op->broadcast.waitNotifyMap;

  CtranMapperContext context("CtranBroadcast", sendSize, sendSize);
  mapper->setContext(std::move(context));
  for (int p = 0; p < nRanks; ++p) {
    remoteAccessKeys.emplace_back();
  }

  if (nRanks == 1) {
    return commSuccess;
  }

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(broadcastAlgoName(myAlgo)));

  if (myRank == root) {
    FB_COMMCHECK(mapper->searchRegHandle(
        op->broadcast.sendbuff, sendSize, &sendHdl, &localRegSend));

    for (int p = 0; p < nRanks; p++) {
      if (p != root) {
        FB_COMMCHECK(mapper->irecvCtrl(
            &remoteRecvBuffs[p], &remoteAccessKeys[p], p, &req));
        irecvReq.push_back(std::unique_ptr<CtranMapperRequest>(req));
      }
    }

    // Complete control messages with receive buffers and issue put operations
    // that match
    while (!irecvReq.empty()) {
      auto it = irecvReq.begin();
      while (it != irecvReq.end()) {
        auto& recvCtrlReq = *it;
        int peer = recvCtrlReq->peer;

        bool isComplete;
        FB_COMMCHECK(
            comm->ctran_->mapper->testRequest(recvCtrlReq.get(), &isComplete));

        if (isComplete) {
          timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));

          KernelElem* elem = nullptr;
          if (comm->ctran_->mapper->getBackend(peer) ==
              CtranMapperBackend::NVL) {
            if (putNotifyMap.contains(peer)) {
              elem = putNotifyMap[peer];
            } else {
              CLOGF(
                  WARN,
                  "Expecting NVLink putNotify for peer {}. Something bad probably happened.",
                  peer);
            }
          }

          FB_COMMCHECK(mapper->iput(
              op->broadcast.sendbuff,
              (void*)((uintptr_t)remoteRecvBuffs[peer]),
              sendSize,
              peer,
              CtranMapperConfig{
                  .memHdl_ = sendHdl,
                  .remoteAccessKey_ = remoteAccessKeys[peer],
                  .notify_ = true,
                  .kernElem_ = elem},
              &req));
          iputReq.push_back(std::unique_ptr<CtranMapperRequest>(req));
          timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
          irecvReq.erase(it);
        } else {
          it++;
        }
      }
    }

    // Wait for all puts to complete
    while (!iputReq.empty()) {
      FB_COMMCHECK(comm->ctran_->mapper->testSomeRequests(
          iputReq, timestamp->putComplete));
    }

    if (localRegSend == true) {
      FB_COMMCHECK(mapper->deregDynamic(sendHdl));
    }
  } else {
    KernelElem* elem = nullptr;
    if (comm->ctran_->mapper->getBackend(root) == CtranMapperBackend::NVL) {
      if (waitNotifyMap.contains(root)) {
        elem = waitNotifyMap[root];
      } else {
        CLOGF(
            WARN,
            "Expecting NVLink waitNotify for root {}. Something bad probably happened.",
            root);
      }
    }

    FB_COMMCHECK(mapper->searchRegHandle(
        op->broadcast.recvbuff, sendSize, &recvHdl, &localRegRecv));

    FB_COMMCHECK(
        mapper->isendCtrl(op->broadcast.recvbuff, recvHdl, root, &req));

    // Initialize notify flag to receive from root
    auto notifyRoot = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(mapper->initNotify(root, recvHdl, elem, notifyRoot.get()));

    FB_COMMCHECK(mapper->waitRequest(req));

    // Wait for the put from the sender to complete
    FB_COMMCHECK(mapper->waitNotify(notifyRoot.get()));

    if (localRegRecv == true) {
      FB_COMMCHECK(mapper->deregDynamic(recvHdl));
    }
  }

  mapper->timestamps.emplace_back(std::move(timestamp));
  mapper->reportProfiling();

  return commSuccess;
}

commResult_t ctranBroadcastDirect(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int root,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      broadcastAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      count,
      datatype,
      root,
      comm,
      stream);
  const auto statex = comm->statex_.get();

  if (sendbuff != recvbuff && statex->rank() == root) {
    FB_COMMCHECK(comm->ctran_->mapper->icopy(
        recvbuff, sendbuff, count * commTypeSize(datatype), stream));
  }

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;

  size_t typeSize = commTypeSize(datatype);
  void* sbuf = const_cast<void*>(sendbuff);
  void* dbuf = recvbuff;

  // FIXME: We perform an extra copy here before we submit to the GPE
  // thread.  Ideally we should be doing this copy inside the GPE
  // thread, but that requires two changes first: (1) our
  // searchRegHandle cannot try to dynamically register the buffer (as
  // that will fail); and (2) we need a copy kernel which does not
  // currently exist.
  if (count * typeSize < CTRAN_MIN_REGISTRATION_SIZE) {
    // make sure tmpbuf is allocated and registered
    FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());
    sbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_SRC_TMPBUF);
    dbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_DST_TMPBUF);
    FB_CUDACHECK(cudaMemcpyAsync(
        sbuf, sendbuff, count * typeSize, cudaMemcpyDefault, stream));
  }

  op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::BROADCAST, comm, opCount));
  op->broadcast.sendbuff = reinterpret_cast<const void*>(sbuf);
  op->broadcast.recvbuff = dbuf;
  op->broadcast.count = count;
  op->broadcast.datatype = datatype;
  op->broadcast.root = root;
  opGroup.push_back(std::move(op));

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::BROADCAST,
      stream,
      broadcastAlgoName(myAlgo),
      opCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.args.collective.broadcast.sendbuff =
      reinterpret_cast<const void*>(sbuf);
  config.args.collective.broadcast.recvbuff = dbuf;
  config.args.collective.broadcast.datatype = datatype;
  config.args.collective.broadcast.count = count;

  FB_COMMCHECK(setupPlan(comm, opGroup, config));
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      impl,
      config,
      reinterpret_cast<void*>(ncclKernelBroadcast</*UNPACK=*/false>)));

  if (count * typeSize < CTRAN_MIN_REGISTRATION_SIZE &&
      statex->rank() != root) {
    FB_CUDACHECK(cudaMemcpyAsync(
        recvbuff, dbuf, count * typeSize, cudaMemcpyDefault, stream));
  }

  return commSuccess;
}
