// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <deque>
#include <optional>

#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/SendRecv/SendRecvImpl.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/logger/LogUtils.h"

static thread_local std::deque<OpElem*> CtranOpGroup;

std::unordered_map<KernelConfig::KernelType, void*> kernelFns = {
    {KernelConfig::KernelType::SEND, reinterpret_cast<void*>(ncclKernelSend)},
    {KernelConfig::KernelType::RECV,
     reinterpret_cast<void*>(ncclKernelRecv</*UNPACK=*/false>)},
    {KernelConfig::KernelType::SENDRECV,
     reinterpret_cast<void*>(ncclKernelSendRecv</*UNPACK=*/false>)},
    {KernelConfig::KernelType::SEND_NOTIFY,
     reinterpret_cast<void*>(ncclKernelSendNotifyOnly)},
    {KernelConfig::KernelType::RECV_NOTIFY,
     reinterpret_cast<void*>(ncclKernelRecvNotifyOnly)},
    {KernelConfig::KernelType::SENDRECV_NOTIFY,
     reinterpret_cast<void*>(ncclKernelSendRecvNotifyOnly)},
    {KernelConfig::KernelType::RECV_UNPACK,
     reinterpret_cast<void*>(ncclKernelRecv</*UNPACK=*/true>)},
    {KernelConfig::KernelType::SENDRECV_UNPACK,
     reinterpret_cast<void*>(ncclKernelSendRecv</*UNPACK=*/true>)},
};

static const auto myAlgo = NCCL_SENDRECV_ALGO::ctran;

// The result of the exchange: send op's op->send.recvbuff and
// op->send.remoteAccessKey are updated to the remote buffer address and remote
// access key.
// NOTE: users should pre-register the send/recv buffers before calling this
// function
static commResult_t exchangeSendRecvHandles(
    const std::vector<std::unique_ptr<OpElem>>& opGroup) {
  if (opGroup.empty()) {
    return commSuccess;
  }
  auto& firstOp = opGroup.front();
  const auto comm = firstOp->comm_;
  std::vector<OpElem*> sendOpGroup, recvOpGroup;

  for (auto& op : opGroup) {
    if (op->type == OpElem::opType::SEND) {
      sendOpGroup.push_back(op.get());
    } else {
      recvOpGroup.push_back(op.get());
    }
  }

  auto& mapper = comm->ctran_->mapper;
  std::vector<void*> remoteRecvBuff(sendOpGroup.size());
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKey(
      sendOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperRequest>> sendCtrlReqs(
      sendOpGroup.size());

  std::vector<void*> recvMemHdl(recvOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperRequest>> recvCtrlReqs(
      recvOpGroup.size());

  std::vector<void*> tmpRegHdls;

  for (auto i = 0; i < sendOpGroup.size(); i++) {
    auto& op = sendOpGroup[i];

    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->irecvCtrl(
        &remoteRecvBuff[i], &remoteAccessKey[i], op->send.peerRank, &req));
    sendCtrlReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
  }

  for (auto i = 0; i < recvOpGroup.size(); i++) {
    auto& op = recvOpGroup[i];
    size_t recvSize = op->recv.count * commTypeSize(op->recv.datatype);
    bool localReg = false;

    FB_COMMCHECK(mapper->searchRegHandle(
        op->recv.recvbuff, recvSize, &recvMemHdl[i], &localReg));

    if (localReg) {
      tmpRegHdls.push_back(recvMemHdl[i]);
    }

    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->isendCtrl(
        op->recv.recvbuff, recvMemHdl[i], op->recv.peerRank, &req));
    recvCtrlReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
  }

  for (auto i = 0; i < recvOpGroup.size(); i++) {
    FB_COMMCHECK(mapper->waitRequest(recvCtrlReqs[i].get()));
  }

  for (auto i = 0; i < sendOpGroup.size(); i++) {
    FB_COMMCHECK(mapper->waitRequest(sendCtrlReqs[i].get()));
    auto& op = sendOpGroup[i];
    op->send.recvbuff->store(remoteRecvBuff[i]);
    if (remoteAccessKey[i].backend == CtranMapperBackend::NVL) {
      op->send.remoteAccessKey = remoteAccessKey[i];
    } else {
      CLOGF(
          ERR,
          "Invalid usage: remote access key exchanged in exchangeSendRecvHandles is not NVL. ");
      return commInvalidUsage;
    }
  }

  if (!tmpRegHdls.empty()) {
    CLOGF(ERR, "Invalid usage: buffers are not pre-regiestoered.");
    for (auto hdl : tmpRegHdls) {
      FB_COMMCHECK(mapper->deregDynamic(hdl));
    }
    return commInvalidUsage;
  }

  return commSuccess;
}

namespace {
inline commResult_t sendRecvCopyEngineImpl(
    const std::vector<OpElem*>& sendNvlOps) {
  if (sendNvlOps.empty()) {
    return commSuccess;
  }

  auto& firstOp = sendNvlOps.front();
  const auto comm = firstOp->comm_;

  for (int i = 0; i < sendNvlOps.size(); i++) {
    // cudaMemcpyAsync data from send buffer to recv buffer.
    auto& op = sendNvlOps[i];
    void* recvBuff = op->send.recvbuff->load();
    FB_COMMCHECK(comm->ctran_->mapper->icopy(
        recvBuff,
        op->send.sendbuff,
        op->send.count * commTypeSize(op->send.datatype),
        op->stream));
  }

  return commSuccess;
}
} // namespace

// Function submitted to GPE thread for both IB and NVL using SM
static commResult_t sendRecvImpl(
    const std::vector<std::unique_ptr<OpElem>>& opGroup) {
  std::vector<OpElem*> sendOpGroup, recvOpGroup, allOpGroup;

  auto& firstOp = opGroup.front();
  const auto opCount = firstOp->opCount;
  const auto comm = firstOp->comm_;

  for (auto& op : opGroup) {
    allOpGroup.push_back(op.get());
    if (op->type == OpElem::opType::SEND) {
      sendOpGroup.push_back(op.get());
    } else {
      recvOpGroup.push_back(op.get());
    }
  }

  const std::string algoName = sendRecvAlgoName(myAlgo, allOpGroup);
  CtranAlgoLogger logger(algoName, opCount, comm);

  auto& mapper = comm->ctran_->mapper;
  std::vector<void*> sendMemHdl(sendOpGroup.size());
  std::vector<void*> remoteRecvBuff(sendOpGroup.size());
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKey(
      sendOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperRequest>> sendCtrlReqs(
      sendOpGroup.size());
  std::unordered_map<int, std::unique_ptr<CtranMapperRequest>> putReqs;

  std::vector<void*> recvMemHdl(recvOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperRequest>> recvCtrlReqs(
      recvOpGroup.size());
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec(recvOpGroup.size());
  std::vector<int> recvPeerRanks(recvOpGroup.size());
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::make_unique<CtranMapperTimestamp>(algoName);

  std::vector<void*> tmpRegHdls;
  ctran::Profiler* profiler = comm->ctran_->profiler.get();
  if (profiler) {
    profiler->initForEachColl(
        opCount, NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT);
  }

  if (sendOpGroup.size() > 0 || recvOpGroup.size() > 0) {
    std::vector<size_t> sendSizes(sendOpGroup.size(), 0);
    uint64_t peerRank = 0;
    for (auto i = 0; i < sendOpGroup.size(); i++) {
      auto op = sendOpGroup[i];
      size_t sendSize = op->send.count * commTypeSize(op->send.datatype);
      sendSizes[i] = sendSize;
      peerRank = op->send.peerRank;
    }
    std::vector<size_t> recvSizes(recvOpGroup.size(), 0);
    for (auto i = 0; i < recvOpGroup.size(); i++) {
      auto op = recvOpGroup[i];
      size_t recvSize = op->recv.count * commTypeSize(op->recv.datatype);
      recvSizes[i] = recvSize;
      peerRank = op->recv.peerRank;
    }
    CtranMapperContext context(algoName, sendSizes, recvSizes);
    context.unpackPoolId = opGroup.front()->unpackPoolId;
    comm->ctran_->mapper->setContext(std::move(context));

    CTRAN_PROFILER_IF(profiler, {
      auto& algoContext = profiler->algoContext;
      algoContext.peerRank = peerRank;
      algoContext.algorithmName = algoName;
      algoContext.sendContext.messageSizes = folly::join(',', sendSizes);
      algoContext.recvContext.messageSizes = folly::join(',', recvSizes);
    });
  }

  // Issue control messages for send operations
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));
  for (auto i = 0; i < sendOpGroup.size(); ++i) {
    auto op = sendOpGroup[i];
    size_t sendBytes = op->send.count * commTypeSize(op->send.datatype);
    bool localReg = false;

    FB_COMMCHECK(mapper->searchRegHandle(
        op->send.sendbuff, sendBytes, &sendMemHdl[i], &localReg));
    if (localReg) {
      tmpRegHdls.push_back(sendMemHdl[i]);
    }
  }
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));

  CTRAN_PROFILER_CONDITION_IF(
      profiler,
      sendOpGroup.size() > 0,
      profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL));
  for (auto i = 0; i < sendOpGroup.size(); ++i) {
    auto op = sendOpGroup[i];
    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->irecvCtrl(
        &remoteRecvBuff[i], &remoteAccessKey[i], op->send.peerRank, &req));
    sendCtrlReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
  }

  // Issue control messages for recv operations
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));
  for (auto i = 0; i < recvOpGroup.size(); i++) {
    auto op = recvOpGroup[i];
    size_t recvBytes = op->recv.count * commTypeSize(op->recv.datatype);
    bool localReg = false;

    FB_COMMCHECK(mapper->searchRegHandle(
        op->recv.recvbuff, recvBytes, &recvMemHdl[i], &localReg));

    if (localReg) {
      tmpRegHdls.push_back(recvMemHdl[i]);
    }
  }
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));

  for (auto i = 0; i < recvOpGroup.size(); i++) {
    auto op = recvOpGroup[i];
    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->isendCtrl(
        op->recv.recvbuff, recvMemHdl[i], op->recv.peerRank, &req));
    recvCtrlReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
    recvPeerRanks[i] = op->recv.peerRank;

    // Initialize notify flag to receive from peer
    notifyVec[i] = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(mapper->initNotify(
        op->recv.peerRank, recvMemHdl[i], op->recv.kElem, notifyVec[i].get()));
  }

  // As we recv control msgs, issue PUT operations
  bool isIssuedFirst = false;
  while (putReqs.size() < sendOpGroup.size() && !comm->testAbort()) {
    for (auto i = 0; i < sendOpGroup.size(); i++) {
      // Already issued PUT
      if (putReqs.find(i) != putReqs.end()) {
        continue;
      }

      bool isComplete = false;
      FB_COMMCHECK(mapper->testRequest(sendCtrlReqs[i].get(), &isComplete));
      if (isComplete) {
        auto op = sendOpGroup[i];
        size_t sendSize = op->send.count * commTypeSize(op->send.datatype);

        timestamp->recvCtrl.push_back(
            CtranMapperTimestampPoint(op->send.peerRank));
        // iput internally dispatches to either network put or NVL copy
        CtranMapperRequest* req = nullptr;
        CTRAN_PROFILER_CONDITION_IF(profiler, !isIssuedFirst, {
          profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA);
          isIssuedFirst = true;
        });

        // ALGO_CTRL records duration from first irecvCtrl to the completion of
        // last irecvCtrl, i.e., when issuing the last put
        CTRAN_PROFILER_CONDITION_IF(
            profiler,
            putReqs.size() == sendOpGroup.size() - 1,
            profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL));
        FB_COMMCHECK(mapper->iput(
            op->send.sendbuff,
            remoteRecvBuff[i],
            sendSize,
            op->send.peerRank,
            CtranMapperConfig{
                .memHdl_ = sendMemHdl[i],
                .remoteAccessKey_ = remoteAccessKey[i],
                .notify_ = true,
                .kernElem_ = op->send.kElem},
            &req));
        putReqs[i] = std::unique_ptr<CtranMapperRequest>(req);
        timestamp->putIssued.push_back(
            CtranMapperTimestampPoint(op->send.peerRank));
      }
    }
  }

  // Wait for all PUT messages to complete
  for (auto i = 0; i < sendOpGroup.size(); i++) {
    auto op = sendOpGroup[i];
    FB_COMMCHECK(mapper->waitRequest(putReqs[i].get()));
    timestamp->putComplete.push_back(
        CtranMapperTimestampPoint(op->send.peerRank));
  }
  CTRAN_PROFILER_CONDITION_IF(
      profiler,
      sendOpGroup.size() > 0,
      profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  // Wait for all control messages and notifications to complete
  for (auto i = 0; i < recvOpGroup.size(); i++) {
    FB_COMMCHECK(mapper->waitRequest(recvCtrlReqs[i].get()));
    FB_COMMCHECK(mapper->waitNotify(notifyVec[i].get()));
  }

  // Deregister temporary registrations
  for (auto hdl : tmpRegHdls) {
    FB_COMMCHECK(mapper->deregDynamic(hdl));
  }

  CTRAN_PROFILER_IF(profiler, { profiler->reportToScuba(); });

  mapper->timestamps.push_back(std::move(timestamp));
  mapper->reportProfiling();

  return commSuccess;
}

bool ctranSendRecvSupport(int peer, CtranComm* comm) {
  const auto statex = comm->statex_.get();

  // Self peer is handled by CE directly, other peers require a valid ctran
  // backend
  if (ctranInitialized(comm) &&
      (peer == statex->rank() ||
       comm->ctran_->mapper->getBackend(peer) != CtranMapperBackend::UNSET)) {
    return true;
  } else {
    return false;
  }
}

commResult_t ctranSend(
    const void* sendbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      "CtranSend", sendbuff, nullptr, count, datatype, peer, comm, stream);

  auto op = new OpElem(OpElem::opType::SEND, stream, comm, opCount);
  op->send.sendbuff = sendbuff;
  op->send.count = count;
  op->send.datatype = datatype;
  op->send.peerRank = peer;
  if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE &&
      comm->ctran_->mapper->getBackend(peer) == CtranMapperBackend::NVL &&
      comm->statex_.get()->rank() != peer) {
    // used for storing recv address (updated by GPE thread), delete after
    // the address is no longer needed
    op->send.recvbuff = new std::atomic<void*>();
  }

  CtranOpGroup.push_back(op);

  return commSuccess;
}

commResult_t ctranRecv(
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      "CtranRecv", nullptr, recvbuff, count, datatype, peer, comm, stream);

  auto op = new OpElem(OpElem::opType::RECV, stream, comm, opCount);
  op->recv.recvbuff = recvbuff;
  op->recv.count = count;
  op->recv.datatype = datatype;
  op->recv.peerRank = peer;

  CtranOpGroup.push_back(op);

  return commSuccess;
}

static unsigned int bestThreadBlockSize = 0;

static inline int getNumGroups(size_t nbytes) {
  if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
    // if copy engine is enabled, we only need 1 group
    return 1;
  }
  // compute needed thread blocks for given bytes
  int nGroups = nbytes / NCCL_CTRAN_NVL_SENDRECV_CHUNK_SIZE;
  return std::min(
      std::max(1, nGroups), // at least 1 thread block
      // not exceed max theshold
      NCCL_CTRAN_NVL_SENDRECV_MAX_NUM_THREAD_BLOCKS);
}

static inline unsigned int getThreadBlockSize() {
  // If first time call, query cuda recommended blockSize
  if (bestThreadBlockSize == 0) {
    int minGridSize = 0;
    FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        (int*)&bestThreadBlockSize,
        reinterpret_cast<const void*>(ncclKernelSendRecv</*UNPACK=*/false>),
        0 /* dynamicSMemSize */,
        0 /* blockSizeLimit */));

    // TODO: bestThreadBlockSize may still be 0 after above function, need a
    // check here to avoid causing error in cudaLaunchKernel. Also for other
    // collectives calling getThreadBlockSize().
  }

  return NCCL_CTRAN_NVL_SENDRECV_THREAD_BLOCK_SIZE == -1
      ? bestThreadBlockSize
      : NCCL_CTRAN_NVL_SENDRECV_THREAD_BLOCK_SIZE;
}

static inline commResult_t setupPlan(
    CtranComm* comm,
    const std::vector<OpElem*>& opGroup,
    KernelConfig& config) {
  const auto statex = comm->statex_.get();
  config.args.devState_d = comm->ctran_->algo->getDevState();
  auto putNotifyList = CommonList<KernelElem>();
  auto waitNotifyList = CommonList<KernelElem>();
  int maxNumBlocks = 1;

  for (auto op : opGroup) {
    // For each non-zero NVL op, allocate a p2pElem to coordinate with kernel.
    // - For putNotify elem per send op, recvbuff will be assigned and the elem
    // will be posted to kernel once GPE thread imports remote memory.
    // - For waitNotify elem per recv op, the elem will be posted once GPE
    // thread confirmed the local memory registration.
    // - If an elem with a buffer not qualified for NVL backend, the elem will
    // be revoked by GPE thread, thus kernel will skip it.
    if (op->type == OpElem::opType::SEND &&
        comm->ctran_->mapper->getBackend(op->send.peerRank) ==
            CtranMapperBackend::NVL &&
        op->send.count > 0) {
      size_t nbytes = op->send.count * commTypeSize(op->send.datatype);
      int nGroups = getNumGroups(nbytes);
      // record the max number of thread blocks as final launching grid size
      maxNumBlocks = std::max(maxNumBlocks, nGroups);

      KernelElem* elem = nullptr;
      FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, nGroups, &elem));
      elem->putNotify.sendbuff = op->send.sendbuff;
      elem->putNotify.nbytes = nbytes;
      elem->putNotify.peerLocalRank = statex->localRank(op->send.peerRank);
      elem->putNotify.ngroups = nGroups;
      elem->putNotify.notify = true; // each put will be notified to remote peer
      op->send.kElem = elem;
      putNotifyList.enqueue(elem);
    } else if (
        op->type == OpElem::opType::RECV &&
        comm->ctran_->mapper->requiresRecvNotify(op->recv.peerRank) &&
        op->recv.count > 0) {
      KernelElem* elem = nullptr;
      // only 1 group handles waitNotify elem
      FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, 1, &elem));
      elem->waitNotify.peerLocalRank = statex->localRank(op->recv.peerRank);

      // pass the ngroups used by remote put
      size_t nbytes = op->recv.count * commTypeSize(op->recv.datatype);
      elem->waitNotify.recvbuff = op->recv.recvbuff;
      elem->waitNotify.nbytes = nbytes;
      elem->waitNotify.ngroups = getNumGroups(nbytes);

      op->recv.kElem = elem;
      if (comm->ctran_->mapper->requiresPostRecvNotify(op->recv.peerRank)) {
        waitNotifyList.enqueue(elem);
      }
    }
  }

  if (putNotifyList.count > 0) {
    // Allow user to increase SM usuage for putNotify involved kernel
    config.numBlocks = maxNumBlocks;
    config.numThreads = getThreadBlockSize();
  }

  if (config.type == KernelConfig::KernelType::SENDRECV_UNPACK ||
      config.type == KernelConfig::KernelType::RECV_UNPACK) {
    config.numBlocks = NCCL_CTRAN_UNPACK_NUM_THREAD_BLOCKS;
    config.numThreads = NCCL_CTRAN_UNPACK_THREAD_BLOCK_SIZE;
  }

  if (config.type == KernelConfig::KernelType::SENDRECV ||
      config.type == KernelConfig::KernelType::SENDRECV_NOTIFY ||
      config.type == KernelConfig::KernelType::SENDRECV_UNPACK) {
    config.args.collective.sendrecv.putNotifyList = putNotifyList.head;
    config.args.collective.sendrecv.waitNotifyList = waitNotifyList.head;
    FB_COMMCHECK(comm->ctran_->mapper->prepareUnpackConsumer(
        &config.args.collective.sendrecv.unpack,
        NCCL_CTRAN_UNPACK_NUM_THREAD_BLOCKS,
        opGroup,
        config));
  } else if (
      config.type == KernelConfig::KernelType::SEND ||
      config.type == KernelConfig::KernelType::SEND_NOTIFY) {
    config.args.collective.send.putNotifyList = putNotifyList.head;
    config.args.collective.send.sendbuff = nullptr;
    if (opGroup.size() == 1) {
      const auto op = opGroup[0];
      config.args.collective.send.sendbuff = op->send.sendbuff;
      config.args.collective.send.count = op->send.count;
      config.args.collective.send.datatype = op->send.datatype;
    }
  } else if (
      config.type == KernelConfig::KernelType::RECV ||
      config.type == KernelConfig::KernelType::RECV_NOTIFY ||
      config.type == KernelConfig::KernelType::RECV_UNPACK) {
    config.args.collective.recv.waitNotifyList = waitNotifyList.head;
    config.args.collective.recv.recvbuff = nullptr;
    if (opGroup.size() == 1) {
      const auto op = opGroup[0];
      config.args.collective.recv.recvbuff = op->recv.recvbuff;
      config.args.collective.recv.count = op->recv.count;
      config.args.collective.recv.datatype = op->recv.datatype;
    }
    FB_COMMCHECK(comm->ctran_->mapper->prepareUnpackConsumer(
        &config.args.collective.recv.unpack,
        NCCL_CTRAN_UNPACK_NUM_THREAD_BLOCKS,
        opGroup,
        config));
  }

  return commSuccess;
}

static inline commResult_t sendRecvSelfImpl(
    std::vector<OpElem*>& selfSends,
    std::vector<OpElem*>& selfRecvs,
    CtranComm* comm) {
  const auto statex = comm->statex_.get();
  if (selfSends.size() != selfRecvs.size()) {
    CLOGF(
        ERR,
        "Invalid usage: number of self ncclSend ({}) and ncclRecv ({}) does not match on rank {}",
        selfSends.size(),
        selfRecvs.size(),
        statex->rank());
    return commInvalidUsage;
  }

  for (int i = 0; i < selfSends.size(); i++) {
    // cudaMemcpyAsync data from local send buffer to recv buffer.
    // No need track completion in CTran, as it will finish when user
    // synchronizes the stream
    if (selfSends[i]->send.sendbuff != selfRecvs[i]->recv.recvbuff) {
      FB_COMMCHECK(comm->ctran_->mapper->icopy(
          selfRecvs[i]->recv.recvbuff,
          selfSends[i]->send.sendbuff,
          selfSends[i]->send.count * commTypeSize(selfSends[i]->send.datatype),
          selfSends[i]->stream));
    }
  }

  return commSuccess;
}

commResult_t submitHandleExchangeToGpe(const std::vector<OpElem*>& ops) {
  if (ops.empty()) {
    return commSuccess;
  }
  std::vector<std::unique_ptr<struct OpElem>> exchangeOpGroup;
  CtranComm* comm = ops.front()->comm_;
  cudaStream_t stream = ops.front()->stream;
  auto opCount = ops.front()->opCount;
  auto mapper = comm->ctran_->mapper.get();
  for (const auto& op : ops) {
    auto exchangeOp =
        std::make_unique<OpElem>(op->type, stream, comm, op->opCount);
    // only exchange handles for NVL backend
    if (op->type == OpElem::opType::SEND &&
        mapper->getBackend(op->send.peerRank) == CtranMapperBackend::NVL) {
      exchangeOp->send.sendbuff = op->send.sendbuff;
      exchangeOp->send.count = op->send.count;
      exchangeOp->send.datatype = op->send.datatype;
      exchangeOp->send.peerRank = op->send.peerRank;
      exchangeOp->send.recvbuff = op->send.recvbuff;
      exchangeOpGroup.push_back(std::move(exchangeOp));
    } else if (
        op->type == OpElem::opType::RECV &&
        mapper->getBackend(op->recv.peerRank) == CtranMapperBackend::NVL) {
      exchangeOp->recv.recvbuff = op->recv.recvbuff;
      exchangeOp->recv.count = op->recv.count;
      exchangeOp->recv.datatype = op->recv.datatype;
      exchangeOp->recv.peerRank = op->recv.peerRank;
      exchangeOpGroup.push_back(std::move(exchangeOp));
    }
  }

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::SENDRECV,
      stream,
      "CtranSendHanldeExchange",
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = comm->ctran_->algo->getDevState();

  FB_COMMCHECK(comm->ctran_->gpe->submitHost(
      std::move(exchangeOpGroup),
      exchangeSendRecvHandles,
      config,
      nullptr /* exReq */));

  return commSuccess;
}

commResult_t ctranGroupEndHook(
    std::optional<std::chrono::milliseconds> timeout) {
  while (!CtranOpGroup.empty()) {
    std::vector<OpElem*> allOps;
    std::vector<OpElem*> selfSends, selfRecvs, sendNvlOps, nvlOps, ibOps;
    std::deque<OpElem*> pending;
    bool hasSend = false;
    bool hasRecv = false;
    bool hasTcpDmRecv = false;

    // Submit ops with the same comm and stream in a single batch
    CtranComm* comm = CtranOpGroup.front()->comm_;
    cudaStream_t stream = CtranOpGroup.front()->stream;
    const auto statex = comm->statex_.get();
    auto mapper = comm->ctran_->mapper.get();

    while (!CtranOpGroup.empty()) {
      auto op = dequeFront(CtranOpGroup);

      if (op->comm_ == comm && op->stream == stream) {
        if (op->type == OpElem::opType::SEND) {
          hasSend = true;
          if (op->send.peerRank == statex->rank()) {
            selfSends.push_back(op);
            continue;
          }

          // Async buffer registration for send and recv buffers to hide
          // registration cost.
          // - If the buffer has already been registered, regAsync will return
          //   immediately.
          // - If the buffer is not yet registered at regAsync internal query, a
          //   request will be enqueued to asyncReg thread.
          // - A first-used buffer will be registered either by asyncReg thread
          //   or GPE thread (see CtranMapperRegCache::regRange).
          // - regAsync is a no-op if NCCL_CTRAN_REGISTER is not async mode.
          //
          // Expected performance improvement for communication involving
          // first-time registration:
          // - [Improved case] If the buffer is registered by asyncReg thread
          //   ahead of time, it hides registration cost.
          // - If the buffer is registered by GPE thread, e.g., due to too busy
          //   asyncReg thread or not-advanced CPU schedule, the registration
          //   cost has to be exposed similar to lazy registatration mode.
          size_t nbytes = op->send.count * commTypeSize(op->send.datatype);
          FB_COMMCHECK(mapper->regAsync(op->send.sendbuff, nbytes));
          if (comm->ctran_->mapper->getBackend(op->send.peerRank) ==
              CtranMapperBackend::NVL) {
            sendNvlOps.push_back(op);
            nvlOps.push_back(op);
          } else {
            ibOps.push_back(op);
          }
        } else if (op->type == OpElem::opType::RECV) {
          hasRecv = true;
          if (op->recv.peerRank == statex->rank()) {
            selfRecvs.push_back(op);
            continue;
          }

          // For TCP Device Memory, if we have peers we are going to receive
          // from, we need to unpack the data from the bounce buffer.
          if (comm->ctran_->mapper->getBackend(op->recv.peerRank) ==
              CtranMapperBackend::TCPDM) {
            hasTcpDmRecv = true;
          }

          size_t nbytes = op->recv.count * commTypeSize(op->recv.datatype);
          FB_COMMCHECK(mapper->regAsync(op->recv.recvbuff, nbytes));
          if (comm->ctran_->mapper->getBackend(op->recv.peerRank) ==
              CtranMapperBackend::NVL) {
            nvlOps.push_back(op);
          } else {
            ibOps.push_back(op);
          }
        }

        allOps.push_back(op);
      } else {
        // If not belong to this batch, put to pending and handle in next batch
        pending.push_back(op);
      }
    }

    // Handle self sends and recvs via CE-based icopy
    FB_COMMCHECK(sendRecvSelfImpl(selfSends, selfRecvs, comm));

    // For non-self sends and recvs: decide the kernel function and submit to
    // GPE
    if (!allOps.empty()) {
      KernelConfig::KernelType kernelType = KernelConfig::KernelType::SENDRECV;
      if (hasSend && hasRecv) {
        if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
          kernelType = KernelConfig::KernelType::SENDRECV_NOTIFY;
        } else if (hasTcpDmRecv) {
          kernelType = KernelConfig::KernelType::SENDRECV_UNPACK;
        }
      } else {
        if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
          kernelType = hasSend ? KernelConfig::KernelType::SEND_NOTIFY
                               : KernelConfig::KernelType::RECV_NOTIFY;
        } else if (hasTcpDmRecv) {
          kernelType = hasSend ? KernelConfig::KernelType::SEND
                               : KernelConfig::KernelType::RECV_UNPACK;
        } else {
          kernelType = hasSend ? KernelConfig::KernelType::SEND
                               : KernelConfig::KernelType::RECV;
        }
      }

      std::vector<std::unique_ptr<OpElem>> toSubmitUniquePtr;

      if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
        // first, send/recv NVL ops with copy engine
        if (!nvlOps.empty()) {
          submitHandleExchangeToGpe(nvlOps);
          for (auto& op : sendNvlOps) {
            while (!op->send.recvbuff->load()) {
              FB_COMMCHECK(comm->getAsyncResult());
            }
          }
          sendRecvCopyEngineImpl(sendNvlOps);
          for (auto& op : sendNvlOps) {
            if (op->send.recvbuff) {
              delete op->send.recvbuff;
            }
          }

          nvlOps.clear();
          sendNvlOps.clear();
        }

        // next, deal with IB ops
        if (!ibOps.empty()) {
          toSubmitUniquePtr.reserve(ibOps.size());
          for (auto x : ibOps) {
            toSubmitUniquePtr.push_back(std::unique_ptr<OpElem>(x));
          }
          ibOps.clear();
        }
      } else {
        // if copy engine is not enabled, submit all ops to GPE
        toSubmitUniquePtr.reserve(allOps.size());
        for (auto x : allOps) {
          toSubmitUniquePtr.push_back(std::unique_ptr<OpElem>(x));
        }
      }
      void* gpeFn = kernelFns.at(kernelType);
      auto config = KernelConfig(
          kernelType,
          stream,
          sendRecvAlgoName(myAlgo, allOps),
          allOps.front()->opCount);
      FB_COMMCHECK(setupPlan(comm, allOps, config));
      FB_COMMCHECK(comm->ctran_->gpe->submit(
          std::move(toSubmitUniquePtr), sendRecvImpl, config, gpeFn, timeout));
    }

    // No kernel would be submitted if only self sendrecv is called, update op
    // count here
    if (allOps.empty() && !selfSends.empty()) {
      comm->ctran_->updateOpCount();
    }
    allOps.clear();

    comm->ctran_->numGroupedDefaultOps = 0;

    // handle next batch
    CtranOpGroup = std::move(pending);
  }

  return commSuccess;
}

void ctranGroupTrackDefaultOp(CtranComm* comm) {
  if (ctranInitialized(comm)) {
    comm->ctran_->numGroupedDefaultOps++;
  }
}
