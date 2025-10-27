// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cstddef>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicCommon.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/logger/LogUtils.h"

#define dceil(x, y) ((x / y) + !!(x % y))

#define PUT_AND_WAIT(perfconfig)                                              \
  do {                                                                        \
    if (algoType == OpElem::opType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG) {     \
      FB_COMMCHECK(                                                           \
          peerPutNonContig<perfconfig>(                                       \
              comm,                                                           \
              sendbuffs,                                                      \
              remoteRecvBuffs,                                                \
              sendCountsTmpbufCPU,                                            \
              sendcountsLength,                                               \
              datatype,                                                       \
              tmpRegHdls,                                                     \
              nRanks,                                                         \
              myRank,                                                         \
              timestamp,                                                      \
              remoteAccessKeys,                                               \
              ibPutReqs,                                                      \
              ibRecvCtrlReqs,                                                 \
              maxRecvcount,                                                   \
              maxSendcount));                                                 \
    } else {                                                                  \
      FB_COMMCHECK(peerPutContig(                                             \
          comm,                                                               \
          sendbuffs,                                                          \
          remoteRecvBuffs,                                                    \
          sendCountsTmpbufCPU,                                                \
          datatype,                                                           \
          tmpRegHdls,                                                         \
          nRanks,                                                             \
          myRank,                                                             \
          timestamp,                                                          \
          remoteAccessKeys,                                                   \
          ibPutReqs,                                                          \
          ibRecvCtrlReqs));                                                   \
    }                                                                         \
    /* Wait for all puts to complete */                                       \
    for (auto& req : ibPutReqs) {                                             \
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest<perfconfig>(req.get())); \
    }                                                                         \
    /* Wait for all receives (i.e., remote IB puts) to complete */            \
    for (auto& notify : notifyVec) {                                          \
      FB_COMMCHECK(                                                           \
          comm->ctran_->mapper->waitNotify<perfconfig>(notify.get()));        \
    }                                                                         \
    if (statex->nNodes() > 1) {                                               \
      elem->post(1);                                                          \
      elem->wait(1);                                                          \
    }                                                                         \
    /* Always wait for all sendCtrlReqs to complete so that the memory can be \
       safely reused in next collective; otherwise, ibvc may complete the     \
       previous request while the memory has already been assigned to a new   \
       request. */                                                            \
    for (const auto& sendCtrlrReq : ibSendCtrlReqs) {                         \
      FB_COMMCHECK(                                                           \
          comm->ctran_->mapper->waitRequest<perfconfig>(sendCtrlrReq.get())); \
    }                                                                         \
  } while (0)

static inline commResult_t regIsendCtrl(
    CtranComm*& comm,
    int peer,
    void* recvbuff,
    size_t recvBytes,
    std::vector<void*>& recvMemHdl,
    std::vector<void*>& tmpRegHdls,
    std::vector<std::unique_ptr<CtranMapperRequest>>& reqs) {
  bool localReg = false;
  CtranMapperRequest* req = nullptr;
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      recvbuff, recvBytes, &recvMemHdl[peer], &localReg));
  if (localReg) {
    tmpRegHdls.push_back(recvMemHdl[peer]);
  }
  FB_COMMCHECK(
      comm->ctran_->mapper->isendCtrl(recvbuff, recvMemHdl[peer], peer, &req));
  reqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
  return commSuccess;
}

static inline commResult_t regIrecvCtrl(
    CtranComm*& comm,
    int peer,
    std::vector<void*>& remoteRecvBuffs,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
    std::vector<std::unique_ptr<CtranMapperRequest>>& reqs) {
  CtranMapperRequest* req = nullptr;

  FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
      &remoteRecvBuffs[peer], &remoteAccessKeys[peer], peer, &req));
  reqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
  return commSuccess;
}

static inline commResult_t peerPutContig(
    CtranComm* comm,
    const void* const* sendbuffs,
    std::vector<void*>& remoteRecvBuffs,
    size_t* sendCountsTmpbufCPU,
    commDataType_t datatype,
    std::vector<void*>& tmpRegHdls,
    int nRanks,
    int myRank,
    std::unique_ptr<CtranMapperTimestamp> const& timestamp,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
    std::vector<std::unique_ptr<CtranMapperRequest>>& ibPutReqs,
    std::vector<std::unique_ptr<CtranMapperRequest>>& ibRecvCtrlReqs) {
  std::vector<void*> sendMemHdls(nRanks);
  // Search handlers for all sendbuffs.
  for (int i = 0; i < nRanks; i++) {
    int peer = (myRank + i) % nRanks;
    if (comm->statex_->isSameNode(myRank, peer)) {
      continue;
    }
    size_t sendCount = sendCountsTmpbufCPU[peer];
    if (sendCount * commTypeSize(datatype) < CTRAN_MIN_REGISTRATION_SIZE) {
      sendCount = dceil(CTRAN_MIN_REGISTRATION_SIZE, commTypeSize(datatype));
    }

    bool localReg = false;
    FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
        sendbuffs[peer],
        sendCount * commTypeSize(datatype),
        &sendMemHdls[peer],
        &localReg));
    if (localReg) {
      tmpRegHdls.push_back(sendMemHdls[peer]);
    }
  }
  // issue network puts:
  // - Sender puts data for peers, whenever received the remote recvbuff
  // handle
  // - Exit until all peers' put have been issued (putPeers becomes empty)
  while (!ibRecvCtrlReqs.empty()) {
    auto it = ibRecvCtrlReqs.begin();
    while (it != ibRecvCtrlReqs.end()) {
      auto& recvCtrlReq = *it;
      int peer = recvCtrlReq->peer;

      bool completed = false;
      FB_COMMCHECK(
          comm->ctran_->mapper->testRequest(recvCtrlReq.get(), &completed));
      if (!completed) {
        it++;
        continue;
      }
      std::vector<CtranMapperPutMsg> putMsgs;
      timestamp->recvCtrl.emplace_back(peer);

      size_t offsetCount = comm->ctran_->algo->getTmpBufOffset(
                               CtranAlgo::TmpbufType::RECVCOUNTS_TMPBUF) /
          sizeof(size_t);

      auto [interNodeRemoteTmpbuff, interNodeRemoteTmpAccessKey] =
          comm->ctran_->algo->getInterNodeTmpBufInfo(peer);
      size_t* remoteTmpRecvCountsBufGPU =
          (size_t*)interNodeRemoteTmpbuff + offsetCount;

      auto [sendCountsTmpbufGPU, tmpbufRegHdl] =
          comm->ctran_->algo->getTmpBufInfo(
              CtranAlgo::TmpbufType::SENDCOUNTS_TMPBUF);
      putMsgs.emplace_back(
          CtranMapperPutMsg{
              .sbuf = &reinterpret_cast<size_t*>(sendCountsTmpbufGPU)[peer],
              .dbuf = &remoteTmpRecvCountsBufGPU[myRank],
              .len = sizeof(size_t),
              .config =
                  CtranMapperConfig{
                      .memHdl_ = tmpbufRegHdl,
                      .remoteAccessKey_ = interNodeRemoteTmpAccessKey,
                      .notify_ = false /*notify*/},
              .req = nullptr});

      ibPutReqs.push_back(std::make_unique<CtranMapperRequest>());

      // Only notify the peer at the last message. If we notify every iput,
      // the peer may exist without receiving all the data.
      putMsgs.emplace_back(
          CtranMapperPutMsg{
              .sbuf = sendbuffs[peer],
              .dbuf = remoteRecvBuffs[peer],
              .len = sendCountsTmpbufCPU[peer] * commTypeSize(datatype),
              .config =
                  CtranMapperConfig{
                      .memHdl_ = sendMemHdls[peer],
                      .remoteAccessKey_ = remoteAccessKeys[peer],
                      .notify_ = true /*notify*/},
              .req = ibPutReqs.back().get()});
      FB_COMMCHECK(comm->ctran_->mapper->iputBatch(std::move(putMsgs), peer));
      timestamp->putIssued.emplace_back(peer);
      it = ibRecvCtrlReqs.erase(it);
    }
  }
  return commSuccess;
}

commResult_t ctranAllToAllvDynamicIbImpl(
    const void* const* sendbuffs,
    void* const* recvbuffs,
    size_t sendcountsLength,
    size_t maxSendcount,
    size_t maxRecvcount,
    commDataType_t datatype,
    OpElem::opType algoType,
    CtranComm* comm,
    std::unique_ptr<CtranMapperTimestamp> timestamp,
    KernelElem* elem) {
  const auto& statex = comm->statex_;
  const int myRank = statex->rank();
  const int nRanks = statex->nRanks();

  std::vector<void*> remoteRecvBuffs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nRanks);

  std::vector<std::unique_ptr<CtranMapperRequest>> ibSendCtrlReqs,
      ibRecvCtrlReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> ibPutReqs;
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec;

  std::vector<void*> recvMemHdl(nRanks);
  std::vector<void*> tmpRegHdls;

  size_t* sendCountsTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDCOUNTS_TMPBUF_CPU));
  std::unordered_set<int> ibPeers;
  for (int i = 0; i < nRanks; i++) {
    int peer = (myRank + i) % nRanks;
    if (statex->isSameNode(myRank, peer)) {
      continue;
    }
    ibPeers.insert(peer);
  }
  // pre-connect all peers
  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    comm->ctran_->mapper->preConnect(ibPeers);
  }

  // Prepare buffers shifted with displacement, and set ctrl/put/notify
  // schedules. Try to schedule ctrl message and put sequence as rank i start
  // sending to rank i+1 to avoid congestion in potential all-to-one case.
  // Specified in putPeers, sendCtrlPeers.
  // To optimize: can change i = 1; we don't need intra-node?
  for (int i = 0; i < nRanks; i++) {
    int peer = (myRank + i) % nRanks;
    if (statex->isSameNode(myRank, peer)) {
      continue;
    }
    // schedule IB ctrl messages
    FB_COMMCHECK(regIsendCtrl(
        comm,
        peer,
        recvbuffs[peer],
        maxRecvcount * commTypeSize(datatype),
        recvMemHdl,
        tmpRegHdls,
        ibSendCtrlReqs));

    // Initialize notify flag to receive from peer
    auto notify = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(
        comm->ctran_->mapper->initNotify(peer, recvMemHdl[peer], notify.get()));
    notifyVec.push_back(std::move(notify));

    FB_COMMCHECK(regIrecvCtrl(
        comm, peer, remoteRecvBuffs, remoteAccessKeys, ibRecvCtrlReqs));
  }

  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    PUT_AND_WAIT(LowLatencyCollConfig);
  } else {
    PUT_AND_WAIT(DefaultPerfCollConfig);
  }

  comm->ctran_->mapper->timestamps.emplace_back(std::move(timestamp));
  comm->ctran_->mapper->reportProfiling();

  /* deregister temporary registrations */
  // FIXME: let GPE kernel to finish then deregister to avoid race condition
  // on cuda context
  for (auto& hdl : tmpRegHdls) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(hdl));
  }

  return commSuccess;
}

commResult_t setupKernelConfig(
    const size_t* sendcounts,
    size_t sendcountsLength,
    void* const* recvbuffs,
    size_t* actualRecvcounts,
    commDataType_t datatype,
    CtranComm* comm,
    KernelConfig& config,
    KernelElem** elem) {
  // Unlike alltoall, we cannot automatically detect grid size because each
  // rank may see different counts; use static gridSize for now.
  config.numThreads = NCCL_CTRAN_ALLTOALLV_DYNAMIC_THREAD_BLOCK_SIZE;
  config.numBlocks = NCCL_CTRAN_ALLTOALLV_DYNAMIC_NUM_THREAD_BLOCKS;

  // Adjust gridSize to fit alltoallv kernel algorithm:
  // 1. gridSize must be even number, because we split blocks into two sets of
  //   groups, one for sends and the other for receives, each send and receive
  //   pair must use the same number of blocks
  if (config.numBlocks % 2) {
    config.numBlocks += 1;
  }
  // 2. gridSize must be <= CTRAN_ALGO_MAX_THREAD_BLOCKS, since internal
  //   states/flags holds at most CTRAN_ALGO_MAX_THREAD_BLOCKS blocks
  if (config.numBlocks < 2 || config.numBlocks > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
    config.numBlocks = CTRAN_ALGO_MAX_THREAD_BLOCKS;
  }

  FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, elem));

  config.args.devState_d = comm->ctran_->algo->getDevState();

  config.args.collective.alltoallv_dynamic.datatype = datatype;

  config.args.collective.alltoallv_dynamic.sendbuffsPtrTmpbufCPU =
      reinterpret_cast<void**>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDBUFFS_PTR_TMPBUF_CPU));

  config.args.collective.alltoallv_dynamic.sendcounts = sendcounts;
  config.args.collective.alltoallv_dynamic.sendCountsTmpbufGPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDCOUNTS_TMPBUF));
  config.args.collective.alltoallv_dynamic.sendCountsTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDCOUNTS_TMPBUF_CPU));
  config.args.collective.alltoallv_dynamic.sendcountsLength = sendcountsLength;

  for (int i = 0; i < comm->statex_->nRanks(); i++) {
    config.args.collective.alltoallv_dynamic.recvbuffsPtrGPU[i] = recvbuffs[i];
  }

  config.args.collective.alltoallv_dynamic.actualRecvcounts = actualRecvcounts;
  config.args.collective.alltoallv_dynamic.recvCountsTmpbufGPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::RECVCOUNTS_TMPBUF));

  config.args.collective.alltoallv_dynamic.kElem = *elem;

  return commSuccess;
}

commResult_t opIbImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  if (opGroup.empty()) {
    CLOGF(ERR, "Empty opGroup passed in AllToAllvDynamic");
    return commInternalError;
  }

  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = opGroup.front()->comm_;

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("ncclx::alltoallvDynamic"));

  return ctranAllToAllvDynamicIbImpl(
      op->alltoallv_dynamic.sendbuffs,
      op->alltoallv_dynamic.recvbuffs,
      op->alltoallv_dynamic.sendcountsLength,
      op->alltoallv_dynamic.maxSendcount,
      op->alltoallv_dynamic.maxRecvcount,
      op->alltoallv_dynamic.datatype,
      op->type,
      comm,
      std::move(timestamp),
      op->alltoallv_dynamic.kElem);
}

commResult_t setupGpeOp(
    const void* const* sendbuffs,
    void* const* recvbuffs,
    size_t sendcountsLength,
    size_t maxSendcount,
    size_t maxRecvcount,
    commDataType_t datatype,
    OpElem::opType opType,
    CtranComm* comm,
    uint64_t opCount,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    KernelElem* elem) {
  std::unique_ptr<struct OpElem> op =
      std::unique_ptr<struct OpElem>(new OpElem(opType, comm, opCount));
  op->alltoallv_dynamic.sendbuffs = sendbuffs;
  op->alltoallv_dynamic.recvbuffs = recvbuffs;
  op->alltoallv_dynamic.datatype = datatype;
  op->alltoallv_dynamic.sendcountsLength = sendcountsLength;
  op->alltoallv_dynamic.maxSendcount = maxSendcount;
  op->alltoallv_dynamic.maxRecvcount = maxRecvcount;
  op->alltoallv_dynamic.kElem = elem;

  opGroup.push_back(std::move(op));

  return commSuccess;
}

commResult_t ctranAllToAllvDynamicSupport(
    CtranComm* comm,
    const meta::comms::Hints& hints,
    size_t maxSendcount,
    size_t maxRecvcount,
    commDataType_t datatype) {
  const auto statex = comm->statex_.get();

  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(commInvalidUsage, "CTRAN is not initialized on local rank");
  } else {
    // Check if all remote peers are supported by ctran
    // For intra-node peers, ctranAlgo supports copy based path;
    // for inter-node peers, we need a mapper backend to support.
    const int myNode = statex->node();
    for (int rank = 0; rank < statex->nRanks(); rank++) {
      if (statex->node(rank) != myNode &&
          comm->ctran_->mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
        FB_ERRORRETURN(
            commInvalidUsage,
            "CTRAN is not initialized on remote rank {}",
            rank);
      }
    }
  }

  // FIXME:
  // a proper way to do this check is first look up the registered buffer,
  // - if buffer is registered and registered buffer size <
  // CTRAN_MIN_REGISTRATION_SIZE, fail (regardless of maxRecvcounts here)
  // - if buffer is not registered then check with maxRecvcounts

  if (maxSendcount * commTypeSize(datatype) < CTRAN_MIN_REGISTRATION_SIZE) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "maxSendcount {} is too small for CTRAN, expect minimal length {}",
        maxSendcount,
        CTRAN_MIN_REGISTRATION_SIZE / commTypeSize(datatype));
  }
  if (maxRecvcount * commTypeSize(datatype) < CTRAN_MIN_REGISTRATION_SIZE) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "maxRecvcount {} is too small for CTRAN, expect minimal length {}",
        maxRecvcount,
        CTRAN_MIN_REGISTRATION_SIZE / commTypeSize(datatype));
  }

  commResult_t res;
  std::string locationRes;

  res = hints.get("ncclx_alltoallv_dynamic_sendbuffs_location", locationRes);
  if (res == commSuccess && locationRes != "cpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_sendbuffs_location, supported values: cpu");
  }

  res = hints.get("ncclx_alltoallv_dynamic_recvbuffs_location", locationRes);
  if (res == commSuccess && locationRes != "cpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_recvbuffs_location, supported values: cpu");
  }

  res = hints.get("ncclx_alltoallv_dynamic_sendcounts_location", locationRes);
  if (res == commSuccess && locationRes != "gpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_sendcounts_location, supported values: gpu");
  }

  res =
      hints.get("ncclx_alltoallv_dynamic_max_sendcounts_location", locationRes);
  if (res == commSuccess && locationRes != "cpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_max_sendcounts_location, supported values: cpu");
  }

  res =
      hints.get("ncclx_alltoallv_dynamic_max_recvcounts_location", locationRes);
  if (res == commSuccess && locationRes != "cpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_max_recvcounts_location, supported values: cpu");
  }

  res = hints.get(
      "ncclx_alltoallv_dynamic_actual_recvcounts_location", locationRes);
  if (res == commSuccess && locationRes != "gpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_actual_recvcounts_location, supported values: gpu");
  }

  return commSuccess;
}
