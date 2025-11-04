// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_ALL_TO_ALLV_IMPL_H_
#define CTRAN_ALL_TO_ALLV_IMPL_H_

#include <folly/synchronization/CallOnce.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/utils/cvars/nccl_cvars.h"

static inline const std::string allToAllAlgoName(enum NCCL_ALLTOALL_ALGO algo) {
  switch (algo) {
    case NCCL_ALLTOALL_ALGO::ctran:
      return "CtranAllToAll";
    case NCCL_ALLTOALL_ALGO::orig:
      return "Baseline";
    default:
      return "Unknown";
  }
}

static inline const std::string allToAllvAlgoName(
    enum NCCL_ALLTOALLV_ALGO algo) {
  switch (algo) {
    case NCCL_ALLTOALLV_ALGO::ctran:
      return "CtranAllToAllv";
    case NCCL_ALLTOALLV_ALGO::compCtran:
      return "CtranCompressedAllToAllv";
    case NCCL_ALLTOALLV_ALGO::bsCompCtran:
      return "CtranBootstrapCompressedAllToAllv";
    case NCCL_ALLTOALLV_ALGO::orig:
      return "Baseline";
    default:
      return "Unknown";
  }
}

static inline commResult_t searchRegHandle(
    CtranComm*& comm,
    const void* buff,
    size_t bytes,
    void*& hdl,
    std::vector<void*>& tmpRegHdls) {
  bool localReg = false;
  FB_COMMCHECK(
      comm->ctran_->mapper->searchRegHandle(buff, bytes, &hdl, &localReg));
  if (localReg) {
    tmpRegHdls.push_back(hdl);
  }
  return commSuccess;
}

template <typename PerfConfig = DefaultPerfCollConfig>
commResult_t ctranAllToAllvIbImpl(
    const void* sendbuff,
    std::vector<size_t>& sendCounts,
    std::vector<size_t>& sDispls,
    void* recvbuff,
    std::vector<size_t>& recvCounts,
    std::vector<size_t>& rDispls,
    commDataType_t datatype,
    uint64_t opCount,
    CtranComm* comm,
    std::unique_ptr<CtranMapperTimestamp> timestamp) {
  const auto& statex = comm->statex_;
  const int myRank = statex->rank();
  const int nRanks = statex->nRanks();

  static const auto myAlgo = NCCL_ALLTOALLV_ALGO::ctran;
  const std::string algoName = allToAllvAlgoName(myAlgo);
  const bool useProfiler = NCCL_CTRAN_PROFILING != NCCL_CTRAN_PROFILING::none;

  std::vector<const void*> sendBuffs(nRanks);
  std::vector<void*> recvBuffs(nRanks);
  std::vector<void*> remoteRecvBuffs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nRanks);

  std::vector<void*> sendMemHdl(nRanks);
  std::vector<void*> recvMemHdl(nRanks);
  std::vector<void*> tmpRegHdls;

  std::vector<int> ibRecvPeers, ibSendPeers;
  std::unordered_set<int> ibPeers;

  ctran::Profiler* profiler = comm->ctran_->profiler.get();
  if (profiler) {
    profiler->initForEachColl(
        opCount, NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT);
  }

  if (sendCounts.size() > 0) {
    std::vector<size_t> sendSizes(nRanks, 0);
    for (int i = 0; i < nRanks; i++) {
      sendSizes[i] = sendCounts[i] * commTypeSize(datatype);
    }
    std::vector<size_t> recvSizes(nRanks, 0);
    for (int i = 0; i < nRanks; i++) {
      recvSizes[i] = recvCounts[i] * commTypeSize(datatype);
    }
    CtranMapperContext context(algoName, sendSizes, recvSizes);
    comm->ctran_->mapper->setContext(std::move(context));

    CTRAN_PROFILER_IF(profiler, {
      auto& algoContext = profiler->algoContext;
      algoContext.algorithmName = algoName;
      algoContext.sendContext.messageSizes = folly::join(',', sendSizes);
      algoContext.recvContext.messageSizes = folly::join(',', recvSizes);
    });
  }

  // Prepare buffers shifted with displacement, and set ctrl/put/notify
  // schedules. Try to schedule ctrl message and put sequence as rank i start
  // sending to rank i+1 to avoid congestion in potential all-to-one case.
  // Specified in putPeers, sendCtrlPeers.
  size_t contigSendBufSize = 0;
  size_t contigRecvBufSize = 0;
  for (int i = 0; i < nRanks; i++) {
    int peer = (myRank + i) % nRanks;
    if (sendCounts[peer]) {
      sendBuffs[peer] = static_cast<const char*>(sendbuff) +
          sDispls[peer] * commTypeSize(datatype);
      ibSendPeers.push_back(peer);
      ibPeers.insert(peer);
      contigSendBufSize =
          std::max(contigSendBufSize, sDispls[peer] + sendCounts[peer]);
    }
    if (recvCounts[peer]) {
      recvBuffs[peer] =
          static_cast<char*>(recvbuff) + rDispls[peer] * commTypeSize(datatype);
      ibRecvPeers.push_back(peer);
      ibPeers.insert(peer);
      contigRecvBufSize =
          std::max(contigRecvBufSize, rDispls[peer] + recvCounts[peer]);
    }
  }

  // pre-connect all peers
  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    comm->ctran_->mapper->preConnect(ibPeers);
  }

  // schedule IB ctrl messages
  std::vector<CtranMapperRequest> ibSendCtrlReqs(ibRecvPeers.size()),
      ibRecvCtrlReqs(ibSendPeers.size());
  std::vector<CtranMapperNotify> notifyVec(ibRecvPeers.size());

  void* tmpHdl = nullptr;
  // Search for the handle only when there are RecvPeers to avoid attempting to
  // search/register with a buffer size of 0.
  if (!ibRecvPeers.empty()) {
    CTRAN_PROFILER_IF(
        profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));

    FB_COMMCHECK(searchRegHandle(
        comm,
        recvbuff,
        contigRecvBufSize * commTypeSize(datatype),
        tmpHdl,
        tmpRegHdls));

    CTRAN_PROFILER_IF(
        profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));
  }

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL));

  FB_COMMCHECK(comm->ctran_->mapper->isendCtrlBatch<PerfConfig>(
      recvBuffs, tmpHdl, ibRecvPeers, ibSendCtrlReqs, CtranMapperBackend::IB));
  FB_COMMCHECK(comm->ctran_->mapper->initNotifyBatchIB(ibRecvPeers, notifyVec));

  tmpHdl = nullptr;
  // Search for the handle only when there are SendPeers to avoid attempting to
  // search/register with a buffer size of 0.
  if (!ibSendPeers.empty()) {
    CTRAN_PROFILER_IF(
        profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));

    FB_COMMCHECK(searchRegHandle(
        comm,
        sendbuff,
        contigSendBufSize * commTypeSize(datatype),
        tmpHdl,
        tmpRegHdls));

    CTRAN_PROFILER_IF(
        profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));
  }
  int idx = 0;
  for (auto peer : ibSendPeers) {
    sendMemHdl[peer] = tmpHdl;
    FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl<PerfConfig>(
        &remoteRecvBuffs[peer],
        &remoteAccessKeys[peer],
        peer,
        &ibRecvCtrlReqs[idx++]));
  }

  static thread_local auto alltoallvIbConfig =
      comm->ctran_->algo->getCollToVcConfig(CollType::ALLTOALL);

  // issue network puts:
  // - Sender puts data for peers, whenever received the remote recvbuff handle
  // - Exit until all peers' put have been issued (putPeers becomes empty)
  std::vector<CtranMapperRequest> ibPutReqs(ibSendPeers.size());
  idx = 0;
  for (auto& recvCtrlReq : ibRecvCtrlReqs) {
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest<PerfConfig>(&recvCtrlReq));

    // Check whether it is the last request. We should end the algo ctrl event
    // after finish waiting the last request.
    if (&recvCtrlReq == &ibRecvCtrlReqs.back()) {
      CTRAN_PROFILER_IF(
          profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL));
    }

    const int peer = recvCtrlReq.peer;
    if (useProfiler) {
      timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));
    }
    auto sendSize = sendCounts[peer] * commTypeSize(datatype);

    // Check whether it is the first request. We should start timing the data
    // event after finish waiting the first request.
    if (&recvCtrlReq == &ibRecvCtrlReqs.front()) {
      CTRAN_PROFILER_IF(
          profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA));
    }
    // FIXME: we should compare sendSize with real maxWqeSize:
    // NCCL_CTRAN_IB_QP_SCALING_THRESHOLD may not be maxWqeSize if user
    // specified NCCL_CTRAN_IB_QP_CONFIG_ALGO to overwrite qp_scaling_threshold
    // for certain algo.
    bool enableFastPath = NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS &&
        (sendSize <= NCCL_CTRAN_IB_QP_SCALING_THRESHOLD);
    FB_COMMCHECK(comm->ctran_->mapper->iput<PerfConfig>(
        sendBuffs[peer],
        remoteRecvBuffs[peer],
        sendCounts[peer] * commTypeSize(datatype),
        peer,
        CtranMapperConfig{
            .memHdl_ = sendMemHdl[peer],
            .remoteAccessKey_ = remoteAccessKeys[peer],
            .notify_ = true, /*notify*/
            .ibConfig_ = alltoallvIbConfig,
            .ibFastPath_ = enableFastPath},
        &ibPutReqs[idx++]));
    if (useProfiler) {
      timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
    }
  }

  // Wait for all puts to complete
  FB_COMMCHECK(comm->ctran_->mapper->waitAllRequests<PerfConfig>(
      ibPutReqs, useProfiler ? (&timestamp->putComplete) : nullptr));
  // Wait for all receives (i.e., remote IB puts) to complete
  FB_COMMCHECK(comm->ctran_->mapper->waitAllNotifies<PerfConfig>(notifyVec));

  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  // Always wait for all sendCtrlReqs to complete so that the memory can be
  // safely reused in next collective; otherwise, ibvc may complete the previous
  // request while the memory has already been assigned to a new request.
  FB_COMMCHECK(
      comm->ctran_->mapper->waitAllRequests<PerfConfig>(ibSendCtrlReqs));

  CTRAN_PROFILER_IF(profiler, { profiler->reportToScuba(); });

  if (useProfiler) {
    comm->ctran_->mapper->timestamps.emplace_back(std::move(timestamp));
    comm->ctran_->mapper->reportProfiling();
  }

  /* deregister temporary registrations */
  // FIXME: let GPE kernel to finish then deregister to avoid race condition on
  // cuda context
  for (auto& hdl : tmpRegHdls) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(hdl));
  }

  return commSuccess;
}
#endif
