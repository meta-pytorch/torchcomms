// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::alltoallp::AlgoImpl;
using ctran::alltoallp::PersistArgs;

#define RETURN_ALLTOALLV_IB_IMPL(perfconfig) \
  return ctranAllToAllvIbImpl<perfconfig>(   \
      op->alltoallP.sendbuff,                \
      counts,                                \
      displs,                                \
      pArgs->recvbuff,                       \
      counts,                                \
      displs,                                \
      pArgs->datatype,                       \
      comm,                                  \
      std::move(timestamp));

#define RETURN_ALLTOALLP_IB_IMPL(perfconfig) \
  return ctranAllToAllPIbImpl<perfconfig>(   \
      op->alltoallP.sendbuff,                \
      counts,                                \
      displs,                                \
      pArgs->recvbuff,                       \
      counts,                                \
      displs,                                \
      pArgs->datatype,                       \
      comm,                                  \
      std::move(timestamp),                  \
      &remoteRecvBuffs,                      \
      &pArgs->remoteAccessKeys);

#define RETURN_IMPL(config)           \
  if (pArgs->skipCtrlMsg) {           \
    RETURN_ALLTOALLP_IB_IMPL(config); \
  } else {                            \
    RETURN_ALLTOALLV_IB_IMPL(config); \
  }

namespace {
const auto myAlgo = NCCL_ALLTOALL_ALGO::ctran;

template <typename PerfConfig = DefaultPerfCollConfig>
commResult_t ctranAllToAllPIbImpl(
    const void* sendbuff,
    std::vector<size_t>& sendCounts,
    std::vector<size_t>& sDispls,
    void* recvbuff,
    std::vector<size_t>& recvCounts,
    std::vector<size_t>& rDispls,
    commDataType_t datatype,
    CtranComm* comm,
    std::unique_ptr<CtranMapperTimestamp> timestamp,
    const std::vector<void*>* remoteRecvBuffsPtr,
    const std::vector<struct CtranMapperRemoteAccessKey>* remoteAccessKeysPtr) {
  const auto& statex = comm->statex_;
  const int myRank = statex->rank();
  const int nRanks = statex->nRanks();

  const std::string algoName = AlgoImpl::algoName(myAlgo);
  const bool useProfiler = NCCL_CTRAN_PROFILING != NCCL_CTRAN_PROFILING::none;

  std::vector<const void*> sendBuffs(nRanks);

  void* sendMemHdl = nullptr;
  std::vector<void*> tmpRegHdls;

  std::vector<int> ibRecvPeers, ibSendPeers;
  std::unordered_set<int> ibPeers;

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
  }

  // Prepare buffers shifted with displacement, and set ctrl/put/notify
  // schedules. Try to schedule ctrl message and put sequence as rank i start
  // sending to rank i+1 to avoid congestion in potential all-to-one case.
  // Specified in putPeers, sendCtrlPeers.
  size_t contigSendBufSize = 0;
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
      ibRecvPeers.push_back(peer);
      ibPeers.insert(peer);
    }
  }

  std::vector<CtranMapperNotify> notifyVec(ibRecvPeers.size());
  FB_COMMCHECK(comm->ctran_->mapper->initNotifyBatchIB(ibRecvPeers, notifyVec));

  // Search for the handle only when there are SendPeers to avoid attempting to
  // search/register with a buffer size of 0.
  if (!ibSendPeers.empty()) {
    // TODO: move this to main thread before submitting to GPE
    FB_COMMCHECK(searchRegHandle(
        comm,
        sendbuff,
        contigSendBufSize * commTypeSize(datatype),
        sendMemHdl,
        tmpRegHdls));
  }

  std::vector<CtranMapperRequest> ibPutReqs(ibSendPeers.size());
  int idx = 0;
  // Issue network puts using provided remote recvbuff handles
  for (const auto& peer : ibSendPeers) {
    if (useProfiler) {
      timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));
    }
    auto sendSize = sendCounts[peer] * commTypeSize(datatype);
    // FIXME: we should compare sendSize with real maxWqeSize:
    // NCCL_CTRAN_IB_QP_SCALING_THRESHOLD may not be maxWqeSize if user
    // specified NCCL_CTRAN_IB_QP_CONFIG_ALGO to overwrite qp_scaling_threshold
    // for certain algo.
    bool enableFastPath = NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS &&
        (sendSize <= NCCL_CTRAN_IB_QP_SCALING_THRESHOLD);
    FB_COMMCHECK(comm->ctran_->mapper->iput<PerfConfig>(
        sendBuffs[peer],
        (*remoteRecvBuffsPtr)[peer],
        sendSize,
        peer,
        CtranMapperConfig{
            .memHdl_ = sendMemHdl,
            .remoteAccessKey_ = (*remoteAccessKeysPtr)[peer],
            .notify_ = true /*notify*/,
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

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllToAllPIbImpl: rank {} completed AllToAllP execution sendbuff {} recvbuff {} count {} size {} ibPeers {}",
      comm->statex_->rank(),
      sendbuff,
      recvbuff,
      sendCounts[ibSendPeers[0]],
      sendCounts[ibSendPeers[0]] * commTypeSize(datatype),
      ibPeers.size());
  return commSuccess;
}

commResult_t exchangeMemHdl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = op->comm_;
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();
  const auto myRank = statex->rank();
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->alltoallP.pArgs);

  auto recvbuff = pArgs->recvbuff;
  const std::vector<void*> recvBuffs(nRanks, recvbuff);
  auto recvMemHdl = pArgs->recvHdl;
  auto& remoteRecvBuffs = pArgs->remoteRecvBuffs;
  auto& remoteAccessKeys = pArgs->remoteAccessKeys;
  remoteRecvBuffs.resize(nRanks);
  remoteAccessKeys.resize(nRanks);

  std::vector<int> ibPeers;
  for (int p = 1; p < nRanks; p++) {
    const int peer = (myRank + p) % nRanks;
    if (!statex->isSameNode(myRank, peer)) {
      ibPeers.push_back(peer);
    }
  }
  std::vector<CtranMapperRequest> ibSendCtrlReqs(ibPeers.size()),
      ibRecvCtrlReqs(ibPeers.size());

  auto mapper = comm->ctran_->mapper.get();
  FB_COMMCHECK(mapper->isendCtrlBatch(
      recvBuffs, recvMemHdl, ibPeers, ibSendCtrlReqs, CtranMapperBackend::IB));
  int id = 0;
  for (const auto peer : ibPeers) {
    FB_COMMCHECK(mapper->irecvCtrl(
        &remoteRecvBuffs[peer],
        &remoteAccessKeys[peer],
        peer,
        &ibRecvCtrlReqs[id++]));
  }
  FB_COMMCHECK(mapper->waitAllRequests(ibSendCtrlReqs));
  FB_COMMCHECK(mapper->waitAllRequests(ibRecvCtrlReqs));
  return commSuccess;
}

commResult_t gpeFn(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = op->comm_;
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  std::vector<size_t> counts(nRanks, 0);
  std::vector<size_t> displs(nRanks, 0);

  CtranAlgoLogger logger(AlgoImpl::algoName(myAlgo), op->opCount, comm);

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(AlgoImpl::algoName(myAlgo)));

  const int myNode = statex->node();
  for (int i = 0; i < nRanks; i++) {
    int peerNode = statex->node(i);
    displs[i] = op->alltoallP.count * i;
    // GPE thread handles only remote peers
    if (myNode != peerNode) {
      counts[i] = op->alltoallP.count;
    }
  }

  // Update remoterecvbuf based on displs. Since displs[] is the same on all
  // peers in AllToAll, we can use my own rank's displs to calculate the
  // recvbuff on remote peers.
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->alltoallP.pArgs);
  std::vector<void*> remoteRecvBuffs(statex->nRanks());
  const int myrank = comm->statex_->rank();
  for (int i = 0; i < statex->nRanks(); i++) {
    remoteRecvBuffs[i] = static_cast<char*>(pArgs->remoteRecvBuffs[i]) +
        displs[myrank] * commTypeSize(pArgs->datatype);
  }

  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    RETURN_IMPL(LowLatencyCollConfig)
  } else {
    RETURN_IMPL(DefaultPerfCollConfig)
  }
}
} // namespace

namespace ctran::alltoallp {
extern __global__ void ncclKernelAllToAllPInitWait(
    int* flag,
    CtranAlgoDeviceState* devState);

commResult_t AlgoImpl::init() {
  // No MemHdl exchange needed if no remote peer.
  if (comm_->statex_->nNodes() < 2) {
    return commSuccess;
  }
  auto opCount = comm_->ctran_->getOpCount();

  // A dummy kernel that does nothing, only used to track op completion
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALL, stream_, algoName(myAlgo), opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = comm_->ctran_->algo->getDevState();
  // For colltrace to work, need to init followings.
  // FIXME: Add colltrace support for stub kernel used in PInit.
  config.args.collective.alltoall.datatype = pArgs.datatype;
  config.args.collective.alltoall.count = 0;
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLTOALLP, stream_, comm_, opCount);
  op->alltoallP.pArgs = &pArgs;

  opGroup.push_back(std::move(op));
  CLOGF_TRACE(
      COLL,
      "AllToAllPInit: rank {} submit GPE op pArgs {}",
      comm_->statex_->rank(),
      opGroup.front().get()->alltoallP.pArgs);

  // Fixme: kernelConfig is only needed for colltrace, remove it.
  // Use submitHost for two reasons: 1. no cuda kernel is needed in
  // exchangeMemHdl 2. avoid cudagraph capture exchangeMemHdl
  FB_COMMCHECK(comm_->ctran_->gpe->submitHost(
      std::move(opGroup),
      exchangeMemHdl,
      config,
      /* exReq */ nullptr,
      /* allowNullReq */ true));
  return commSuccess;
}

commResult_t AlgoImpl::exec(const void* sendbuff, const size_t count) {
  auto recvbuff = pArgs.recvbuff;
  auto datatype = pArgs.datatype;
  auto opCount = comm_->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      algoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      0UL,
      datatype,
      -1,
      comm_,
      stream_);

  if (count == 0) {
    return commSuccess;
  }
  if (count * comm_->statex_->nRanks() > pArgs.maxRecvCount) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "AllToAllP send/recv count {} times nRanks {} exceeds maximum count {}.",
        count,
        comm_->statex_->nRanks(),
        pArgs.maxRecvCount);
  }

  // prepare kernel config for NVL copies, reuse the alltoall kernel
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALL, stream_, algoName(myAlgo), opCount);
  FB_COMMCHECK(ctran::alltoall::setupKernelConfig(
      sendbuff, recvbuff, count, datatype, comm_, stream_, config));
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  // Passing op only when remote peers are present
  if (comm_->statex_->nNodes() > 1) {
    auto op = std::make_unique<OpElem>(
        OpElem::opType::ALLTOALLP, stream_, comm_, opCount);
    op->alltoallP.pArgs = &pArgs;
    op->alltoallP.sendbuff = sendbuff;
    op->alltoallP.count = count;
    opGroup.push_back(std::move(op));
    CLOGF_TRACE(
        COLL,
        "AllToAllPExec: rank {} submit op sendbuff {} count {}",
        comm_->statex_->rank(),
        opGroup.front().get()->alltoallP.sendbuff,
        opGroup.front().get()->alltoallP.count);
  }

  FB_COMMCHECK(comm_->ctran_->gpe->submit(
      std::move(opGroup),
      gpeFn,
      config,
      reinterpret_cast<void*>(ctran::alltoall::alltoallKerns[datatype])));
  return commSuccess;
}
} // namespace ctran::alltoallp
