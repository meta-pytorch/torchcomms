// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAll/AllToAllvDynamicPImpl.h"
#include "Types.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicCommon.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::alltoallvdynamicp::AlgoImpl;
using ctran::alltoallvdynamicp::PersistArgs;

#define PUT_AND_WAIT(perfconfig)                                              \
  do {                                                                        \
    FB_COMMCHECK(                                                             \
        peerPutNonContig<perfconfig>(                                         \
            comm,                                                             \
            op->alltoallv_dynamic.sendbuffs,                                  \
            pArgs->remoteRecvBuffs,                                           \
            sendCountsTmpbufCPU,                                              \
            op->alltoallv_dynamic.sendcountsLength,                           \
            op->alltoallv_dynamic.datatype,                                   \
            tmpRegHdls,                                                       \
            nRanks,                                                           \
            myRank,                                                           \
            timestamp,                                                        \
            pArgs->remoteAccessKeys,                                          \
            ibPutReqs,                                                        \
            completedIbRecvCtrlReqs,                                          \
            pArgs->maxRecvCount,                                              \
            pArgs->maxSendCount,                                              \
            op->alltoallv_dynamic.combine,                                    \
            /* skipWaitRecvCtrl */ true));                                    \
    /* Wait for all puts to complete */                                       \
    for (auto& req : ibPutReqs) {                                             \
      FB_COMMCHECK(comm->ctran_->mapper->waitRequest<perfconfig>(req.get())); \
    }                                                                         \
    /* Wait for all receives (i.e., remote IB puts) to complete */            \
    for (auto& notify : notifyVec) {                                          \
      FB_COMMCHECK(                                                           \
          comm->ctran_->mapper->waitNotify<perfconfig>(notify.get()));        \
    }                                                                         \
  } while (0)

namespace {
const auto myAlgo = NCCL_ALLTOALL_ALGO::ctran;

// FIXME: may share the code with AllToAllP
commResult_t exchangeMemHdl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = op->comm_;
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();
  const auto myRank = statex->rank();
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->alltoallv_dynamic.pArgs);

  auto recvBuffs = pArgs->recvbuffs;
  auto recvHdls = pArgs->recvHdls;
  auto& remoteRecvBuffs = pArgs->remoteRecvBuffs;
  auto& remoteAccessKeys = pArgs->remoteAccessKeys;
  remoteRecvBuffs.resize(nRanks);
  remoteAccessKeys.resize(nRanks);

  std::vector<int> ibPeers;
  for (int i = 1; i < nRanks; i++) {
    const int peer = (myRank + i) % nRanks;
    if (!statex->isSameNode(myRank, peer)) {
      ibPeers.push_back(peer);
    }
  }
  std::vector<CtranMapperRequest> ibSendCtrlReqs(ibPeers.size()),
      ibRecvCtrlReqs(ibPeers.size());

  auto mapper = comm->ctran_->mapper.get();

  int id = 0;
  for (const auto peer : ibPeers) {
    FB_COMMCHECK(mapper->isendCtrl(
        recvBuffs[peer], recvHdls[peer], peer, &ibSendCtrlReqs[id]));
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
  // The impl is almost the same as ctranAllToAllvDynamicIbImpl, the only
  // difference is that not exchanging memhdl.
  // FIXME: may share more code with non-persistent version
  // ctranAllToAllvDynamicIbImpl besides the put part.
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = op->comm_;
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();
  const auto myRank = statex->rank();

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("ncclx::alltoallvDynamic"));

  size_t* sendCountsTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDCOUNTS_TMPBUF_CPU));
  std::vector<void*> tmpRegHdls;
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->alltoallv_dynamic.pArgs);
  std::vector<std::unique_ptr<CtranMapperRequest>> ibPutReqs;
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec;
  std::vector<std::unique_ptr<CtranMapperRequest>> completedIbRecvCtrlReqs;
  for (int peer = 0; peer < nRanks; peer++) {
    if (statex->isSameNode(myRank, peer)) {
      continue;
    }
    // To share the same interface with non-persisent collective, init the
    // completed ctrl reqs. The only usefuly info in it is "peer".
    completedIbRecvCtrlReqs.push_back(
        std::make_unique<CtranMapperRequest>(
            CtranMapperRequest::RECV_CTRL, peer));
    // Initialize notify flag to receive from peer
    auto notify = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(comm->ctran_->mapper->initNotify(
        peer, pArgs->recvHdls[peer], notify.get()));
    notifyVec.push_back(std::move(notify));
  }

  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    PUT_AND_WAIT(LowLatencyCollConfig);
  } else {
    PUT_AND_WAIT(DefaultPerfCollConfig);
  }

  if (statex->nNodes() > 1) {
    op->alltoallv_dynamic.kElem->post(1);
    op->alltoallv_dynamic.kElem->wait(1);
  }

  /* deregister temporary registrations */
  // FIXME: let GPE kernel to finish then deregister to avoid race condition
  // on cuda context
  for (auto& hdl : tmpRegHdls) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(hdl));
  }
  CLOGF_SUBSYS(
      INFO, COLL, "AllToAllvDynamicP: rank {} completed execution", myRank);
  return commSuccess;
}
} // namespace

namespace ctran::alltoallvdynamicp {

commResult_t AlgoImpl::init() {
  // No MemHdl exchange needed if no remote peer.
  if (comm_->statex_->nNodes() < 2) {
    return commSuccess;
  }
  // search recv handles
  const int nRanks = comm_->statex_->nRanks();
  const int myRank = comm_->statex_->rank();
  pArgs.recvHdls.resize(nRanks);
  for (int i = 0; i < nRanks; i++) {
    if (comm_->statex_->isSameNode(myRank, i)) {
      continue;
    }
    bool localReg = false;
    FB_COMMCHECK(comm_->ctran_->mapper->searchRegHandle(
        pArgs.recvbuffs[i], pArgs.maxRecvCount, &pArgs.recvHdls[i], &localReg));
    if (localReg) {
      // AllToAllvDynamicP requires recvbuff to be pre-registered. Otherwise, it
      // will fail.
      comm_->ctran_->mapper->deregDynamic(pArgs.recvHdls[i]);
      CLOGF(
          ERR,
          "recvbuff is not registered. Pointer: {} length: {}",
          pArgs.recvbuffs[i],
          pArgs.maxRecvCount);
      return commInternalError;
    }
  }
  auto opCount = comm_->ctran_->getOpCount();

  // For colltrace to work, need to init followings. The kernel won't launch so
  // these are all dummy values.
  // FIXME: Add colltrace support for stub kernel used in PInit.
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALL, stream_, algoName(myAlgo), opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = comm_->ctran_->algo->getDevState();
  config.args.collective.alltoall.datatype = pArgs.datatype;
  config.args.collective.alltoall.count = 0;
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  // FIXME: only support split_non_contig for now
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG_P,
      stream_,
      comm_,
      opCount);
  op->alltoallv_dynamic.pArgs = &pArgs;

  opGroup.push_back(std::move(op));
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllToAllvDynamicPInit: rank {} submit GPE op pArgs {}",
      comm_->statex_->rank(),
      (void*)opGroup.front().get()->alltoallv_dynamic.pArgs);

  // Fixme: kernelConfig is only needed for colltrace, remove it.
  // Use submitHost for two reasons: 1. no cuda kernel is needed in
  // exchangeMemHdl 2. avoid cudagraph capture exchangeMemHdl
  FB_COMMCHECK(comm_->ctran_->gpe->submitHost(
      std::move(opGroup),
      exchangeMemHdl,
      config,
      /* exReq */ nullptr));
  return commSuccess;
}

commResult_t AlgoImpl::updatePersistFuncAndOp(
    opFunc& opFunc,
    struct OpElem* op) {
  opFunc = gpeFn;
  // FIXME: only support split_non_contig for now
  op->type = OpElem::opType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG_P;
  op->alltoallv_dynamic.pArgs = &pArgs;
  CLOGF_TRACE(
      COLL,
      "AllToAllvDynamicP: rank {} updated op to {} and gpeFn to persistent version.",
      comm_->statex_->rank(),
      (void*)op);
  return commSuccess;
}
} // namespace ctran::alltoallvdynamicp
