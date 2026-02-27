// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <iostream>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/checks.h"

#include "comms/ctran/algos/AllToAllvP/AllToAllvPImpl.h"
#include "comms/ctran/algos/AllToAllvP/CommUtils.h"
#include "comms/ctran/algos/AllToAllvP/Types.h"

using ctran::alltoallvp::AlgoImpl;
using ctran::alltoallvp::PersistArgs;

namespace {
const auto myAlgo = NCCL_ALLTOALLV_P_ALGO::ctran;
const std::string algoInitName = "CtranAllToAllvPInit";
const std::string algoExecName = "CtranAllToAllvPExec";

// exchangeMemHdlInit: Called during init() phase
// Uses allGatherCtrl to share the base recvbuff address with all peers
// (rdispls is not available at init time)
commResult_t exchangeMemHdlInit(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = op->comm_;
  auto mapper = comm->ctran_->mapper.get();

  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  CtranAlgoLogger logger(algoInitName, op->opCount, comm);

  auto* pArgs = reinterpret_cast<PersistArgs*>(op->alltoallvP.pArgs);
  pArgs->remoteAccessKeys.resize(nRanks, CtranMapperRemoteAccessKey());
  pArgs->remoteRecvBuffs.resize(nRanks, nullptr);

  // Use allGatherCtrl to share the base recvbuff with all peers
  FB_COMMCHECK(mapper->allGatherCtrl(
      pArgs->recvbuff,
      pArgs->recvHdl,
      pArgs->remoteRecvBuffs,
      pArgs->remoteAccessKeys));

  // Ensure all ranks have finished remote importing before return
  FB_COMMCHECK(mapper->barrier());

  if (NCCL_CTRAN_ENABLE_TRACE_LOG) {
    for (int i = 0; i < nRanks; i++) {
      CLOGF_TRACE(
          INIT,
          "    remoteRecvBuffs[{}]: {}, remoteAccessKey: {}",
          i,
          (void*)pArgs->remoteRecvBuffs[i],
          pArgs->remoteAccessKeys[i].toString());
    }
  }

  // Mark the remote registration as initialized, so that consequent execution
  // can schedule CE based NVL copy
  pArgs->initialized.store(true);
  return commSuccess;
}

// exchangeMemHdlExec: Called during exec() phase via gpeFn
// Uses allToAllCtrl to share per-peer recvbuff offsets based on rdispls
commResult_t exchangeMemHdlExec(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = op->comm_;
  auto mapper = comm->ctran_->mapper.get();

  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  CtranAlgoLogger logger(algoExecName, op->opCount, comm);

  auto* pArgs = reinterpret_cast<PersistArgs*>(op->alltoallvP.pArgs);

  // Get rdispls and datatype from op
  auto& rDispls = op->alltoallvP.rdispls;
  auto datatype = op->alltoallvP.datatype;
  auto recvbuff = pArgs->recvbuff;
  auto recvHdl = pArgs->recvHdl;

  // Initialize per-peer recvBuffs and recvHdls based on rdispls offsets
  pArgs->recvBuffs.resize(nRanks);
  pArgs->recvHdls.resize(nRanks);
  for (int i = 0; i < nRanks; i++) {
    auto recvOffset = rDispls[i] * commTypeSize(datatype);
    pArgs->recvBuffs[i] =
        static_cast<void*>(static_cast<char*>(recvbuff) + recvOffset);
    pArgs->recvHdls[i] = recvHdl;
  }

  pArgs->remoteAccessKeys.resize(nRanks, CtranMapperRemoteAccessKey());
  pArgs->remoteRecvBuffs.resize(nRanks, nullptr);

  // Use allToAllCtrl to exchange per-peer receive buffer offsets
  FB_COMMCHECK(mapper->allToAllCtrl(
      pArgs->recvBuffs,
      pArgs->recvHdls,
      pArgs->remoteRecvBuffs,
      pArgs->remoteAccessKeys));

  // Ensure all ranks have finished remote importing before return
  FB_COMMCHECK(mapper->barrier());

  if (NCCL_CTRAN_ENABLE_TRACE_LOG) {
    for (int i = 0; i < nRanks; i++) {
      CLOGF_TRACE(
          COLL,
          "    remoteRecvBuffs[{}]: {}, remoteAccessKey: {}",
          i,
          (void*)pArgs->remoteRecvBuffs[i],
          pArgs->remoteAccessKeys[i].toString());
    }
  }

  // Signal to the main thread that address exchange is complete
  pArgs->exchanged.store(true);
  return commSuccess;
}

// gpeFn: Called during exec() phase
// 1. Calls exchangeMemHdlExec to exchange per-peer addresses using allToAllCtrl
// 2. Issues PUT operations for IB domain peers (inter-node)
// NVL domain peers (intra-node) are handled by nvlCeBcast() on the main thread
template <typename PerfConfig = DefaultPerfCollConfig>
commResult_t gpeFn(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();

  CtranComm* comm = op->comm_;
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();
  const auto myRank = statex->rank();

  // Step 1: Exchange per-peer receive buffer addresses using allToAllCtrl
  FB_COMMCHECK(exchangeMemHdlExec(opGroup));

  // persistent variables (pre-allocated in exchangeMemHdl during init)
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->alltoallvP.pArgs);
  auto& remoteRecvBuffs = pArgs->remoteRecvBuffs;
  auto& remoteAccessKeys = pArgs->remoteAccessKeys;

  // non-persistent variables
  auto sendbuff = op->alltoallvP.sendbuff;
  auto& sendCounts = op->alltoallvP.sendcounts;
  auto& sDispls = op->alltoallvP.sdispls;
  auto datatype = op->alltoallvP.datatype;

  // Step 2: Issue PUT operations for IB domain peers only
  // IB peers are peers NOT on the same node (inter-node communication)
  // NVL peers (same node) are handled by nvlCeBcast() on the main thread
  std::vector<int> ibSendPeers;
  for (int i = 0; i < nRanks; i++) {
    int peer = (myRank + i) % nRanks;
    if (!statex->isSameNode(myRank, peer) && sendCounts[peer] > 0) {
      ibSendPeers.push_back(peer);
    }
  }

  // Return if there are no IB peers to send to
  if (ibSendPeers.empty()) {
    return commSuccess;
  }

  auto mapper = comm->ctran_->mapper.get();

  // Prepare send buffers for IB peers
  std::vector<const void*> sendBuffs(nRanks);
  size_t maxSendBufCount = 0;
  for (const auto& peer : ibSendPeers) {
    sendBuffs[peer] = static_cast<const char*>(sendbuff) +
        sDispls[peer] * commTypeSize(datatype);
    maxSendBufCount =
        std::max(maxSendBufCount, sDispls[peer] + sendCounts[peer]);
  }

  // Get send handle for the sendbuff
  void* sendHdl = nullptr;
  bool localReg = false;
  FB_COMMCHECK(mapper->searchRegHandle(
      sendbuff, maxSendBufCount * commTypeSize(datatype), &sendHdl, &localReg));
  auto guard = folly::makeGuard([sendHdl, localReg, mapper]() {
    if (localReg) {
      FB_COMMCHECKIGNORE(mapper->deregDynamic(sendHdl));
    }
  });

  // Initialize notify batch for IB PUTs
  std::vector<CtranMapperNotify> notifyVec(ibSendPeers.size());
  FB_COMMCHECK(mapper->initNotifyBatchIB(ibSendPeers, notifyVec));

  // Issue PUT operations to all IB peers
  std::vector<CtranMapperRequest> ibPutReqs(ibSendPeers.size());
  int idx = 0;
  for (const auto& peer : ibSendPeers) {
    auto sendSize = sendCounts[peer] * commTypeSize(datatype);
    FB_COMMCHECK(mapper->iput<PerfConfig>(
        sendBuffs[peer],
        remoteRecvBuffs[peer],
        sendSize,
        peer,
        CtranMapperConfig{
            .memHdl_ = sendHdl,
            .remoteAccessKey_ = remoteAccessKeys[peer],
            .notify_ = true},
        &ibPutReqs[idx++]));
  }

  // Wait for all PUT operations to complete
  FB_COMMCHECK(mapper->waitAllRequests<PerfConfig>(ibPutReqs));
  // Wait for all receives (i.e., remote IB puts from other ranks) to complete
  FB_COMMCHECK(mapper->waitAllNotifies<PerfConfig>(notifyVec));

  return commSuccess;
}

} // namespace

namespace ctran::alltoallvp {

extern __global__ void ncclKernelAllToAllvPInit(
    int* flag,
    CtranAlgoDeviceState* devState);

commResult_t AlgoImpl::init() {
  auto opCount = comm_->ctran_->getOpCount();

  // set up GPE KernelConfig
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALLVP, stream_, algoName(myAlgo), opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = comm_->ctran_->algo->getDevState();

  // set up GPE op
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLTOALLVP, stream_, comm_, opCount);
  // persistent variables
  op->alltoallvP.pArgs = &pArgs;
  // non-persistent variables
  // None in init()
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.push_back(std::move(op));

  CLOGF_TRACE(
      COLL,
      "AllToAllvPInit: rank {} submit GPE op pArgs {}",
      comm_->statex_->rank(),
      opGroup.front().get()->alltoallvP.pArgs);

  FB_COMMCHECK(comm_->ctran_->gpe->submit(
      std::move(opGroup),
      exchangeMemHdlInit,
      config,
      reinterpret_cast<void*>(ncclKernelAllToAllvPInit)));

  return commSuccess;
}

extern __global__ void ncclKernelAllToAllvP(
    int* flag,
    CtranAlgoDeviceState* devState);

commResult_t AlgoImpl::exec(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    const size_t recvcounts[],
    const size_t rdispls[],
    const commDataType_t datatype) {
  const auto statex = comm_->statex_.get();
  // const auto nNodes = statex->nNodes();
  const auto nRanks = statex->nRanks();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto myRank = statex->rank();

  // Copy data to self for out-of-place alltoallv
  auto recvbuff = pArgs.recvbuff;
  FB_COMMCHECK(copyToSelf(
      comm_,
      sendbuff,
      recvbuff,
      sendcounts[myRank],
      sdispls[myRank],
      rdispls[myRank],
      datatype,
      stream_));

  // Wait till async init is done, so that we can schedule copy operations with
  // the remote address
  if (nRanks > 1) {
    FB_COMMCHECK(waitInit());
  }

  // Set up GPE for allToAllCtrl exchange
  auto ctran = comm_->ctran_.get();
  auto opCount = ctran->getOpCount();
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLTOALLVP,
      stream_,
      AlgoImpl::algoName(myAlgo),
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.algoArgs = reinterpret_cast<void*>(&pArgs);
  config.args.devState_d = ctran->algo->getDevState();

  // Set up GPE op with rdispls for per-peer address exchange
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLTOALLVP, stream_, comm_, opCount);
  op->alltoallvP.pArgs = &pArgs;
  op->alltoallvP.sendbuff = sendbuff;
  op->alltoallvP.datatype = datatype;

  op->alltoallvP.sendcounts.assign(sendcounts, sendcounts + nRanks);
  op->alltoallvP.sdispls.assign(sdispls, sdispls + nRanks);
  op->alltoallvP.recvcounts.assign(recvcounts, recvcounts + nRanks);
  op->alltoallvP.rdispls.assign(rdispls, rdispls + nRanks);

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.push_back(std::move(op));

  // Reset the address exchange flag before submitting GPE
  // exchangeMemHdlExec() will set this to true after address exchange completes
  pArgs.exchanged.store(false);

  // Submit GPE to:
  // 1. Exchange per-peer receive buffer addresses using allToAllCtrl
  // 2. Issue PUT operations for IB domain peers (inter-node)
  FB_COMMCHECK(ctran->gpe->submit(
      std::move(opGroup),
      gpeFn<>,
      config,
      reinterpret_cast<void*>(ncclKernelAllToAllvP)));

  if (nLocalRanks > 1) {
    // Wait for GPE thread to complete the address exchange
    // before using remoteRecvBuffs for NVL CE copy
    if (nRanks > 1) {
      FB_COMMCHECK(waitExchange());
    }

    // Copy data to other local ranks using NVL Copy Engine
    // Now uses the exchanged per-peer remote addresses from allToAllCtrl
    FB_COMMCHECK(nvlCeBcast(
        comm_, sendbuff, sendcounts, sdispls, datatype, pArgs, stream_));
  }
  return commSuccess;
}
} // namespace ctran::alltoallvp
