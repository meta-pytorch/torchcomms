// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <iostream>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/CommUtils.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/checks.h"

using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::PersistArgs;
using ncclx::CommStateX;
namespace {
const auto myAlgo = NCCL_ALLGATHER_P_ALGO::ctdirect;

commResult_t gpnFn(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  auto op = opGroup.front().get();
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->allgatherP.pArgs);
  const auto sendSize =
      op->allgatherP.count * commTypeSize(op->allgatherP.datatype);
  const void* sendBuff = op->allgatherP.sendbuff;
  CtranComm* comm = opGroup.front()->comm_;

  const auto statex = comm->statex_.get();
  const auto rank = statex->rank();
  const auto nRanks = statex->nRanks();

  CtranAlgoLogger logger(AlgoImpl::algoName(myAlgo), op->opCount, comm);

  ctran::Profiler* profiler = comm->ctran_->profiler.get();
  if (profiler) {
    profiler->initForEachColl(
        op->opCount, NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT);
  }

  CTRAN_PROFILER_IF(profiler, {
    auto& algoContext = profiler->algoContext;
    algoContext.algorithmName = AlgoImpl::algoName(myAlgo);
    algoContext.sendContext.messageSizes = std::to_string(sendSize);
    algoContext.recvContext.messageSizes = std::to_string(sendSize * nRanks);
  });

  std::vector<std::unique_ptr<CtranMapperRequest>> pReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> sReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> rReqs;
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec;
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranAllgatherPDirect"));

  auto mapper = comm->ctran_->mapper.get();

  void* sendHdl = nullptr;
  bool localReg = false;
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));
  FB_COMMCHECK(
      mapper->searchRegHandle(sendBuff, sendSize, &sendHdl, &localReg));
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));
  auto guard = folly::makeGuard([sendHdl, localReg, mapper]() {
    if (localReg) {
      FB_COMMCHECKIGNORE(mapper->deregDynamic(sendHdl));
    }
  });

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL));
  // Sync to make sure ib peers are ready to receive
  for (int p = 1; p < nRanks; p++) {
    CtranMapperRequest* req = nullptr;
    const int peer = (rank + p) % nRanks;
    if (pArgs->remoteAccessKeys[peer].backend == CtranMapperBackend::IB) {
      FB_COMMCHECK(mapper->irecvCtrl(peer, &req));
      rReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
      FB_COMMCHECK(mapper->isendCtrl(peer, &req));
      sReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
    }
  }
  for (auto& req : rReqs) {
    FB_COMMCHECK(mapper->waitRequest(req.get()));
  }
  for (auto& req : sReqs) {
    FB_COMMCHECK(mapper->waitRequest(req.get()));
  }
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL));

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA));
  // Issue PUT operations
  for (auto p = 1; p < nRanks; p++) {
    CtranMapperRequest* req = nullptr;
    const auto peer = (rank + p) % nRanks;
    if (pArgs->remoteAccessKeys[peer].backend == CtranMapperBackend::IB) {
      // Initialize notify flag to receive from peer
      auto notify = std::make_unique<CtranMapperNotify>();
      FB_COMMCHECK(mapper->initNotify(peer, pArgs->recvHdl, notify.get()));
      notifyVec.push_back(std::move(notify));

      // Issue put to IB peers
      FB_COMMCHECK(mapper->iput(
          sendBuff,
          (void*)((uintptr_t)pArgs->remoteRecvBuffs[peer] + rank * sendSize),
          sendSize,
          peer,
          CtranMapperConfig{
              .memHdl_ = sendHdl,
              .remoteAccessKey_ = pArgs->remoteAccessKeys[peer],
              .notify_ = true},
          &req));
      pReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
      timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
    }
  }

  // Wait for all remote PUTs to arrive
  for (auto& notify : notifyVec) {
    FB_COMMCHECK(mapper->waitNotify(notify.get()));
  }
  // Wait for all local PUTs to complete
  for (auto& req : pReqs) {
    FB_COMMCHECK(mapper->waitRequest(req.get()));
  }
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  mapper->timestamps.emplace_back(std::move(timestamp));
  mapper->reportProfiling();

  CTRAN_PROFILER_IF(profiler, { profiler->reportToScuba(); });

  return commSuccess;
}
} // namespace

namespace ctran::allgatherp {
extern __global__ void ncclKernelAllGatherPDirect(
    int* flag,
    CtranAlgoDeviceState* devState);

commResult_t AlgoImpl::execDirect(
    const void* sendbuff,
    const size_t count,
    const commDataType_t datatype) {
  auto recvbuff = pArgs.recvbuff;
  auto ctran = comm_->ctran_.get();
  const auto opCount = ctran->getOpCount();
  const auto myRank = comm_->statex_->rank();
  const auto nLocalRanks = comm_->statex_->nRanks();

  CTRAN_COLL_INFO(
      AlgoImpl::algoName(myAlgo),
      sendbuff,
      recvbuff,
      count,
      datatype,
      -1,
      comm_,
      stream_);

  const auto sendSize = count * commTypeSize(datatype);

  // Copy data to self for out-of-place allgather
  FB_COMMCHECK(copyToSelf(comm_, sendbuff, sendSize, pArgs, stream_));

  // Wait till async init is done, so that we can schedule copy operations with
  // the remote address
  if (nLocalRanks > 1) {
    FB_COMMCHECK(waitInit());
  }

  // Copy data to other local ranks via NVL CE broadcast, if NVL is available
  const auto statex = comm_->statex_.get();
  const auto actualNLocalRanks = statex->nLocalRanks();
  if (actualNLocalRanks > 1) {
    const auto localPeer =
        statex->localRankToRank((statex->localRank() + 1) % actualNLocalRanks);
    if (pArgs.remoteAccessKeys[localPeer].backend == CtranMapperBackend::NVL) {
      FB_COMMCHECK(nvlCeBcast(
          comm_, sendbuff, sendSize, myRank * sendSize, pArgs, stream_));
    }
  }

  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLGATHERP, stream_, comm_, opCount);
  op->allgatherP.pArgs = &pArgs;
  op->allgatherP.algoResource = &resource_;
  op->allgatherP.sendbuff = sendbuff;
  op->allgatherP.count = count;
  op->allgatherP.datatype = datatype;

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.push_back(std::move(op));

  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP,
      stream_,
      AlgoImpl::algoName(myAlgo),
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.algoArgs = reinterpret_cast<void*>(&pArgs);
  config.args.devState_d = ctran->algo->getDevState();

  FB_COMMCHECK(ctran->gpe->submit(
      std::move(opGroup),
      gpnFn,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherPDirect)));
  return commSuccess;
}
} // namespace ctran::allgatherp
