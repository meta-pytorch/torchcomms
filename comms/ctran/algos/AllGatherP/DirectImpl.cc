// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <iostream>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/CommUtils.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/checks.h"

using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::Buffer;
using ncclx::CommStateX;
namespace {
const auto myAlgo = NCCL_ALLGATHER_P_ALGO::ctdirect;

commResult_t gpnFn(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  auto op = opGroup.front().get();
  auto* buffer = reinterpret_cast<Buffer*>(op->allgatherP.pArgs);
  const auto sendSize =
      op->allgatherP.count * commTypeSize(op->allgatherP.datatype);
  const void* sendBuff = op->allgatherP.sendbuff;
  CtranComm* comm = opGroup.front()->comm_;

  const auto statex = comm->statex_.get();
  const auto rank = statex->rank();
  const auto nRanks = statex->nRanks();

  CtranAlgoLogger logger(AlgoImpl::algoName(myAlgo), op->opCount, comm);

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
  FB_COMMCHECK(
      mapper->searchRegHandle(sendBuff, sendSize, &sendHdl, &localReg));
  auto guard = folly::makeGuard([sendHdl, localReg, mapper]() {
    if (localReg) {
      FB_COMMCHECKIGNORE(mapper->deregDynamic(sendHdl));
    }
  });

  // Sync to make sure ib peers are ready to receive
  for (int p = 1; p < nRanks; p++) {
    CtranMapperRequest* req = nullptr;
    const int peer = (rank + p) % nRanks;
    if ((*buffer->remoteAccessKeys)[peer].backend == CtranMapperBackend::IB) {
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

  // Issue PUT operations
  for (auto p = 1; p < nRanks; p++) {
    CtranMapperRequest* req = nullptr;
    const auto peer = (rank + p) % nRanks;
    if ((*buffer->remoteAccessKeys)[peer].backend == CtranMapperBackend::IB) {
      // Initialize notify flag to receive from peer
      auto notify = std::make_unique<CtranMapperNotify>();
      FB_COMMCHECK(mapper->initNotify(peer, buffer->recvHdl, notify.get()));
      notifyVec.push_back(std::move(notify));

      // Issue put to IB peers
      FB_COMMCHECK(mapper->iput(
          sendBuff,
          (void*)((uintptr_t)(*buffer->remoteRecvBuffs)[peer] +
                  rank * sendSize),
          sendSize,
          peer,
          CtranMapperConfig{
              .memHdl_ = sendHdl,
              .remoteAccessKey_ = (*buffer->remoteAccessKeys)[peer],
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
  mapper->timestamps.emplace_back(std::move(timestamp));
  mapper->reportProfiling();

  return commSuccess;
}
} // namespace

namespace ctran::allgatherp {
extern __global__ void ncclKernelAllGatherPDirect(
    int* flag,
    CtranAlgoDeviceState* devState);

commResult_t execDirectCore(
    const void* sendbuff,
    const size_t count,
    const commDataType_t datatype,
    Buffer& buffer,
    Resource& resource,
    CtranComm* comm,
    cudaStream_t stream) {
  auto ctran = comm->ctran_.get();
  const auto opCount = ctran->getOpCount();
  const auto myRank = comm->statex_->rank();

  CTRAN_COLL_INFO(
      AlgoImpl::algoName(myAlgo),
      sendbuff,
      buffer.recvbuff,
      count,
      datatype,
      -1,
      comm,
      stream);

  const auto sendSize = count * commTypeSize(datatype);

  // Copy data to self for out-of-place allgather
  FB_COMMCHECK(copyToSelf(comm, sendbuff, sendSize, buffer.recvbuff, stream));

  // Copy data to other local ranks
  FB_COMMCHECK(nvlCeBcast(
      comm,
      sendbuff,
      sendSize,
      myRank * sendSize,
      *buffer.remoteRecvBuffs,
      *buffer.remoteAccessKeys,
      stream));

  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLGATHERP, stream, comm, opCount);
  op->allgatherP.pArgs = &buffer;
  op->allgatherP.algoResource = &resource;
  op->allgatherP.sendbuff = sendbuff;
  op->allgatherP.count = count;
  op->allgatherP.datatype = datatype;

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.push_back(std::move(op));

  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP,
      stream,
      AlgoImpl::algoName(myAlgo),
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = ctran->algo->getDevState();

  FB_COMMCHECK(ctran->gpe->submit(
      std::move(opGroup),
      gpnFn,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherPDirect)));
  return commSuccess;
}

commResult_t AlgoImpl::execDirect(
    const void* sendbuff,
    const size_t count,
    const commDataType_t datatype) {
  const auto nLocalRanks = comm_->statex_->nRanks();

  // Wait till async init is done, so that we can schedule copy operations with
  // the remote address
  if (nLocalRanks > 1) {
    FB_COMMCHECK(waitInit());
  }

  return execDirectCore(
      sendbuff, count, datatype, buffer, resource_, comm_, stream_);
}
} // namespace ctran::allgatherp
