// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <iostream>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/CommUtils.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/ExtUtils.h"

using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::PersistArgs;
using ctran::allgatherp::Resource;
using ncclx::CommStateX;
namespace {
const auto myAlgo = NCCL_ALLGATHER_P_ALGO::ctpipeline;

// Get the index of the chunk in recvBuff to receive from the internode Ring
// neighbor in the rail. E.g., for nRanks = 8, nLocalRanks = 2, rank = 2, it
// would receive chunkIdx 0, 6, 4 of the recvBuff in a 3-step Ring.
inline size_t
getRecvChunkIdxInRail(int rank, int step, int nLocalRanks, int nRanks) {
  return (rank - step * nLocalRanks + nRanks) & (nRanks - 1);
}

commResult_t gpeFn(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto* resource = reinterpret_cast<Resource*>(op->allgatherP.algoResource);
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->allgatherP.pArgs);
  const void* sendBuff = op->allgatherP.sendbuff;
  const auto sendSize =
      op->allgatherP.count * commTypeSize(op->allgatherP.datatype);
  CtranComm* comm = opGroup.front()->comm_;

  const auto statex = comm->statex_.get();
  const auto rank = statex->rank();
  const auto nRanks = statex->nRanks();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();

  CtranAlgoLogger logger(AlgoImpl::algoName(myAlgo), op->opCount, comm);

  // Receive data from upPeer, and put to downPeer
  const int downPeer = (nRanks + rank + nLocalRanks) % nRanks;
  const int upPeer = (nRanks + rank - nLocalRanks) % nRanks;

  auto mapper = comm->ctran_->mapper.get();

  void* sendHdl = nullptr;
  bool localReg;
  FB_COMMCHECK(
      mapper->searchRegHandle(sendBuff, sendSize, &sendHdl, &localReg));
  auto guard = folly::makeGuard([sendHdl, localReg, mapper]() {
    if (localReg) {
      FB_COMMCHECKIGNORE(mapper->deregDynamic(sendHdl));
    }
  });

  CtranMapperRequest syncSreq, syncRreq;

  // Sync to notify upPeer that I am ready to receive data
  FB_COMMCHECK(mapper->isendCtrl(upPeer, &syncSreq));
  FB_COMMCHECK(mapper->irecvCtrl(downPeer, &syncRreq));

  FB_COMMCHECK(mapper->waitRequest(&syncRreq));

  // Initialize notify flag to receive from upstream peer
  auto notify = std::make_unique<CtranMapperNotify>();
  FB_COMMCHECK(mapper->initNotify(upPeer, pArgs->recvHdl, notify.get()));

  std::vector<CtranMapperRequest> putReqs(nNodes - 1);
  for (auto step = 0; step < nNodes - 1; step++) {
    const auto offset =
        getRecvChunkIdxInRail(rank, step, nLocalRanks, nRanks) * sendSize;

    // First step transfers local chunk, and remaining steps transfer from
    // previously received chunk.
    auto sendPtr = step == 0
        ? sendBuff
        : ctran::allgatherp::getPtr(pArgs->recvbuff, offset);
    auto sendHdl_ = step == 0 ? sendHdl : pArgs->recvHdl;

    // Issue put to IB peers
    FB_COMMCHECK(mapper->iput(
        sendPtr,
        ctran::allgatherp::getPtr(pArgs->remoteRecvBuffs[downPeer], offset),
        sendSize,
        downPeer,
        CtranMapperConfig{
            .memHdl_ = sendHdl_,
            .remoteAccessKey_ = pArgs->remoteAccessKeys[downPeer],
            .notify_ = true},
        &putReqs.at(step)));

    // Wait till received data from upstream peer
    FB_COMMCHECK(mapper->waitNotify(notify.get()));

    // Kick off local broadcast of the received data
    resource->pipeSync->post(step);
  }

  // Wait all local PUTs to complete before returning.
  for (auto& putReq : putReqs) {
    FB_COMMCHECK(mapper->waitRequest(&putReq));
  }

  // Wait till the isendCtrl has completed so we don't have leak
  FB_COMMCHECK(mapper->waitRequest(&syncSreq));

  return commSuccess;
}
} // namespace

namespace ctran::allgatherp {
extern __global__ void ncclKernelAllGatherPPipeStart(
    int* flag,
    CtranAlgoDeviceState* devState);
extern __global__ void ncclKernelAllGatherPPipeSync(
    int* flag,
    CtranAlgoDeviceState* devState,
    PipeSyncKernArgs args);
extern __global__ void ncclKernelAllGatherPPipeEnd(
    int* flag,
    CtranAlgoDeviceState* devState,
    PipeEndKernArgs args);
extern __global__ void ncclKernelAllGatherPPipe(
    int* flag,
    CtranAlgoDeviceState* devState);

commResult_t AlgoImpl::execPipeline(
    const void* sendbuff,
    const size_t count,
    const commDataType_t datatype) {
  auto recvbuff = pArgs.recvbuff;
  auto ctran = comm_->ctran_.get();
  const auto statex = comm_->statex_.get();
  const auto opCount = ctran->getOpCount();
  const auto sendSize = count * commTypeSize(datatype);

  const auto nRanks = statex->nRanks();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto myRank = statex->rank();
  const auto nNodes = statex->nNodes();

  CTRAN_COLL_INFO(
      AlgoImpl::algoName(myAlgo),
      sendbuff,
      recvbuff,
      count,
      datatype,
      -1,
      comm_,
      stream_);

  // Wait till async init is done, so that we can schedule copy operations with
  // the remote address
  if (nRanks > 1) {
    FB_COMMCHECK(waitInit());
  }

  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP,
      stream_,
      AlgoImpl::algoName(myAlgo),
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = ctran->algo->getDevState();

  // TODO: ensure colltrace can capture the group of operations as single
  // allgather

  if (nNodes > 1) {
    // Submit inter-node Ring pipeline for GPE thread to execute. Skip if single
    // node.
    auto op = std::make_unique<OpElem>(
        OpElem::opType::ALLGATHERP, stream_, comm_, opCount);
    op->allgatherP.pArgs = &pArgs;
    op->allgatherP.algoResource = &resource_;
    op->allgatherP.sendbuff = sendbuff;
    op->allgatherP.count = count;
    op->allgatherP.datatype = datatype;

    std::vector<std::unique_ptr<struct OpElem>> opGroup;
    opGroup.push_back(std::move(op));

    if (nLocalRanks > 1) {
      // - For nLocalRanks > 1 case, use ncclKernelAllGatherPPipeStart to hold
      //   GPE thread till allgather starts. ncclKernelAllGatherPStart returns
      //   immediately after started GPE, thus the inter-node pipeline can be
      //   overlapped with the following intra-node copies.
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPPipeStart)));
    } else {
      // - For nLocalRanks == 1 case, ncclKernelAllGatherPPipe holds the stream
      //   till GPE thread finishes entire transfer.
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPPipe)));
    }
  }

  // Copy data to self for out-of-place allgather
  FB_COMMCHECK(copyToSelf(comm_, sendbuff, sendSize, pArgs, stream_));

  // Submit intra-node copies in the pipeline
  if (nLocalRanks > 1) {
    // - Step 0: Broadcast local chunk to intra-node peers
    // Copy data to other local ranks
    FB_COMMCHECK(nvlCeBcast(
        comm_,
        sendbuff,
        sendSize,
        myRank * sendSize,
        pArgs,
        stream_,
        ceStreams_));

    const int upPeer = (nRanks + myRank - nLocalRanks) & (nRanks - 1);

    // -  Remaining steps: broadcast received chunk from internode upPeer
    for (int step = 0; step < nNodes - 1; step++) {
      // - ncclKernelAllGatherPPipeSync waits till the GPE thread fnished
      // step-n exchange and has posted via the shared pipeSync flag.
      PipeSyncKernArgs kernArgs = {
          .stepId = step,
          .pipeSync = resource_.pipeSync,
      };
      config.algoArgs = reinterpret_cast<void*>(&kernArgs);
      FB_COMMCHECK(ctran->gpe->submit(
          {},
          nullptr,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPPipeSync)));

      // - Intra-node forwarding chunk received at step-n from upPeer, broadcast
      //  to the same offset on other local ranks
      const auto offset =
          getRecvChunkIdxInRail(upPeer, step, nLocalRanks, nRanks) * sendSize;
      const auto sendPtr = getPtr(pArgs.recvbuff, offset);
      FB_COMMCHECK(nvlCeBcast(
          comm_, sendPtr, sendSize, offset, pArgs, stream_, ceStreams_));
    }

    PipeEndKernArgs kernArgs = {
        // Pass pipeSync to reset the flag before starting the next pipeline
        .pipeSync = resource_.pipeSync,
    };
    config.algoArgs = reinterpret_cast<void*>(&kernArgs);
    FB_COMMCHECK(ctran->gpe->submit(
        {},
        nullptr,
        config,
        reinterpret_cast<void*>(ncclKernelAllGatherPPipeEnd)));
  }

  return commSuccess;
}

} // namespace ctran::allgatherp
