// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/ScopeGuard.h>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/CommUtils.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/utils/DevUtils.cuh"
#include "comms/ctran/utils/ExtUtils.h"

using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::PersistArgs;
using ctran::allgatherp::Resource;

namespace {
const auto myAlgo = NCCL_ALLGATHER_P_ALGO::ctpat;

inline int distNodesAtStep(int nNodes, int step) {
  return nNodes >> (step + 1);
}

inline int
peerAtStep(int nodeId, int localRank, int nLocalRanks, int nNodes, int step) {
  const int dist = distNodesAtStep(nNodes, step);
  const int pos = (nodeId / dist) % 2;
  const int peerNode = pos == 0 ? nodeId + dist : nodeId - dist;
  return peerNode * nLocalRanks + localRank;
}

inline int nodeChunkStrideAtStep(int nNodes, int step) {
  return nNodes >> step;
}

inline size_t rankChunkOffset(
    int anchorNode,
    int localRank,
    int nLocalRanks,
    int nNodes,
    int step,
    int j) {
  const int stride = nodeChunkStrideAtStep(nNodes, step);
  const int nodePos = j * stride + (anchorNode % stride);
  return static_cast<size_t>(nodePos) * nLocalRanks + localRank;
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
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();
  const auto nodeId = rank / nLocalRanks;
  const auto nSteps = static_cast<int>(ctran::utils::log2i(nNodes));

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

  auto mapper = comm->ctran_->mapper.get();

  void* sendHdl = nullptr;
  bool localReg = false;
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));
  FB_COMMCHECK(
      mapper->searchRegHandle(sendBuff, sendSize, &sendHdl, &localReg));
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));
  auto regGuard = folly::makeGuard([sendHdl, localReg, mapper]() {
    if (localReg) {
      FB_COMMCHECKIGNORE(mapper->deregDynamic(sendHdl));
    }
  });

  std::vector<int> peers(nSteps);
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec(nSteps);

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL));

  for (int i = 0; i < nSteps; i++) {
    peers[i] = peerAtStep(nodeId, localRank, nLocalRanks, nNodes, i);
    notifyVec[i] = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(
        mapper->initNotify(peers[i], pArgs->recvHdl, notifyVec[i].get()));
  }

  std::vector<CtranMapperRequest> syncSreqs(nSteps);
  std::vector<CtranMapperRequest> syncRreqs(nSteps);
  for (int i = 0; i < nSteps; i++) {
    FB_COMMCHECK(mapper->isendCtrl(peers[i], &syncSreqs[i]));
    FB_COMMCHECK(mapper->irecvCtrl(peers[i], &syncRreqs[i]));
  }
  for (int i = 0; i < nSteps; i++) {
    FB_COMMCHECK(mapper->waitRequest(&syncRreqs[i]));
  }

  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL));

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::make_unique<CtranMapperTimestamp>(AlgoImpl::algoName(myAlgo));

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA));

  std::vector<CtranMapperRequest> lastPutReqs(nSteps);

  for (int i = 0; i < nSteps; i++) {
    const int peer = peers[i];
    const int nPuts = 1 << i;
    timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));

    for (int j = 0; j < nPuts; j++) {
      const auto chunkIdx =
          rankChunkOffset(nodeId, localRank, nLocalRanks, nNodes, i, j);
      const size_t byteOffset = chunkIdx * sendSize;

      const void* srcPtr = nullptr;
      void* srcHdl = nullptr;
      if (i == 0) {
        srcPtr = sendBuff;
        srcHdl = sendHdl;
      } else {
        srcPtr = ctran::allgatherp::getPtr(pArgs->recvbuff, byteOffset);
        srcHdl = pArgs->recvHdl;
      }
      void* dstPtr =
          ctran::allgatherp::getPtr(pArgs->remoteRecvBuffs[peer], byteOffset);

      const bool isLast = (j == nPuts - 1);
      FB_COMMCHECK(mapper->iput(
          srcPtr,
          dstPtr,
          sendSize,
          peer,
          CtranMapperConfig{
              .memHdl_ = srcHdl,
              .remoteAccessKey_ = pArgs->remoteAccessKeys[peer],
              .notify_ = isLast},
          isLast ? &lastPutReqs[i]
                 : static_cast<CtranMapperRequest*>(nullptr)));
    }

    FB_COMMCHECK(mapper->waitRequest(&lastPutReqs[i]));
    timestamp->putComplete.push_back(CtranMapperTimestampPoint(peer));

    FB_COMMCHECK(mapper->waitNotify(notifyVec[i].get()));

    if (nLocalRanks > 1) {
      resource->pipeSync->post(i);
    }
  }

  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  for (int i = 0; i < nSteps; i++) {
    FB_COMMCHECK(mapper->waitRequest(&syncSreqs[i]));
  }

  mapper->timestamps.push_back(std::move(timestamp));
  mapper->reportProfiling();

  CTRAN_PROFILER_IF(profiler, { profiler->reportToScuba(); });

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

commResult_t AlgoImpl::execPat(
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
  const auto localRank = statex->localRank();
  const auto nNodes = statex->nNodes();
  const auto myNode = myRank / nLocalRanks;

  if (nNodes <= 1) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "AllGatherP ctpat requires nNodes > 1, got {}",
        nNodes);
  }
  if ((nNodes & (nNodes - 1)) != 0) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "AllGatherP ctpat requires nNodes ({}) to be a power of 2",
        nNodes);
  }
  if (nLocalRanks > 1 && nRanks % nLocalRanks != 0) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "AllGatherP ctpat requires nRanks ({}) to be evenly divisible by "
        "nLocalRanks ({}), nNodes={}",
        nRanks,
        nLocalRanks,
        nNodes);
  }
  if (nLocalRanks > 1) {
    auto mapper = comm_->ctran_->mapper.get();
    for (int lr = 0; lr < nLocalRanks; lr++) {
      const int peer = statex->localRankToRank(lr);
      if (peer != myRank &&
          mapper->getBackend(peer) != CtranMapperBackend::NVL) {
        FB_ERRORRETURN(
            commInvalidUsage,
            "AllGatherP ctpat requires NVL backend for all local peers, "
            "peer rank {} has non-NVL backend",
            peer);
      }
    }
  }

  CTRAN_COLL_INFO(
      AlgoImpl::algoName(myAlgo),
      sendbuff,
      recvbuff,
      count,
      datatype,
      -1,
      comm_,
      stream_);

  if (nLocalRanks > 1) {
    FB_COMMCHECK(waitInit());
  }

  const int nSteps = static_cast<int>(ctran::utils::log2i(nNodes));

  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP,
      stream_,
      AlgoImpl::algoName(myAlgo),
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = ctran->algo->getDevState();

  FB_COMMCHECK(copyToSelf(comm_, sendbuff, sendSize, pArgs, stream_));

  if (nNodes > 1) {
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
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPPipeStart)));
    } else {
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPPipe)));
    }
  }

  if (nLocalRanks > 1) {
    FB_COMMCHECK(nvlCeBcast(
        comm_, sendbuff, sendSize, myRank * sendSize, pArgs, stream_));

    for (int i = 0; i < nSteps; i++) {
      PipeSyncKernArgs syncArgs = {
          .stepId = i,
          .pipeSync = resource_.pipeSync,
      };
      config.algoArgs = reinterpret_cast<void*>(&syncArgs);
      FB_COMMCHECK(ctran->gpe->submit(
          {},
          nullptr,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPPipeSync)));

      const int distNodes = distNodesAtStep(nNodes, i);
      const int pos = (myNode / distNodes) % 2;
      const int peerNode = pos == 0 ? myNode + distNodes : myNode - distNodes;

      const int stride = nodeChunkStrideAtStep(nNodes, i);
      const int nodeOffset = peerNode % stride;
      const int nChunks = 1 << i;

      for (int j = 0; j < nChunks; j++) {
        const int nodePos = j * stride + nodeOffset;
        const size_t chunkIdx = static_cast<size_t>(nodePos) * nLocalRanks +
            static_cast<size_t>(localRank);
        const size_t byteOffset = chunkIdx * sendSize;
        auto srcPtr = ctran::allgatherp::getPtr(pArgs.recvbuff, byteOffset);
        const bool needBarrier = (j == 0);
        FB_COMMCHECK(nvlCeBcast(
            comm_, srcPtr, sendSize, byteOffset, pArgs, stream_, needBarrier));
      }
    }

    PipeEndKernArgs endArgs = {
        .pipeSync = resource_.pipeSync,
    };
    config.algoArgs = reinterpret_cast<void*>(&endArgs);
    FB_COMMCHECK(ctran->gpe->submit(
        {},
        nullptr,
        config,
        reinterpret_cast<void*>(ncclKernelAllGatherPPipeEnd)));
  }

  return commSuccess;
}
} // namespace ctran::allgatherp
