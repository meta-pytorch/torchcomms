// Copyright (c) Meta Platforms, Inc. and affiliates.

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
const auto myAlgo = NCCL_ALLGATHER_P_ALGO::ctrdpipeline;

// Node-level recursive doubling distance at step `i` (i=0 is the largest
// distance). Expressed in units of nodes.
inline int distNodesAtStep(int nNodes, int step) {
  // dist = nNodes / 2^(step+1)
  return nNodes >> (step + 1);
}

// Rail-peer rank for a given step using the same pattern as ctrdpipeline
// allgather. Operates on nodes so the peer is always on a different node in the
// same rail.
inline int
peerAtStep(int nodeId, int localRank, int nLocalRanks, int nNodes, int step) {
  const int dist = distNodesAtStep(nNodes, step);
  const int pos = (nodeId / dist) % 2;
  const int peerNode = pos == 0 ? nodeId + dist : nodeId - dist;
  return peerNode * nLocalRanks + localRank;
}

// Stride between node-chunks owned/sent at step `i`. After step `i`, each node
// owns 2^(i+1) node-chunks with node indices of the form
// j * stride + (nodeId % stride) for j in [0, 2^(i+1)). At the start of step
// i, stride = nNodes / 2^i and 2^i chunks are owned.
inline int nodeChunkStrideAtStep(int nNodes, int step) {
  return nNodes >> step;
}

// For a rank on `anchorNode`, return the chunk offset (in units of rank
// chunks) that local rank `lr` uses for its j-th PUT at step i. anchorNode
// is the node whose chunks are being striped (my node on send, peer node on
// recv position lookup).
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

  // Compute rail peer per step and initialize notify flags up-front so a
  // peer PUT that arrives fast can be tracked immediately.
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

  // Exchange a ready-to-receive handshake with every rail peer before
  // launching the first PUT. This mirrors the guarantee ctpipeline gets from
  // its single sendCtrl/recvCtrl pair so that RDMA writes don't race the
  // receiver's buffer registration becoming remote-addressable.
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

  // Last-put request per step, used to wait for local completion.
  std::vector<CtranMapperRequest> lastPutReqs(nSteps);

  for (int i = 0; i < nSteps; i++) {
    const int peer = peers[i];
    const int nPuts = 1 << i;
    timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));

    for (int j = 0; j < nPuts; j++) {
      // Send chunk sourced from the `anchorNode = nodeId` stripe at column
      // `localRank`. At step 0 (chunkIdx == rank) the data is taken from
      // sendBuff. At later steps the same column slot is read from recvbuff:
      // slot `rank` was written by copyToSelf (stream-ordered before the
      // PipeStart kernel that released this GPE thread, so it has landed);
      // other slots were written by earlier IB receives on this same GPE
      // thread.
      const auto chunkIdx =
          rankChunkOffset(nodeId, localRank, nLocalRanks, nNodes, i, j);
      const size_t byteOffset = chunkIdx * sendSize;

      const void* srcPtr = nullptr;
      void* srcHdl = nullptr;
      if (i == 0) {
        // j == 0 only (nPuts == 1). chunkIdx == rank.
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

    // Notify the stream that step `i` IB exchange is done. The stream can
    // now issue the intra-node CE broadcast for the 2^i chunks just
    // received at column `localRank`.
    if (nLocalRanks > 1) {
      resource->pipeSync->post(i);
    }
  }

  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  // Wait for our isendCtrls to complete so we don't leak requests.
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

commResult_t AlgoImpl::execRecursiveDoubling(
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

  if (nLocalRanks > 1 && nRanks % nLocalRanks != 0) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "AllGatherP ctrdpipeline requires nRanks ({}) to be evenly divisible by "
        "nLocalRanks ({}), nNodes={}",
        nRanks,
        nLocalRanks,
        nNodes);
  }
  if (nNodes > 1 && (nNodes & (nNodes - 1)) != 0) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "AllGatherP ctrdpipeline requires nNodes ({}) to be a power of 2",
        nNodes);
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

  // copyToSelf must be enqueued BEFORE the PipeStart kernel that releases the
  // GPE thread. Unlike ctpipeline, whose GPE reads sendBuff at step 0 and
  // recvbuff only at positions filled by prior IB receives, the
  // recursive-doubling GPE reads recvbuff[rank*sendSize] at exactly one j per
  // step (j = nodeId / stride_i) — that slot is populated only by copyToSelf
  // on the stream. Enqueuing copyToSelf first gives stream ordering:
  // copyToSelf -> PipeStart kernel -> GPE signal, so GPE cannot read that
  // slot before the CE copy has landed.
  FB_COMMCHECK(copyToSelf(comm_, sendbuff, sendSize, pArgs, stream_));

  // Submit inter-node recursive-doubling exchange for the GPE thread to
  // execute. Skip entirely if we are on a single node.
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
      // ncclKernelAllGatherPPipeStart releases the GPE thread and returns
      // immediately so intra-node CE copies can overlap with inter-node
      // exchange.
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPPipeStart)));
    } else {
      // No intra-node copies — hold the stream while GPE runs to completion.
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPPipe)));
    }
  }

  if (nLocalRanks > 1) {
    // Initial intra-node CE broadcast of every rank's own chunk. After this
    // every local rank has all L chunks for its own node populated at
    // recvbuff[m*L .. m*L + L) where m = myNode.
    FB_COMMCHECK(nvlCeBcast(
        comm_, sendbuff, sendSize, myRank * sendSize, pArgs, stream_));

    for (int i = 0; i < nSteps; i++) {
      // Wait for the GPE's step-i IB exchange to complete.
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

      // The just-received chunks at step i live at column `localRank` for
      // the 2^i node positions belonging to the peer node at step i.
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
        // Only the first bcast in each step issues a local barrier; the
        // subsequent bcasts within the same step touch disjoint positions so
        // are safe to fire back-to-back.
        const bool needBarrier = (j == 0);
        FB_COMMCHECK(nvlCeBcast(
            comm_, srcPtr, sendSize, byteOffset, pArgs, stream_, needBarrier));
      }
    }

    // Reset the pipeSync flags before the next call to execRecursiveDoubling.
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
