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
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::PersistArgs;
using ctran::allgatherp::Resource;
using ctran::allgatherp::StagingInfo;

namespace {
const auto myAlgo = NCCL_ALLGATHER_P_ALGO::ctpatcopy;

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
  auto& staging = resource->stagingInfo;

  // If gpeFn fails, set async error for host-side detection.
  // Note: if the failure occurs before pipeSync->post(step), the stream-side
  // PipeSync kernel will hang waiting for a post that never comes. This is
  // an inherited limitation of the PipeSync model, shared with ctrdpipeline
  // and ctpipeline. It is not specific to ctpatcopy. A clean fix requires
  // a cancellable device-side wait, which the current PipeSync kernel does
  // not support (it only observes postFlag, not async error or abort when
  // launched with flag=nullptr).
  commResult_t gpeFnResult = commSuccess;
  auto asyncErrGuard = folly::makeGuard([&gpeFnResult, comm]() {
    if (gpeFnResult != commSuccess) {
      comm->setAsyncException(
          ctran::utils::Exception("ctpatcopy gpeFn failure", gpeFnResult));
    }
  });

#define GPEFN_COMMCHECK(call)  \
  do {                         \
    commResult_t _rc = (call); \
    if (_rc != commSuccess) {  \
      gpeFnResult = _rc;       \
      return _rc;              \
    }                          \
  } while (0)

#define GPEFN_CUDACHECK(call)          \
  do {                                 \
    cudaError_t _err = (call);         \
    if (_err != cudaSuccess) {         \
      gpeFnResult = commInternalError; \
      return commInternalError;        \
    }                                  \
  } while (0)

  std::vector<int> peers(nSteps);
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec(nSteps);

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL));

  for (int i = 0; i < nSteps; i++) {
    peers[i] = peerAtStep(nodeId, localRank, nLocalRanks, nNodes, i);
    notifyVec[i] = std::make_unique<CtranMapperNotify>();
    GPEFN_COMMCHECK(mapper->initNotify(
        peers[i], staging.recvBuf.regHdl, notifyVec[i].get()));
  }

  std::vector<CtranMapperRequest> syncSreqs(nSteps);
  std::vector<CtranMapperRequest> syncRreqs(nSteps);
  for (int i = 0; i < nSteps; i++) {
    GPEFN_COMMCHECK(mapper->isendCtrl(peers[i], &syncSreqs[i]));
    GPEFN_COMMCHECK(mapper->irecvCtrl(peers[i], &syncRreqs[i]));
  }
  for (int i = 0; i < nSteps; i++) {
    GPEFN_COMMCHECK(mapper->waitRequest(&syncRreqs[i]));
  }

  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL));

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::make_unique<CtranMapperTimestamp>(AlgoImpl::algoName(myAlgo));

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA));

  cudaStream_t copyStream = nullptr;
  GPEFN_CUDACHECK(cudaStreamCreate(&copyStream));
  auto streamGuard =
      folly::makeGuard([copyStream]() { cudaStreamDestroy(copyStream); });

  for (int step = 0; step < nSteps; step++) {
    const int peer = peers[step];
    const int nPuts = 1 << step;
    timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));

    // Find the peer's staging recv buf index in our exchanged list
    int peerIdx = -1;
    for (size_t pi = 0; pi < staging.peerRanks.size(); pi++) {
      if (staging.peerRanks[pi] == peer) {
        peerIdx = static_cast<int>(pi);
        break;
      }
    }
    if (peerIdx < 0) {
      gpeFnResult = commInternalError;
      return commInternalError;
    }

    // Batch CE copy: source → tmpSendBuf for all chunks in this step
    for (int j = 0; j < nPuts; j++) {
      const auto chunkIdx =
          rankChunkOffset(nodeId, localRank, nLocalRanks, nNodes, step, j);
      const size_t byteOffset = chunkIdx * sendSize;

      const void* srcPtr = nullptr;
      if (step == 0 && j == 0) {
        srcPtr = sendBuff;
      } else {
        srcPtr = ctran::allgatherp::getPtr(pArgs->recvbuff, byteOffset);
      }

      GPEFN_COMMCHECK(mapper->icopy(
          static_cast<char*>(staging.sendBuf.ptr) + j * sendSize,
          srcPtr,
          sendSize,
          copyStream));
    }

    // Sync: ensure ALL CE copies to staging are complete before IB reads
    GPEFN_CUDACHECK(cudaStreamSynchronize(copyStream));

    // Issue all IB puts for this step (staging is now populated)
    CtranMapperRequest lastPutReq;
    for (int j = 0; j < nPuts; j++) {
      const bool isLast = (j == nPuts - 1);
      GPEFN_COMMCHECK(mapper->iput(
          static_cast<char*>(staging.sendBuf.ptr) + j * sendSize,
          static_cast<char*>(staging.remRecvBufs[peerIdx].ptr) + j * sendSize,
          sendSize,
          peer,
          CtranMapperConfig{
              .memHdl_ = staging.sendBuf.regHdl,
              .remoteAccessKey_ = staging.remRecvBufs[peerIdx].rkey,
              .notify_ = isLast},
          isLast ? &lastPutReq : static_cast<CtranMapperRequest*>(nullptr)));
    }

    // Wait for all puts + peer notification
    GPEFN_COMMCHECK(mapper->waitRequest(&lastPutReq));
    timestamp->putComplete.push_back(CtranMapperTimestampPoint(peer));
    GPEFN_COMMCHECK(mapper->waitNotify(notifyVec[step].get()));

    // Flush all received staging slots
    for (int j = 0; j < nPuts; j++) {
      CtranMapperRequest* flushReq = nullptr;
      GPEFN_COMMCHECK(mapper->iflush(
          static_cast<char*>(staging.recvBuf.ptr) + j * sendSize,
          staging.recvBuf.regHdl,
          &flushReq));
      GPEFN_COMMCHECK(mapper->waitRequest(flushReq));
    }

    // Signal stream: step complete, staging ready for CE copies
    resource->pipeSync->post(step);

    // Wait for stream to finish staging->recvbuff copies before next step.
    // Poll with both abort and async error checks to avoid infinite spin
    // if the stream fails before StepDone is enqueued.
    while (!resource->stepDoneSync->isComplete(step)) {
      if (comm->testAbort()) {
        return commInternalError;
      }
      auto asyncResult = comm->getAsyncResult();
      if (asyncResult != commSuccess) {
        return asyncResult;
      }
    }
  }

  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  for (int i = 0; i < nSteps; i++) {
    GPEFN_COMMCHECK(mapper->waitRequest(&syncSreqs[i]));
  }

  mapper->timestamps.push_back(std::move(timestamp));
  mapper->reportProfiling();

  CTRAN_PROFILER_IF(profiler, { profiler->reportToScuba(); });

  gpeFnResult = commSuccess;
  return commSuccess;
}
#undef GPEFN_COMMCHECK
#undef GPEFN_CUDACHECK
} // namespace

namespace ctran::allgatherp {
extern __global__ void ncclKernelAllGatherPPipeStart(
    int* flag,
    CtranAlgoDeviceState* devState);
extern __global__ void ncclKernelAllGatherPPipeSync(
    int* flag,
    CtranAlgoDeviceState* devState,
    PipeSyncKernArgs args);
extern __global__ void ncclKernelStepDone(
    int* flag,
    CtranAlgoDeviceState* devState,
    StepDoneKernArgs args);
extern __global__ void ncclKernelAllGatherPPatCopyPipeEnd(
    int* flag,
    CtranAlgoDeviceState* devState,
    PatCopyPipeEndKernArgs args);

commResult_t AlgoImpl::execPatCopy(
    const void* sendbuff,
    const size_t count,
    const commDataType_t datatype) {
  auto recvbuff = pArgs.recvbuff;
  auto ctran = comm_->ctran_.get();
  const auto statex = comm_->statex_.get();
  const auto opCount = ctran->getOpCount();
  const auto sendSize = count * commTypeSize(datatype);

  const auto nLocalRanks = statex->nLocalRanks();
  const auto myRank = statex->rank();
  const auto localRank = statex->localRank();
  const auto nNodes = statex->nNodes();
  const auto myNode = myRank / nLocalRanks;

  // Topology eligibility validated at init time (pinned algo).
  // Only check staging capacity here -- fall back to zero-copy if too large.
  const int maxChunksPerStep = nNodes / 2;
  if (sendSize * maxChunksPerStep >
      static_cast<size_t>(NCCL_CTRAN_ALLGATHERP_PATCOPY_STAGING_BUF_SIZE)) {
    return execPat(sendbuff, count, datatype);
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

  FB_COMMCHECK(waitInit());

  const int nSteps = static_cast<int>(ctran::utils::log2i(nNodes));

  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP,
      stream_,
      AlgoImpl::algoName(myAlgo),
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = ctran->algo->getDevState();

  // copyToSelf must be enqueued BEFORE PipeStart
  FB_COMMCHECK(copyToSelf(comm_, sendbuff, sendSize, pArgs, stream_));

  // Submit GPE with PipeStart
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

    FB_COMMCHECK(ctran->gpe->submit(
        std::move(opGroup),
        gpeFn,
        config,
        reinterpret_cast<void*>(ncclKernelAllGatherPPipeStart)));
  }

  // Initial NVL broadcast of own chunk
  FB_COMMCHECK(
      nvlCeBcast(comm_, sendbuff, sendSize, myRank * sendSize, pArgs, stream_));

  // Per-step: PipeSync -> CE staging->recvbuff -> nvlCeBcast -> StepDone
  // After GPE has posted pipeSync, any stream-side failure must publish
  // async error so the GPE's stepDoneSync poll can exit.
  auto setAsyncErrAndReturn = [this](commResult_t err) -> commResult_t {
    comm_->setAsyncException(
        ctran::utils::Exception("ctpatcopy stream-side failure", err));
    return err;
  };

  for (int step = 0; step < nSteps; step++) {
    PipeSyncKernArgs syncArgs = {
        .stepId = step,
        .pipeSync = resource_.pipeSync,
    };
    config.algoArgs = reinterpret_cast<void*>(&syncArgs);
    auto rc = ctran->gpe->submit(
        {},
        nullptr,
        config,
        reinterpret_cast<void*>(ncclKernelAllGatherPPipeSync));
    if (rc != commSuccess) {
      return setAsyncErrAndReturn(rc);
    }

    const int distNodes = distNodesAtStep(nNodes, step);
    const int pos = (myNode / distNodes) % 2;
    const int peerNode = pos == 0 ? myNode + distNodes : myNode - distNodes;
    const int nChunks = 1 << step;

    for (int j = 0; j < nChunks; j++) {
      const size_t chunkIdx =
          rankChunkOffset(peerNode, localRank, nLocalRanks, nNodes, step, j);
      const size_t byteOffset = chunkIdx * sendSize;

      auto cudaErr = cudaMemcpyAsync(
          ctran::allgatherp::getPtr(pArgs.recvbuff, byteOffset),
          static_cast<char*>(resource_.stagingInfo.recvBuf.ptr) + j * sendSize,
          sendSize,
          cudaMemcpyDefault,
          stream_);
      if (cudaErr != cudaSuccess) {
        return setAsyncErrAndReturn(commInternalError);
      }

      const bool needBarrier = (j == 0);
      rc = nvlCeBcast(
          comm_,
          ctran::allgatherp::getPtr(pArgs.recvbuff, byteOffset),
          sendSize,
          byteOffset,
          pArgs,
          stream_,
          needBarrier);
      if (rc != commSuccess) {
        return setAsyncErrAndReturn(rc);
      }
    }

    StepDoneKernArgs doneArgs = {
        .stepId = step,
        .stepDoneSync = resource_.stepDoneSync,
    };
    config.algoArgs = reinterpret_cast<void*>(&doneArgs);
    rc = ctran->gpe->submit(
        {}, nullptr, config, reinterpret_cast<void*>(ncclKernelStepDone));
    if (rc != commSuccess) {
      return setAsyncErrAndReturn(rc);
    }
  }

  // PipeEnd: reset both syncs + NVL barrier.
  // This is the only safe reset point — both GPE and stream are quiesced.
  PatCopyPipeEndKernArgs endArgs = {
      .pipeSync = resource_.pipeSync,
      .stepDoneSync = resource_.stepDoneSync,
  };
  config.algoArgs = reinterpret_cast<void*>(&endArgs);
  FB_COMMCHECK(ctran->gpe->submit(
      {},
      nullptr,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherPPatCopyPipeEnd)));

  // Final async error check: catch GPE failures that occurred between
  // PipeStart and now. Without this, execPatCopy could return commSuccess
  // for a collective where the GPE side failed.
  auto finalAsyncResult = comm_->getAsyncResult();
  if (finalAsyncResult != commSuccess) {
    return finalAsyncResult;
  }

  return commSuccess;
}
} // namespace ctran::allgatherp
