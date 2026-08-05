// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>
#include <vector>

#include <folly/ScopeGuard.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/StreamedRd/Common.h"
#include "comms/ctran/algos/AllGather/StreamedRd/Plan.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/CommUtils.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/IPersistPlan.h"
#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/utils/DevUtils.cuh"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/ctran/utils/MathUtils.h"

using ctran::algos::PersistPlanKey;
using ctran::allgather::ctsrd::createPersistPlan;
using ctran::allgather::ctsrd::PersistPlan;
using ctran::allgather::ctsrd::Plan;
using ctran::allgather::ctsrd::common::exchangeCtrl;
using ctran::allgather::ctsrd::common::progressSteps;
using ctran::allgather::ctsrd::common::resolveFwdPeers;
using ctran::allgather::ctsrd::common::waitCtrl;
using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::PersistArgs;
using ctran::allgatherp::Resource;

namespace {
const auto myAlgo = NCCL_ALLGATHER_P_ALGO::ctsrdpipeline;
using CtsrdAlgoContext = ctran::allgather::ctsrd::AlgoContext;

struct AlgoContext : CtsrdAlgoContext {
  using Base = CtsrdAlgoContext;

  Resource* resource;
  const int localRank;
  const int nLocalRanks;

  inline AlgoContext(
      CtranMapper* mapper,
      Resource* resource,
      PersistArgs* pArgs,
      size_t sendSize,
      int localRank,
      int nLocalRanks,
      int nNodes,
      const Plan& recvPlan,
      const Plan& sendPlan)
      : Base(
            mapper,
            pArgs->recvbuff,
            pArgs->recvHdl,
            sendSize,
            nNodes,
            recvPlan,
            sendPlan),
        resource(resource),
        localRank(localRank),
        nLocalRanks(nLocalRanks) {}

  inline int peer(int step) const {
    return recvPlan.peer(step) * nLocalRanks + localRank;
  }

  inline commResult_t onStepComplete(int step) {
    if (nLocalRanks > 1) {
      // pipeSync releases NVL readers for this step's received chunks.
      FB_COMMCHECK(
          ctran::allgather::ctsrd::common::waitStepFlushes(*this, step));
      resource->pipeSync->post(step);
    }
    return commSuccess;
  }

  inline size_t chunkByteOffset(int node) const {
    return (static_cast<size_t>(node) * nLocalRanks + localRank) * sendSize;
  }

  inline void
  enqueuePut(int step, int node, CtranMapperRequest* flushReq = nullptr) {
    const auto nth = putCount.at(step)++;
    const auto expectedNode = sendPlan.chunk(step, nth);
    FB_CHECKABORT(
        expectedNode == node,
        "ctsrdpipeline enqueuePut: step {} put #{} expected node {} got {}",
        step,
        nth,
        expectedNode,
        node);
    putQ.push_back({step, node, flushReq});
  }
};

commResult_t gpeFn(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto* resource = reinterpret_cast<Resource*>(op->allgatherP.algoResource);
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->allgatherP.pArgs);
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
  auto mapper = comm->ctran_->mapper.get();

  CtranAlgoLogger logger(AlgoImpl::algoName(myAlgo), op->opCount, comm);

  ctran::Profiler* profiler = comm->ctran_->profiler.get();
  if (profiler) {
    profiler->initForEachColl(
        op->opCount, NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT);
  }
  auto profileGuard = folly::makeGuard([&]() {
    CTRAN_PROFILER_IF(profiler, { profiler->reportToScuba(); });
    mapper->reportProfiling();
  });

  CTRAN_PROFILER_IF(profiler, {
    auto& algoContext = profiler->algoContext;
    algoContext.algorithmName = AlgoImpl::algoName(myAlgo);
    algoContext.sendContext.messageSizes = std::to_string(sendSize);
    algoContext.recvContext.messageSizes = std::to_string(sendSize * nRanks);
  });

  CtranMapperContext mapperContext(
      AlgoImpl::algoName(myAlgo), sendSize, sendSize * nRanks);
  mapper->setContext(std::move(mapperContext));

  auto* persistPlan = static_cast<const PersistPlan*>(
      comm->ctran_->algo->getOrCreatePersistPlan(
          PersistPlanKey::kAllgatherPCtsrd, [&]() {
            return std::make_unique<PersistPlan>(createPersistPlan(
                nodeId, nNodes, resolveFwdPeers(ctran::utils::log2i(nNodes))));
          }));
  const auto& recvPlan = persistPlan->recvPlan();
  const auto& sendPlan = persistPlan->sendPlan();
  if (recvPlan.nSteps() == 0) {
    return commSuccess;
  }

  AlgoContext ctx(
      mapper,
      resource,
      pArgs,
      sendSize,
      localRank,
      nLocalRanks,
      nNodes,
      recvPlan,
      sendPlan);

  resetPipeEnd(*resource, comm);

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL));
  FB_COMMCHECK(exchangeCtrl(ctx));
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL));

  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA));
  FB_COMMCHECK(progressSteps(ctx, nodeId));
  waitPipeEnd(*resource, comm);
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA));

  FB_COMMCHECK(waitCtrl(ctx));

  return commSuccess;
}
} // namespace

namespace ctran::allgatherp {
extern __global__ void ncclKernelAllGatherPSrdPipeStart(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState);
extern __global__ void ncclKernelAllGatherPSrdPipeSync(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState,
    PipeSyncKernArgs args);
extern __global__ void ncclKernelAllGatherPSrdPipeEnd(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState,
    PipeEndKernArgs args);
extern __global__ void ncclKernelAllGatherPStreamedRd(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState);

commResult_t AlgoImpl::execStreamedRecursiveDoubling(
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
        "AllGatherP ctsrdpipeline requires nRanks ({}) to be evenly divisible by "
        "nLocalRanks ({}), nNodes={}",
        nRanks,
        nLocalRanks,
        nNodes);
  }
  if (nNodes > 1 && !ctran::utils::isPowerOfTwo(nNodes)) {
    FB_ERRORRETURN(
        commInvalidUsage,
        "AllGatherP ctsrdpipeline requires nNodes ({}) to be a power of 2",
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

  auto* persistPlan =
      static_cast<const PersistPlan*>(ctran->algo->getOrCreatePersistPlan(
          PersistPlanKey::kAllgatherPCtsrd, [&]() {
            return std::make_unique<PersistPlan>(createPersistPlan(
                myNode, nNodes, resolveFwdPeers(ctran::utils::log2i(nNodes))));
          }));
  const auto& recvPlan = persistPlan->recvPlan();

  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP,
      stream_,
      AlgoImpl::algoName(myAlgo),
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = ctran->algo->getDevState();

  // In-kernel colltrace grouping across the Srd multi-kernel collective
  // (SrdPipeStart..SrdPipeSync..SrdPipeEnd): the begin kernel emits the start
  // boundary and opens the group; the end kernel emits the end. Set explicitly
  // per submit because the reused KernelConfig defaults its emit flags to true.
  bool colltraceGroupOpen = false;

  // The streamed GPE path sources every outgoing chunk from recvbuff, including
  // the local rank's own chunk. Keep the copy stream-ordered before PipeStart.
  FB_COMMCHECK(copyToSelf(
      comm_,
      sendbuff,
      getPtr(pArgs.recvbuff, comm_->statex_->rank() * sendSize),
      sendSize,
      stream_));

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
      // Multi-kernel begin: emit the start boundary and open the group; the
      // SrdPipeEnd kernel below reuses this record and emits the end.
      config.colltraceEmitStart = true;
      config.colltraceEmitEnd = false;
      colltraceGroupOpen = true;
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPSrdPipeStart)));
    } else {
      // Single-kernel collective: emit both boundaries explicitly rather than
      // relying on the reused config's KernelConfig defaults.
      config.colltraceEmitStart = true;
      config.colltraceEmitEnd = true;
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPStreamedRd)));
    }
  }

  if (nLocalRanks > 1) {
    // The own-chunk barrier (cross-iteration WAR guard) is folded into
    // ncclKernelAllGatherPSrdPipeStart, saving a separate ncclKernelNvlBarrier
    // launch. PipeStart is only submitted when nNodes > 1, so the single-node
    // case must still emit the standalone barrier here: exactly one of the two
    // must emit it.
    FB_COMMCHECK(nvlCeBcast(
        comm_,
        sendbuff,
        sendSize,
        myRank * sendSize,
        pArgs.remoteRecvBuffs,
        pArgs.remoteAccessKeys,
        stream_,
        /*barrier=*/nNodes == 1,
        pArgs.mcWrite));

    for (int step = 0; step < recvPlan.nSteps(); step++) {
      PipeSyncKernArgs syncArgs = {
          .stepId = step,
          .pipeSync = resource_.pipeSync,
      };
      config.algoArgs = reinterpret_cast<void*>(&syncArgs);
      // Interior kernel: emits neither boundary and reuses the begin kernel's
      // record. Overwrite the values leaked from the reused `config`.
      config.colltraceEmitStart = false;
      config.colltraceEmitEnd = false;
      FB_COMMCHECK(ctran->gpe->submit(
          {},
          nullptr,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPSrdPipeSync)));

      int chunkIndex = 0;
      for (const auto node : recvPlan.chunks(step)) {
        const auto offset =
            (static_cast<size_t>(node) * nLocalRanks + localRank) * sendSize;
        auto srcPtr = ctran::allgatherp::getPtr(pArgs.recvbuff, offset);
        // The per-step barrier fires once per step (first chunk) and only paces
        // the N-1 unicast writes' incast, so it is always kept on that path.
        // The multicast write is a single switch-fanned store with no incast to
        // pace, so NCCL_CTRAN_AGP_SKIP_MC_INTRA_BARRIER can drop it there as a
        // perf A/B. Correctness holds either way via the pre-loop own-chunk
        // barrier (cross-iteration WAR) + PipeEnd, with disjoint per-rank rail
        // columns.
        const bool skipBarrier =
            pArgs.mcWrite && NCCL_CTRAN_AGP_SKIP_MC_INTRA_BARRIER;
        FB_COMMCHECK(nvlCeBcast(
            comm_,
            srcPtr,
            sendSize,
            offset,
            pArgs.remoteRecvBuffs,
            pArgs.remoteAccessKeys,
            stream_,
            chunkIndex++ == 0 && !skipBarrier,
            pArgs.mcWrite));
      }
    }

    PipeEndKernArgs endArgs = {
        .pipeSync = resource_.pipeSync,
    };
    config.algoArgs = reinterpret_cast<void*>(&endArgs);
    if (colltraceGroupOpen) {
      // Multi-kernel end: close the group opened by SrdPipeStart and emit only
      // the end boundary.
      config.colltraceEmitStart = false;
      config.colltraceEmitEnd = true;
    } else {
      // No inter-node begin ran (nNodes == 1), so this end kernel represents
      // the whole intra-node collective: emit both boundaries.
      config.colltraceEmitStart = true;
      config.colltraceEmitEnd = true;
    }
    FB_COMMCHECK(ctran->gpe->submit(
        {},
        nullptr,
        config,
        reinterpret_cast<void*>(ncclKernelAllGatherPSrdPipeEnd)));
  }

  return commSuccess;
}
} // namespace ctran::allgatherp
