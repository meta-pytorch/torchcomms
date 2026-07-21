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
using ctran::allgather::ctsrd::common::resolveFwdPeers;
using ctran::allgather::ctsrd::common::runExchangeAndProgress;
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

  FB_COMMCHECK(runExchangeAndProgress(ctx, nodeId, profiler));

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
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPSrdPipeStart)));
    } else {
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          gpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherPStreamedRd)));
    }
  }

  if (nLocalRanks > 1) {
    FB_COMMCHECK(nvlCeBcast(
        comm_,
        sendbuff,
        sendSize,
        myRank * sendSize,
        pArgs.remoteRecvBuffs,
        pArgs.remoteAccessKeys,
        stream_));

    for (int step = 0; step < recvPlan.nSteps(); step++) {
      PipeSyncKernArgs syncArgs = {
          .stepId = step,
          .pipeSync = resource_.pipeSync,
      };
      config.algoArgs = reinterpret_cast<void*>(&syncArgs);
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
        FB_COMMCHECK(nvlCeBcast(
            comm_,
            srcPtr,
            sendSize,
            offset,
            pArgs.remoteRecvBuffs,
            pArgs.remoteAccessKeys,
            stream_,
            chunkIndex++ == 0));
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
        reinterpret_cast<void*>(ncclKernelAllGatherPSrdPipeEnd)));
  }

  return commSuccess;
}
} // namespace ctran::allgatherp
