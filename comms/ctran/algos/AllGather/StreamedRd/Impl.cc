// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllGather/StreamedRd/Common.h"
#include "comms/ctran/algos/AllGather/StreamedRd/Plan.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/utils/DevUtils.cuh"

__global__ void ncclKernelAllGatherCtranStreamedRd(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState,
    ctran::allgather::KernelArgs args);

namespace {

using ctran::algos::PersistPlanKey;
using ctran::allgather::ctsrd::createPersistPlan;
using ctran::allgather::ctsrd::PersistPlan;
using ctran::allgather::ctsrd::common::resolveFwdPeers;
using ctran::allgather::ctsrd::common::runExchangeAndProgress;

const auto myAlgo = NCCL_ALLGATHER_ALGO::ctsrd;
using CtsrdAlgoContext = ctran::allgather::ctsrd::AlgoContext;

struct AlgoContext : CtsrdAlgoContext {
  using Base = CtsrdAlgoContext;
  using Base::Base;

  inline int peer(int step) const {
    return recvPlan.peer(step);
  }

  inline commResult_t onStepComplete(int /* step */) {
    return commSuccess;
  }

  inline size_t chunkByteOffset(int chunk) const {
    return static_cast<size_t>(chunk) * sendSize;
  }

  inline void enqueuePut(
      int step,
      int chunkOffset,
      CtranMapperRequest* flushReq = nullptr) {
    const auto nth = putCount.at(step)++;
    const auto expChunkOffset = sendPlan.chunk(step, nth);
    FB_CHECKABORT(
        expChunkOffset == chunkOffset,
        "enqueuePut: step {} put #{} expected chunk {} got {}",
        step,
        nth,
        expChunkOffset,
        chunkOffset);
    putQ.push_back({step, chunkOffset, flushReq});
  }
};

commResult_t impl(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  auto* op = opGroup.front().get();
  const auto sendSize =
      op->allgather.sendcount * commTypeSize(op->allgather.datatype);
  auto* comm = opGroup.front()->comm_;
  const auto& statex = comm->statex_;
  const auto rank = statex->rank();
  const auto nRanks = statex->nRanks();
  auto* mapper = comm->ctran_->mapper.get();

  CtranAlgoLogger logger(allGatherAlgoName(myAlgo), op->opCount, comm);

  auto* profiler = comm->ctran_->profiler.get();
  if (profiler) {
    profiler->initForEachColl(
        op->opCount, NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT);
  }

  CTRAN_PROFILER_IF(profiler, {
    auto& algoContext = profiler->algoContext;
    algoContext.algorithmName = allGatherAlgoName(myAlgo);
    algoContext.sendContext.messageSizes = std::to_string(sendSize);
    algoContext.recvContext.messageSizes = std::to_string(sendSize * nRanks);
  });

  auto profileGuard = folly::makeGuard([&]() {
    CTRAN_PROFILER_IF(profiler, { profiler->reportToScuba(); });
    mapper->reportProfiling();
  });

  CtranMapperContext mapperCtx(
      allGatherAlgoName(myAlgo), sendSize, sendSize * nRanks);
  mapper->setContext(std::move(mapperCtx));

  auto* persistPlan = static_cast<const PersistPlan*>(
      comm->ctran_->algo->getOrCreatePersistPlan(
          PersistPlanKey::kAllgatherCtsrd, [&]() {
            return std::make_unique<PersistPlan>(createPersistPlan(
                rank, nRanks, resolveFwdPeers(ctran::utils::log2i(nRanks))));
          }));
  const auto& recvPlan = persistPlan->recvPlan();
  const auto& sendPlan = persistPlan->sendPlan();
  // Short-circuit for single rank; GPE should not have been invoked in this
  // case, single-rank out-of-place is handled by user thread via cudaMemcpy
  if (recvPlan.nSteps() == 0) {
    return commSuccess;
  }
  AlgoContext ctx(
      mapper,
      reinterpret_cast<void*>(op->allgather.recvbuff),
      sendSize,
      nRanks,
      recvPlan,
      sendPlan);

  // Register recv buffer
  CTRAN_PROFILER_IF(
      profiler, profiler->startEvent(ctran::ProfilerEvent::BUF_REG));
  auto localMemReg = false;
  FB_COMMCHECK(mapper->searchRegHandle(
      ctx.recvbuff, nRanks * sendSize, &ctx.memHdl, &localMemReg));
  auto memRegGuard = folly::makeGuard([&]() {
    if (localMemReg) {
      mapper->deregDynamic(ctx.memHdl);
    }
  });
  CTRAN_PROFILER_IF(
      profiler, profiler->endEvent(ctran::ProfilerEvent::BUF_REG));

  FB_COMMCHECK(runExchangeAndProgress(ctx, rank, profiler));

  return commSuccess;
}

} // namespace

commResult_t ctranAllGatherStreamedRd(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  auto* algo = comm->ctran_->algo.get();
  // Get or create the persist plan to drive the algorithm. It is persisted for
  // the lifetime of the communicator since it is defined purely based on nRanks
  // and the (immutable) fwdPeers cvar.
  algo->getOrCreatePersistPlan(PersistPlanKey::kAllgatherCtsrd, [&]() {
    const auto rank = comm->statex_->rank();
    const auto nRanks = comm->statex_->nRanks();
    return std::make_unique<PersistPlan>(createPersistPlan(
        rank, nRanks, resolveFwdPeers(ctran::utils::log2i(nRanks))));
  });

  CTRAN_COLL_INFO(
      allGatherAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      -1,
      comm,
      stream);
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHER,
      stream,
      allGatherAlgoName(myAlgo),
      comm->ctran_->getOpCount());
  void* extraCopyBuff = nullptr;
  FB_COMMCHECK(prepareAllGatherArgs(
      opGroup,
      config,
      &extraCopyBuff,
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      comm,
      stream));
  config.colltraceInlineWrites = true;
  config.colltraceEmitStart = true;
  config.colltraceEmitEnd = true;
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      impl,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherCtranStreamedRd)));
  if (extraCopyBuff != nullptr) {
    FB_CUDACHECK(cudaMemcpyAsync(
        recvbuff,
        extraCopyBuff,
        sendcount * commTypeSize(datatype) * comm->statex_->nRanks(),
        cudaMemcpyDefault,
        stream));
  }
  return commSuccess;
}
