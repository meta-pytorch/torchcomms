// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/logging/xlog.h>
#include "comms/common/algorithms/AlgoFactory.cuh"
#include "comms/utils/checks.h"

namespace meta::comms {

AlgoFactory::AlgoFactory(
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    int nRanks,
    int selfRank,
    int maxBlocks,
    int ddaSendbufSizeBytes,
    const AllReduceOptions& allReduceOpts,
    const AllGatherOptions& allGatherOpts,
    const ReduceScatterOptions& reduceScatterOpts,
    const AllToAllOptions& allToAllOpts)
    : nRanks_(nRanks),
      selfRank_(selfRank),
      maxBlocks_(maxBlocks),
      ddaSendbufSizeBytes_(ddaSendbufSizeBytes) {
  if (allReduceOpts.enableDda || allGatherOpts.enableDda ||
      reduceScatterOpts.enableDda || allToAllOpts.enableDda) {
    XLOG(DBG)
        << "Initializing AllReduce / AllGather / ReduceScatter / AllToAll AlgoManager";

    for (int i = 0; i < nRanks; ++i) {
      if (i == selfRank) {
        continue;
      }
      cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
      if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess) {
        CUDA_CHECK(e);
      }
    }

    auto [barrierResources, barrier] =
        IpcGpuBarrier::mallocAndInit(nRanks_, maxBlocks_, selfRank_, bootstrap);
    barrierResources_ = std::move(barrierResources);
    barrier_ = barrier;

    ddaSendbuf_ = std::make_unique<DeviceBuffer>(ddaSendbufSizeBytes_);
    memHandler_ =
        std::make_unique<IpcMemHandler>(bootstrap, selfRank_, nRanks_);
    memHandler_->addSelfDeviceMemPtr(ddaSendbuf_->get());
    memHandler_->exchangeMemPtrs();

    std::vector<void*> ipcSendbufs(nRanks_);
    for (int i = 0; i < nRanks_; ++i) {
      ipcSendbufs[i] = memHandler_->getPeerDeviceMemPtr(i);
    }

    allRankDdaSendbuffs_ =
        std::make_unique<DeviceBuffer>(sizeof(void*) * nRanks_);
    CUDA_CHECK(cudaMemcpy(
        allRankDdaSendbuffs_->get(),
        ipcSendbufs.data(),
        sizeof(void*) * nRanks_,
        cudaMemcpyDefault));
  }

  if (allReduceOpts.enableDda) {
    allReduceMgr_ = std::make_unique<AllReduceAlgoManager>(
        nRanks,
        selfRank,
        maxBlocks,
        ddaSendbufSizeBytes,
        allReduceOpts.ddaFlatMaxThresholdBytes,
        allReduceOpts.ddaTreeMaxThresholdBytes,
        reinterpret_cast<void**>(allRankDdaSendbuffs_->get()),
        &barrier_);
  }

  if (allGatherOpts.enableDda) {
    allGatherMgr_ = std::make_unique<AllGatherAlgoManager>(
        nRanks,
        selfRank,
        maxBlocks,
        ddaSendbufSizeBytes,
        allGatherOpts.ddaMaxThresholdBytes,
        reinterpret_cast<void**>(allRankDdaSendbuffs_->get()),
        &barrier_);
  }

  if (reduceScatterOpts.enableDda) {
    reduceScatterMgr_ = std::make_unique<ReduceScatterAlgoManager>(
        nRanks,
        selfRank,
        maxBlocks,
        ddaSendbufSizeBytes,
        reduceScatterOpts.ddaMaxThresholdBytes,
        reinterpret_cast<void**>(allRankDdaSendbuffs_->get()),
        &barrier_);
  }

  if (allToAllOpts.enableDda) {
    allToAllMgr_ = std::make_unique<AllToAllAlgoManager>(
        nRanks,
        selfRank,
        maxBlocks,
        ddaSendbufSizeBytes,
        allToAllOpts.ddaMaxThresholdBytes,
        reinterpret_cast<void**>(allRankDdaSendbuffs_->get()),
        &barrier_);
  }
}

AlgoFactory::~AlgoFactory() {}

} // namespace meta::comms
