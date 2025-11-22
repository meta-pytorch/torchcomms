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
    const AllReduceOptions& allReduceOpts,
    const AllGatherOptions& allGatherOpts) {
  if (allReduceOpts.enableDda || allGatherOpts.enableDda) {
    XLOG(DBG) << "Initializing AllReduceAlgoManager / AllGatherAlgoManager";
    for (int i = 0; i < nRanks; ++i) {
      if (i == selfRank) {
        continue;
      }
      cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
      if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess) {
        CUDA_CHECK(e);
      }
    }
  }

  if (allReduceOpts.enableDda) {
    allReduceMgr_ = std::make_unique<AllReduceAlgoManager>(
        bootstrap,
        nRanks,
        selfRank,
        maxBlocks,
        allReduceOpts.ddaSendbufSizeBytes,
        allReduceOpts.ddaFlatMaxThresholdBytes,
        allReduceOpts.ddaTreeMaxThresholdBytes);
    XLOG(DBG) << "Successfully initialized AllReduceAlgoManager";
  }

  if (allGatherOpts.enableDda) {
    allGatherMgr_ = std::make_unique<AllGatherAlgoManager>(
        bootstrap,
        nRanks,
        selfRank,
        maxBlocks,
        allGatherOpts.ddaSendbufSizeBytes,
        allGatherOpts.ddaMaxThresholdBytes);
    XLOG(DBG) << "Successfully initialized AllGatherAlgoManager";
  }
}

} // namespace meta::comms
