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
    const AllReduceOptions& allReduceOpts) {
  if (allReduceOpts.enableDda) {
    XLOG(DBG) << "Initializing AllReduceAlgoManager";
    for (int i = 0; i < nRanks; ++i) {
      if (i == selfRank) {
        continue;
      }
      cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
      if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess) {
        CUDA_CHECK(e);
      }
    }

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
}

} // namespace meta::comms
