// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/common/algorithms/all_gather/AllGatherAlgoManager.h"
#include "comms/common/algorithms/all_reduce/AllReduceAlgoManager.h"
#include "comms/ctran/interfaces/IBootstrap.h" // @manual
#include "comms/utils/commSpecs.h"

namespace meta::comms {

// Forward declaration
class AlgoManagerAllReduce;
class AlgoManagerAllGather;

/**
 * per communicator per rank Algorithm factory that
 * - manages all the available algorithm instances for a given collective
 * - selects an optimal algorithm based on the input and environments
 */
class AlgoFactory {
 public:
  struct AllReduceOptions {
    bool enableDda{false};
    int ddaSendbufSizeBytes{0};
    // If msg size is not larger than the threshold,
    // flat (one-shot) DDA will be used
    int ddaFlatMaxThresholdBytes{0};
    // If msg size is not larger than the threshold,
    // tree (two-shot) DDA will be used
    int ddaTreeMaxThresholdBytes{0};
  };
  struct AllGatherOptions {
    bool enableDda{false};
    int ddaSendbufSizeBytes{0};
    // If msg size is not larger than the threshold,
    // DDA will be used
    int ddaMaxThresholdBytes{0};
  };
  AlgoFactory(
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      int nRanks,
      int selfRank,
      int maxBlocks,
      const AllReduceOptions& allReduceOpts,
      const AllGatherOptions& allGatherOpts);

  std::unique_ptr<AlgoAllReduce> getAllReduceAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      commDataType_t datatype,
      cudaStream_t stream,
      const void* acc = nullptr) {
    if (allReduceMgr_ == nullptr) {
      return nullptr;
    }
    return allReduceMgr_->getAllReduceAlgo(
        sendbuff, recvbuff, count, datatype, stream, acc);
  }

  std::unique_ptr<AlgoAllGather> getAllGatherAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      commDataType_t datatype,
      cudaStream_t stream,
      const void* acc = nullptr) {
    if (allGatherMgr_ == nullptr) {
      return nullptr;
    }
    return allGatherMgr_->getAllGatherAlgo(
        sendbuff, recvbuff, count, datatype, stream, acc);
  }

 private:
  std::unique_ptr<AllReduceAlgoManager> allReduceMgr_{nullptr};
  std::unique_ptr<AllGatherAlgoManager> allGatherMgr_{nullptr};
};
} // namespace meta::comms
