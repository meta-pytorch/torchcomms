// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/common/IpcGpuBarrier.cuh"
#include "comms/common/algorithms/all_gather/AllGatherAlgoManager.h"
#include "comms/common/algorithms/all_reduce/AllReduceAlgoManager.h"
#include "comms/common/algorithms/all_to_all/AllToAllAlgoManager.h"
#include "comms/common/algorithms/reduce_scatter/ReduceScatterAlgoManager.h"
#include "comms/ctran/interfaces/IBootstrap.h" // @manual
#include "comms/utils/CudaRAII.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms {

// Forward declaration
class AlgoManagerAllReduce;
class AlgoManagerAllGather;
class AlgoManagerReduceScatter;
class AlgoManagerAllToAll;

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
  AlgoFactory(
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      int nRanks,
      int selfRank,
      int maxBlocks,
      const AllReduceOptions& allReduceOpts);

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

 private:
  std::unique_ptr<AllReduceAlgoManager> allReduceMgr_{nullptr};
};

/**
 * TBD:
 * - above AlgoFactory is kept to compatible with rcclx-stable snapshot to
 * avoid segmentation fault.
 * - below new AlgoFactoryDev is to support AR + new DDA AG, RS, A2A
 * - once rcclx-stable is updated, above AlgoFactory will be deprecated and
 * AlgoFactoryDev will be updated to AlgoFactory for all DDA algorithms
 */
class AlgoFactoryDev {
 public:
  struct AllReduceOptions {
    bool enableDda{false};
    // If msg size is not larger than the threshold,
    // flat (one-shot) DDA will be used
    int ddaFlatMaxThresholdBytes{0};
    // If msg size is not larger than the threshold,
    // tree (two-shot) DDA will be used
    int ddaTreeMaxThresholdBytes{0};
  };

  struct AllGatherOptions {
    bool enableDda{false};
    // If msg size is not larger than the threshold,
    // DDA will be used
    int ddaMaxThresholdBytes{0};
  };

  struct ReduceScatterOptions {
    bool enableDda{false};
    // If msg size is not larger than the threshold,
    // DDA will be used
    int ddaMaxThresholdBytes{0};
  };

  struct AllToAllOptions {
    bool enableDda{false};
    // If msg size is not larger than the threshold,
    // DDA will be used
    int ddaMaxThresholdBytes{0};
  };

  AlgoFactoryDev(
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      int nRanks,
      int selfRank,
      int maxBlocks,
      int ddaSendbufSizeBytes,
      const AllReduceOptions& allReduceOpts,
      const AllGatherOptions& allGatherOpts,
      const ReduceScatterOptions& reduceScatterOpts,
      const AllToAllOptions& allToAllOpts);

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
      cudaStream_t stream) {
    if (allGatherMgr_ == nullptr) {
      return nullptr;
    }
    return allGatherMgr_->getAllGatherAlgo(
        sendbuff, recvbuff, count, datatype, stream);
  }

  std::unique_ptr<AlgoReduceScatter> getReduceScatterAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      commDataType_t datatype,
      cudaStream_t stream) {
    if (reduceScatterMgr_ == nullptr) {
      return nullptr;
    }
    return reduceScatterMgr_->getReduceScatterAlgo(
        sendbuff, recvbuff, count, datatype, stream);
  }

  std::unique_ptr<AlgoAllToAll> getAllToAllAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      commDataType_t datatype,
      cudaStream_t stream) {
    if (allToAllMgr_ == nullptr) {
      return nullptr;
    }
    return allToAllMgr_->getAllToAllAlgo(
        sendbuff, recvbuff, count, datatype, stream);
  }

  AlgoFactoryDev(const AlgoFactoryDev&) = delete;
  AlgoFactoryDev& operator=(const AlgoFactoryDev&) = delete;
  ~AlgoFactoryDev() = default;

 private:
  int nRanks_{0};
  int selfRank_{-1};
  int maxBlocks_{0};
  int ddaSendbufSizeBytes_{0};

  std::unique_ptr<IpcGpuBarrierResources> barrierResources_;
  IpcGpuBarrier barrier_;
  std::unique_ptr<DeviceBuffer> ddaSendbuf_;
  std::unique_ptr<IpcMemHandler> memHandler_;
  // arrary of void* (all ranks' ipc enabled sendbuf) in device memory
  std::unique_ptr<DeviceBuffer> allRankDdaSendbuffs_;

  std::unique_ptr<AllReduceAlgoManagerDev> allReduceMgr_{nullptr};
  std::unique_ptr<AllGatherAlgoManager> allGatherMgr_{nullptr};
  std::unique_ptr<ReduceScatterAlgoManager> reduceScatterMgr_{nullptr};
  std::unique_ptr<AllToAllAlgoManager> allToAllMgr_{nullptr};
};
} // namespace meta::comms
