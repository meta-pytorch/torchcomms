// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "comms/common/IpcGpuBarrier.cuh"
#include "comms/common/algorithms/AlgoUtils.h"
#include "comms/common/algorithms/all_reduce/all_reduce_dda.cuh"
#include "comms/utils/checks.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms {

/**
 * This class defines common interface for all AllReduce Algorithms
 * subclasses are expected to provide actual implementation
 */
class AlgoAllReduce {
 public:
  // NOTE: acc is applied to final result AFTER all-reduce operation, so it's
  // AR-result + ACC not AR-result + NRANKS*ACC.
  AlgoAllReduce(
      const void* sendbuff,
      void** allRankDdaSendbuffs,
      void* recvbuff,
      size_t count,
      commDataType_t datatype,
      cudaStream_t stream,
      int nRanks,
      int selfRank,
      int maxBlocks,
      IpcGpuBarrier* barrier,
      const void* acc);

  virtual ~AlgoAllReduce() = default;

  virtual void allReduce() = 0;

 protected:
  const void* sendbuff_{nullptr};
  void** allRankDdaSendbuffs_{nullptr};
  void* recvbuff_{nullptr};
  size_t count_{0};
  commDataType_t datatype_{commBfloat16};
  cudaStream_t stream_{nullptr};
  int nRanks_{0};
  int selfRank_{0};
  const size_t maxBlocks_{0};
  IpcGpuBarrier* barrier_;
  const void* acc_{nullptr};
};

class AlgoAllReduceDdaFlatIpc : public AlgoAllReduce {
 public:
  using AlgoAllReduce::AlgoAllReduce;

  void allReduce() override;

 private:
  template <typename T>
  void launchKernel() {
    const void* func = nullptr;
    if (acc_ == nullptr) {
      ASSIGN_FUNC_NRANKS(
          func, ddaAllReduceFlatIpc, nRanks_, false /* hasAcc */);
    } else {
      ASSIGN_FUNC_NRANKS(func, ddaAllReduceFlatIpc, nRanks_, true /* hasAcc */);
    }

    auto gridBlock = getGridAndBlockDims(count_, datatype_, maxBlocks_);
    const auto& grid = gridBlock.first;
    const auto& block = gridBlock.second;

    void* args[] = {
        &allRankDdaSendbuffs_,
        &recvbuff_,
        &count_,
        &sendbuff_,
        &selfRank_,
        barrier_,
        &acc_};
    CUDA_CHECK(cudaLaunchKernel(func, grid, block, args, 0, stream_));
  }
};

class AlgoAllReduceDdaTreeIpc : public AlgoAllReduce {
 public:
  using AlgoAllReduce::AlgoAllReduce;

  void allReduce() override;

 private:
  template <typename T>
  void launchKernel() {
    const void* func = nullptr;
    if (acc_ == nullptr) {
      ASSIGN_FUNC_NRANKS(
          func, ddaAllReduceTreeIpc, nRanks_, false /* hasAcc */);
    } else {
      ASSIGN_FUNC_NRANKS(func, ddaAllReduceTreeIpc, nRanks_, true /* hasAcc */);
    }

    auto gridBlock = getGridAndBlockDims(count_, datatype_, maxBlocks_);
    const auto& grid = gridBlock.first;
    const auto& block = gridBlock.second;

    void* args[] = {
        &allRankDdaSendbuffs_,
        &recvbuff_,
        &count_,
        &sendbuff_,
        &selfRank_,
        barrier_,
        &acc_};
    CUDA_CHECK(cudaLaunchKernel(func, grid, block, args, 0, stream_));
  }
};

} // namespace meta::comms
