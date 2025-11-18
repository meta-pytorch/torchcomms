// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "comms/common/IpcGpuBarrier.cuh"
#include "comms/common/algorithms/AlgoUtils.h"
#include "comms/common/algorithms/all_gather/all_gather_dda.cuh"
#include "comms/utils/checks.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms {

/**
 * This class defines common interface for all AllGather Algorithms
 * subclasses are expected to provide actual implementation
 */
class AlgoAllGather {
 public:
  // NOTE: acc is not used for all-gather
  AlgoAllGather(
      const void* sendbuff,
      void** allRankDdaSendbuffs,
      void* recvbuff,
      size_t count,
      commDataType_t datatype,
      cudaStream_t stream,
      int nRanks,
      int selfRank,
      int maxBlocks,
      IpcGpuBarrier* barrier);

  virtual ~AlgoAllGather() = default;

  virtual void allGather() = 0;

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
};

class AlgoAllGatherDdaIpc : public AlgoAllGather {
 public:
  using AlgoAllGather::AlgoAllGather;

  void allGather() override;

 private:
  template <typename T>
  void launchKernel() {
    const void* func = nullptr;

    ASSIGN_FUNC_NRANKS(func, ddaAllGatherIpc, nRanks_, false /* hasAcc */);

    auto gridBlock =
        getGridAndBlockDims(nRanks_ * count_, datatype_, maxBlocks_);
    const auto& grid = gridBlock.first;
    const auto& block = gridBlock.second;

    void* args[] = {
        &allRankDdaSendbuffs_,
        &recvbuff_,
        &count_,
        &sendbuff_,
        &selfRank_,
        barrier_};
    CUDA_CHECK(cudaLaunchKernel(func, grid, block, args, 0, stream_));
  }
};

} // namespace meta::comms
