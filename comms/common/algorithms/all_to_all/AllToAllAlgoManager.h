// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/common/IpcGpuBarrier.cuh"
#include "comms/common/algorithms/all_to_all/AlgoAllToAll.cuh"
#include "comms/ctran/interfaces/IBootstrap.h" // @manual
#include "comms/utils/CudaRAII.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms {

class AllToAllAlgoManager {
 public:
  AllToAllAlgoManager(
      int nRanks,
      int selfRank,
      int maxBlocks,
      int ddaSendbufSizeBytes,
      int ddaMaxThresholdBytes,
      void** allRankDdaSendbuffs,
      IpcGpuBarrier* barrier);
  AllToAllAlgoManager(const AllToAllAlgoManager&) = delete;
  AllToAllAlgoManager(AllToAllAlgoManager&&) = delete;

  std::unique_ptr<AlgoAllToAll> getAllToAllAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      commDataType_t datatype,
      cudaStream_t stream);

 private:
  int nRanks_{0};
  int selfRank_{-1};
  int maxBlocks_{0};
  int ddaSendbufSizeBytes_{0};
  int ddaMaxThresholdBytes_{0};
  void** allRankDdaSendbuffs_{nullptr};
  IpcGpuBarrier* barrier_;
};

} // namespace meta::comms
