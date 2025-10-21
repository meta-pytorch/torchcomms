// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/common/IpcGpuBarrier.cuh"
#include "comms/common/algorithms/all_reduce/AlgoAllReduce.cuh"
#include "comms/ctran/interfaces/IBootstrap.h" // @manual
#include "comms/utils/CudaRAII.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms {

class AllReduceAlgoManager {
 public:
  AllReduceAlgoManager(
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      int nRanks,
      int selfRank,
      int maxBlocks,
      int ddaSendbufSizeBytes,
      int ddaFlatMaxThresholdBytes,
      int ddaTreeMaxThresholdBytes);
  AllReduceAlgoManager(const AllReduceAlgoManager&) = delete;
  AllReduceAlgoManager(AllReduceAlgoManager&&) = delete;

  std::unique_ptr<AlgoAllReduce> getAllReduceAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      commDataType_t datatype,
      cudaStream_t stream,
      const void* acc);

 private:
  int nRanks_{0};
  int selfRank_{-1};
  int maxBlocks_{0};
  int ddaSendbufSizeBytes_{0};
  int ddaFlatMaxThresholdBytes_{0};
  int ddaTreeMaxThresholdBytes_{0};
  std::unique_ptr<IpcGpuBarrierResources> barrierResources_;
  IpcGpuBarrier barrier_;
  std::unique_ptr<DeviceBuffer> ddaSendbuf_;
  std::unique_ptr<IpcMemHandler> memHandler_;
  // arrary of void* (all ranks' ipc enabled sendbuf) in device memory
  std::unique_ptr<DeviceBuffer> allRankDdaSendbuffs_;
};

} // namespace meta::comms
