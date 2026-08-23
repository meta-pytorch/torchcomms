// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/algorithms/all_reduce/AllReduceAlgoManager.h"

#include "comms/utils/logger/CudaLog.h"

namespace meta::comms {

AllReduceAlgoManager::AllReduceAlgoManager(
    std::shared_ptr<IBootstrap> bootstrap,
    int nRanks,
    int selfRank,
    int maxBlocks,
    int ddaSendbufSizeBytes,
    int ddaFlatMaxThresholdBytes,
    int ddaTreeMaxThresholdBytes)
    : nRanks_(nRanks),
      selfRank_(selfRank),
      maxBlocks_(maxBlocks),
      ddaSendbufSizeBytes_(ddaSendbufSizeBytes),
      ddaFlatMaxThresholdBytes_(ddaFlatMaxThresholdBytes),
      ddaTreeMaxThresholdBytes_(ddaTreeMaxThresholdBytes) {
  auto [barrierResources, barrier] =
      IpcGpuBarrier::mallocAndInit(nRanks_, maxBlocks_, selfRank_, bootstrap);
  barrierResources_ = std::move(barrierResources);
  barrier_ = barrier;

  ddaSendbuf_ = std::make_unique<DeviceBuffer>(ddaSendbufSizeBytes_);
  memHandler_ = std::make_unique<IpcMemHandler>(bootstrap, selfRank_, nRanks_);
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
  COMMS_CUDA_LOG(DBG, "Successfully initialized AllReduceAlgoManager");
}

std::unique_ptr<AlgoAllReduce> AllReduceAlgoManager::getAllReduceAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    cudaStream_t stream,
    const void* acc) {
  return nullptr;
}

AllReduceAlgoManagerDev::AllReduceAlgoManagerDev(
    int nRanks,
    int selfRank,
    int maxBlocks,
    int ddaSendbufSizeBytes,
    int ddaFlatMaxThresholdBytes,
    int ddaTreeMaxThresholdBytes,
    void** allRankDdaSendbuffs,
    IpcGpuBarrier* barrier)
    : nRanks_(nRanks),
      selfRank_(selfRank),
      maxBlocks_(maxBlocks),
      ddaSendbufSizeBytes_(ddaSendbufSizeBytes),
      ddaFlatMaxThresholdBytes_(ddaFlatMaxThresholdBytes),
      ddaTreeMaxThresholdBytes_(ddaTreeMaxThresholdBytes),
      allRankDdaSendbuffs_(allRankDdaSendbuffs),
      barrier_(barrier) {
  COMMS_CUDA_LOG(DBG, "Successfully initialized AllReduceAlgoManager");
}

std::unique_ptr<AlgoAllReduce> AllReduceAlgoManagerDev::getAllReduceAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    cudaStream_t stream,
    const void* acc) {
  if ((count * commTypeSize(datatype)) > ddaSendbufSizeBytes_) {
    // AllReduce: (count x datatype) size must fit into the dda sendbuf
    COMMS_CUDA_LOG(
        DBG,
        "Not using custom all reduce algo because message size %zu is larger than ddaSendbufSizeBytes %d",
        count * commTypeSize(datatype),
        ddaSendbufSizeBytes_);
    return nullptr;
  }

  if (((uintptr_t)sendbuff % 16) || ((uintptr_t)recvbuff % 16) ||
      ((count * commTypeSize(datatype)) % 16)) {
    // 16 byte alignment as we do 16-byte loads in DDA kernel
    COMMS_CUDA_LOG(
        DBG,
        "Not using custom all reduce algo because send/recv buff or msg size is not 16-byte aligned");
    return nullptr;
  }

  if (datatype != commBfloat16 && datatype != commFloat16 &&
      datatype != commFloat) {
    // we currently only support bf16, half, float
    COMMS_CUDA_LOG(
        DBG,
        "Not using custom all reduce algo because cudaDataType_t datatype %d is not supported",
        static_cast<int>(datatype));
    return nullptr;
  }

  std::unique_ptr<AlgoAllReduce> algo;
  if (count * commTypeSize(datatype) > ddaTreeMaxThresholdBytes_) {
    COMMS_CUDA_LOG(
        DBG,
        "Not using custom all reduce algo because msg size %zu is larger than DDA algo threshold %d",
        count * commTypeSize(datatype),
        ddaTreeMaxThresholdBytes_);
    return nullptr;
  } else if (count * commTypeSize(datatype) > ddaFlatMaxThresholdBytes_) {
    if (count % nRanks_ || ((count / nRanks_ * commTypeSize(datatype)) % 16)) {
      // In two-shot algo, each rank is reduces count/nRanks_ elements so we
      // need to make sure that is 16-byte aligned
      COMMS_CUDA_LOG(
          DBG,
          "Not using DDA Tree all reduce algo because send/recv buff or msg size is not 16-byte aligned for each rank");
      return nullptr;
    }
    algo = std::make_unique<AlgoAllReduceDdaTreeIpc>(
        sendbuff,
        allRankDdaSendbuffs_,
        recvbuff,
        count,
        datatype,
        stream,
        nRanks_,
        selfRank_,
        maxBlocks_,
        barrier_,
        acc);
  } else {
    algo = std::make_unique<AlgoAllReduceDdaFlatIpc>(
        sendbuff,
        allRankDdaSendbuffs_,
        recvbuff,
        count,
        datatype,
        stream,
        nRanks_,
        selfRank_,
        maxBlocks_,
        barrier_,
        acc);
  }
  return algo;
}

} // namespace meta::comms
