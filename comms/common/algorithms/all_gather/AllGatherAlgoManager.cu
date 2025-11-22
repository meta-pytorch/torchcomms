// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/common/algorithms/all_gather/AllGatherAlgoManager.h"

namespace meta::comms {

AllGatherAlgoManager::AllGatherAlgoManager(
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    int nRanks,
    int selfRank,
    int maxBlocks,
    int ddaSendbufSizeBytes,
    int ddaMaxThresholdBytes)
    : nRanks_(nRanks),
      selfRank_(selfRank),
      maxBlocks_(maxBlocks),
      ddaSendbufSizeBytes_(ddaSendbufSizeBytes),
      ddaMaxThresholdBytes_(ddaMaxThresholdBytes) {
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
  XLOG(DBG) << "Successfully initialized AllGatherAlgoManager";
}

std::unique_ptr<AlgoAllGather> AllGatherAlgoManager::getAllGatherAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    cudaStream_t stream,
    const void* acc) {
  if (count * commTypeSize(datatype) > ddaSendbufSizeBytes_) {
    // msg size must fit into the dda sendbuf
    XLOG(DBG) << "Not using custom all gather algo because message size "
              << count * commTypeSize(datatype)
              << " is larger than ddaSendbufSizeBytes " << ddaSendbufSizeBytes_;
    return nullptr;
  }
  if (((uintptr_t)sendbuff % 16) || ((uintptr_t)recvbuff % 16) ||
      ((count * commTypeSize(datatype)) % 16)) {
    // 16 byte alignment as we do 16-byte loads in DDA kernel
    XLOG(DBG) << "Not using custom all gather algo because send/recv buff "
                 "or msg size is not 16-byte aligned";
    return nullptr;
  }

  if (datatype != commBfloat16 && datatype != commFloat16) {
    // we currently only support bf16 and half
    XLOG(DBG)
        << "Not using custom all gather algo because cudaDataType_t datatype "
        << static_cast<int>(datatype) << " is not supported";
    return nullptr;
  }

  std::unique_ptr<AlgoAllGather> algo;
  if (count * commTypeSize(datatype) > ddaMaxThresholdBytes_) {
    XLOG(DBG) << "Not using custom all gather algo because msg size "
              << count * commTypeSize(datatype)
              << " is larger than DDA algo threshold " << ddaMaxThresholdBytes_;
    return nullptr;
  } else {
    if ((count * commTypeSize(datatype)) % 16) {
      XLOG(DBG) << "Not using DDA all gather algo because send/recv buff "
                   "or msg size is not 16-byte aligned for each rank";
      return nullptr;
    }
    algo = std::make_unique<AlgoAllGatherDdaIpc>(
        sendbuff,
        reinterpret_cast<void**>(allRankDdaSendbuffs_->get()),
        recvbuff,
        count,
        datatype,
        stream,
        nRanks_,
        selfRank_,
        maxBlocks_,
        &barrier_,
        acc);
  }
  return algo;
}

} // namespace meta::comms
