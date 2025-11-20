// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/common/algorithms/all_gather/AllGatherAlgoManager.h"

namespace meta::comms {

AllGatherAlgoManager::AllGatherAlgoManager(
    int nRanks,
    int selfRank,
    int maxBlocks,
    int ddaSendbufSizeBytes,
    int ddaMaxThresholdBytes,
    void** allRankDdaSendbuffs,
    IpcGpuBarrier* barrier)
    : nRanks_(nRanks),
      selfRank_(selfRank),
      maxBlocks_(maxBlocks),
      ddaSendbufSizeBytes_(ddaSendbufSizeBytes),
      ddaMaxThresholdBytes_(ddaMaxThresholdBytes),
      allRankDdaSendbuffs_(allRankDdaSendbuffs),
      barrier_(barrier) {
  XLOG(DBG) << "Successfully initialized AllGatherAlgoManager";
}

std::unique_ptr<AlgoAllGather> AllGatherAlgoManager::getAllGatherAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    cudaStream_t stream) {
  if ((nRanks_ * count * commTypeSize(datatype)) > ddaSendbufSizeBytes_) {
    // AG: msgSize = (nRanks_ x count x datatype) must fit into the dda sendbuf
    XLOG(DBG) << "Not using custom all gather algo because message size "
              << nRanks_ * count * commTypeSize(datatype)
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

  if (datatype != commBfloat16 && datatype != commFloat16 &&
      datatype != commFloat) {
    // we currently only support bf16, half, float
    XLOG(DBG)
        << "Not using custom all gather algo because cudaDataType_t datatype "
        << static_cast<int>(datatype) << " is not supported";
    return nullptr;
  }

  std::unique_ptr<AlgoAllGather> algo;
  if ((nRanks_ * count * commTypeSize(datatype)) > ddaMaxThresholdBytes_) {
    // AG: msgSize = (nRanks_ x count x datatype) must less than algo threshold
    XLOG(DBG) << "Not using custom all gather algo because msg size "
              << nRanks_ * count * commTypeSize(datatype)
              << " is larger than DDA algo threshold " << ddaMaxThresholdBytes_;
    return nullptr;
  } else {
    if (((count * commTypeSize(datatype)) % 16) ||
        ((nRanks_ * count * commTypeSize(datatype)) % 16)) {
      XLOG(DBG) << "Not using DDA all gather algo because send/recv buff "
                   "or msg size is not 16-byte aligned for each rank";
      return nullptr;
    }
    algo = std::make_unique<AlgoAllGatherDdaIpc>(
        sendbuff,
        allRankDdaSendbuffs_,
        recvbuff,
        count,
        datatype,
        stream,
        nRanks_,
        selfRank_,
        maxBlocks_,
        barrier_);
  }
  return algo;
}

} // namespace meta::comms
