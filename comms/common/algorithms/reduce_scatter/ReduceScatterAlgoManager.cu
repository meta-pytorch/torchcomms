// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/common/algorithms/reduce_scatter/ReduceScatterAlgoManager.h"

#include "comms/utils/logger/CudaLog.h"

namespace meta::comms {

ReduceScatterAlgoManager::ReduceScatterAlgoManager(
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
  COMMS_CUDA_LOG(DBG, "Successfully initialized ReduceScatterAlgoManager");
}

std::unique_ptr<AlgoReduceScatter>
ReduceScatterAlgoManager::getReduceScatterAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    cudaStream_t stream) {
  if ((nRanks_ * count * commTypeSize(datatype)) > ddaSendbufSizeBytes_) {
    // RS: msgSize = (nRanks_ x count x datatype) must fit into the dda sendbuf
    COMMS_CUDA_LOG(
        DBG,
        "Not using custom reduce scatter algo because message size %zu is larger than ddaSendbufSizeBytes %d",
        count * commTypeSize(datatype),
        ddaSendbufSizeBytes_);
    return nullptr;
  }
  if (((uintptr_t)sendbuff % 16) || ((uintptr_t)recvbuff % 16) ||
      ((count * commTypeSize(datatype)) % 16)) {
    // 16 byte alignment as we do 16-byte loads in DDA kernel
    COMMS_CUDA_LOG(
        DBG,
        "Not using custom reduce scatter algo because send/recv buff or msg size is not 16-byte aligned");
    return nullptr;
  }

  if (datatype != commBfloat16 && datatype != commFloat16 &&
      datatype != commFloat) {
    // we currently only support bf16, half, float
    COMMS_CUDA_LOG(
        DBG,
        "Not using custom reduce scatter algo because cudaDataType_t datatype %d is not supported",
        static_cast<int>(datatype));
    return nullptr;
  }

  std::unique_ptr<AlgoReduceScatter> algo;
  if ((nRanks_ * count * commTypeSize(datatype)) > ddaMaxThresholdBytes_) {
    // RS: msgSize = (nRanks_ x count x datatype) must less than algo threshold
    COMMS_CUDA_LOG(
        DBG,
        "Not using custom reduce scatter algo because msg size %zu is larger than DDA algo threshold %d",
        nRanks_ * count * commTypeSize(datatype),
        ddaMaxThresholdBytes_);
    return nullptr;
  } else {
    if (((count * commTypeSize(datatype)) % 16) ||
        ((nRanks_ * count * commTypeSize(datatype)) % 16)) {
      COMMS_CUDA_LOG(
          DBG,
          "Not using DDA reduce scatter algo because send/recv buff or msg size is not 16-byte aligned for each rank");
      return nullptr;
    }
    algo = std::make_unique<AlgoReduceScatterDdaIpc>(
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
