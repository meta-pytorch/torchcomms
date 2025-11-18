// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/common/algorithms/all_to_all/AlgoAllToAll.cuh"
#include "comms/utils/checks.h"

namespace meta::comms {

AlgoAllToAll::AlgoAllToAll(
    const void* sendbuff,
    void** allRankDdaSendbuffs,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    cudaStream_t stream,
    int nRanks,
    int selfRank,
    int maxBlocks,
    IpcGpuBarrier* barrier)
    : sendbuff_(sendbuff),
      allRankDdaSendbuffs_(allRankDdaSendbuffs),
      recvbuff_(recvbuff),
      count_(count),
      datatype_(datatype),
      stream_(stream),
      nRanks_(nRanks),
      selfRank_(selfRank),
      maxBlocks_(maxBlocks),
      barrier_(barrier) {}

void AlgoAllToAllDdaIpc::allToAll() {
  TYPED_CALL(datatype_, launchKernel);
}

} // namespace meta::comms
