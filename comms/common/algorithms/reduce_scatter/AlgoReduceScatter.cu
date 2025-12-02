// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/common/algorithms/reduce_scatter/AlgoReduceScatter.cuh"
#include "comms/utils/checks.h"

namespace meta::comms {

AlgoReduceScatter::AlgoReduceScatter(
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

void AlgoReduceScatterDdaIpc::reduceScatter() {
  // copy src to tmp buffers
  CUDA_CHECK(cudaMemcpyAsync(
      allRankDdaSendbuffs_[selfRank_],
      sendbuff_,
      nRanks_ * count_ * commTypeSize(datatype_),
      cudaMemcpyDefault,
      stream_));
  TYPED_CALL(datatype_, launchKernel);
}

} // namespace meta::comms
