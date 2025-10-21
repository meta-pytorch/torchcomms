// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/algorithms/all_reduce/AlgoAllReduce.cuh"
#include "comms/utils/checks.h"

namespace meta::comms {

AlgoAllReduce::AlgoAllReduce(
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
    const void* acc)
    : sendbuff_(sendbuff),
      allRankDdaSendbuffs_(allRankDdaSendbuffs),
      recvbuff_(recvbuff),
      count_(count),
      datatype_(datatype),
      stream_(stream),
      nRanks_(nRanks),
      selfRank_(selfRank),
      maxBlocks_(maxBlocks),
      barrier_(barrier),
      acc_(acc) {}

void AlgoAllReduceDdaFlatIpc::allReduce() {
  // no need to call cudaMemcpyAsync because one-shot kernel
  // copies the data inside the kernel
  TYPED_CALL(datatype_, launchKernel);
}

void AlgoAllReduceDdaTreeIpc::allReduce() {
  // copy src to tmp buffers
  CUDA_CHECK(cudaMemcpyAsync(
      allRankDdaSendbuffs_[selfRank_],
      sendbuff_,
      count_ * commTypeSize(datatype_),
      cudaMemcpyDefault,
      stream_));
  TYPED_CALL(datatype_, launchKernel);
}

} // namespace meta::comms
