// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
// sync ranks on the same node.
// rank: should be local rank
__global__ void ncclKernelNvlBarrier(
    int rank,
    int nLocalRanks,
    CtranAlgoDeviceState* devState) {
  devStateLoadToShm(devState);
  barrier(rank, nLocalRanks);
}
