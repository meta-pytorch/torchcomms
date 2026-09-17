// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "core.h"

NCCL_API(
    ncclResult_t,
    ncclGetMultiPeerDeviceHandle,
    ncclComm_t comm,
    void** outTransportsPtr,
    int* outMyRank,
    int* outNRanks,
    int* outNumNvlPeers,
    int* outNumIbPeers);
ncclResult_t ncclGetMultiPeerDeviceHandle(
    ncclComm_t /*comm*/,
    void** /*outTransportsPtr*/,
    int* /*outMyRank*/,
    int* /*outNRanks*/,
    int* /*outNumNvlPeers*/,
    int* /*outNumIbPeers*/) {
  return ncclInvalidUsage;
}
