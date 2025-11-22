// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <stdio.h>
#include <cstddef>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

template <typename T>
__device__ __forceinline__ void
sendImpl(const T* sendbuff, size_t count, int groupIdx, int ngroups) {
  int localRank = statex->localRank();
  int localRanks = statex->nLocalRanks();
  size_t bufSize = shmDevState.bufSize;

  for (int r = 1; r < localRanks; r++) {
    // Ensure each rank sends to different peer at a time to avoid alltoone P2P
    // write congestion. For example, with localRanks = 4, the following
    // schedule is used:
    // - Round0:
    // rank0: s(1)r(3); rank1: s(2)r(0); rank2: s(3)r(1); rank3: s(0)r(2)
    // - Round1:
    // rank0: s(2)r(2); rank1: s(3)r(3); rank2: s(0)r(0); rank3: s(1)r(1)
    // - Round2:
    // rank0: s(3)r(1); rank1: s(0)r(2); rank2: s(1)r(3); rank3: s(2)r(0)
    int sendPeer = (localRank + r) % localRanks;
    size_t displ = count * statex->localRankToRank(sendPeer);

    // get shared buffer
    void* buf = shmDevState.remoteStagingBufsMap[sendPeer];
    const T* sendPtr = sendbuff + displ;

    if (canCopy16(sendPtr, count)) {
      ctranKernMultiStagedSend<uint4>(
          reinterpret_cast<const uint4*>(sendPtr),
          count * sizeof(T) / sizeof(uint4),
          reinterpret_cast<uint4*>(buf),
          sendPeer,
          bufSize / sizeof(uint4),
          groupIdx,
          ngroups);
    } else {
      ctranKernMultiStagedSend<T>(
          sendPtr,
          count,
          reinterpret_cast<T*>(buf),
          sendPeer,
          bufSize / sizeof(T),
          groupIdx,
          ngroups);
    }
  }
}

template <typename T>
__device__ __forceinline__ void
recvImpl(T* recvbuff, size_t count, int groupIdx, int ngroups) {
  int localRank = statex->localRank();
  int localRanks = statex->nLocalRanks();
  size_t bufSize = shmDevState.bufSize;

  for (int r = 1; r < localRanks; r++) {
    // Ensure each rank sends to different peer at a time to avoid alltoone P2P
    // write congestion. For example, with localRanks = 4, the following
    // schedule is used:
    // - Round0:
    // rank0: s(1)r(3); rank1: s(2)r(0); rank2: s(3)r(1); rank3: s(0)r(2)
    // - Round1:
    // rank0: s(2)r(2); rank1: s(3)r(3); rank2: s(0)r(0); rank3: s(1)r(1)
    // - Round2:
    // rank0: s(3)r(1); rank1: s(0)r(2); rank2: s(1)r(3); rank3: s(2)r(0)

    int recvPeer = (localRank + localRanks - r) % localRanks;
    size_t displ = count * statex->localRankToRank(recvPeer);

    // get shared buffer
    void* buf = shmDevState.localStagingBufsMap[recvPeer];
    T* recvPtr = recvbuff + displ;

    if (canCopy16(recvPtr, count)) {
      ctranKernMultiStagedRecv<uint4>(
          reinterpret_cast<uint4*>(recvPtr),
          count * sizeof(T) / sizeof(uint4),
          reinterpret_cast<const uint4*>(buf),
          recvPeer,
          bufSize / sizeof(uint4),
          groupIdx,
          ngroups);
    } else {
      ctranKernMultiStagedRecv<T>(
          recvPtr,
          count,
          reinterpret_cast<const T*>(buf),
          recvPeer,
          bufSize / sizeof(T),
          groupIdx,
          ngroups);
    }
  }
}

enum { GROUP_SEND, GROUP_RECV };

template <typename T>
__global__ void ncclKernelAllToAll(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllToAllArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  const T* sendbuff = reinterpret_cast<const T*>(args.sendbuff);
  T* recvbuff = reinterpret_cast<T*>(args.recvbuff);
  size_t count = args.count;

  // All blocks are involved in self D2D copy
  int localRank = statex->localRank();
  size_t selfDisp = count * statex->localRankToRank(localRank);
  ctranKernCopy<T>(
      sendbuff + selfDisp, recvbuff + selfDisp, count, blockIdx.x, gridDim.x);

  // Partition blocks into a set of send groups and a set of receive groups
  // Let even blocks handle NVL sends, and odd blocks handle NVL receives,
  // and assign groupIdx 0, 1, 2... for block{0,2,4...}@sender and
  // block{1,3,5...}@receiver. The same groupIdx on sender and receiver
  // coordinates to finish a pair of send-receive.
  const auto ngroups = gridDim.x / 2;
  const auto groupIdx = blockIdx.x / 2;
  const bool groupType = blockIdx.x % 2 == 0 ? GROUP_SEND : GROUP_RECV;

  if (groupType == GROUP_RECV) {
    recvImpl(recvbuff, count, groupIdx, ngroups);
  } else {
    sendImpl(sendbuff, count, groupIdx, ngroups);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

#define DECL_CTRAN_ALLTOALL_KERN(T)               \
  template __global__ void ncclKernelAllToAll<T>( \
      int* flag, CtranAlgoDeviceState* devState, CtranKernelAllToAllArgs args)
