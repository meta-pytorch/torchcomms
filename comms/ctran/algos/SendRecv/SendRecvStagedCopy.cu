// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/SendRecv/Types.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

enum { GROUP_SEND, GROUP_RECV };

template <typename T>
__device__ __forceinline__ void sendImpl(
    const ctran::sendrecv::SendRecvOp* sends,
    size_t numSends,
    int groupIdx) {
  const auto bufSize = shmDevState.bufSize;

  for (auto i = 0; i < numSends; i++) {
    const auto sendPeer = sends[i].peerLocalRank;
    const auto nbytes = sends[i].nbytes;
    const auto nGroups = sends[i].nGroups;
    if (groupIdx >= nGroups) {
      continue;
    }
    const T* sendPtr = reinterpret_cast<const T*>(sends[i].buff);
    size_t count = nbytes / sizeof(T);

    void* buf = shmDevState.remoteStagingBufsMap[sendPeer];
    if (canCopy16(sendPtr, count)) {
      ctranKernMultiStagedSend<uint4>(
          reinterpret_cast<const uint4*>(sendPtr),
          count * sizeof(T) / sizeof(uint4),
          reinterpret_cast<uint4*>(buf),
          sendPeer,
          bufSize / sizeof(uint4),
          groupIdx,
          nGroups);
    } else {
      ctranKernMultiStagedSend<T>(
          sendPtr,
          count,
          reinterpret_cast<T*>(buf),
          sendPeer,
          bufSize / sizeof(T),
          groupIdx,
          nGroups);
    }
  }
}

template <typename T>
__device__ __forceinline__ void recvImpl(
    const ctran::sendrecv::SendRecvOp* recvs,
    size_t numRecvs,
    int groupIdx) {
  size_t bufSize = shmDevState.bufSize;

  for (auto i = 0; i < numRecvs; i++) {
    const auto recvPeer = recvs[i].peerLocalRank;
    const auto nbytes = recvs[i].nbytes;
    const auto nGroups = recvs[i].nGroups;
    if (groupIdx >= nGroups) {
      continue;
    }
    T* recvPtr = reinterpret_cast<T*>(recvs[i].buff);
    size_t count = nbytes / sizeof(T);

    void* buf = shmDevState.localStagingBufsMap[recvPeer];
    if (canCopy16(recvPtr, count)) {
      ctranKernMultiStagedRecv<uint4>(
          reinterpret_cast<uint4*>(recvPtr),
          count * sizeof(T) / sizeof(uint4),
          reinterpret_cast<const uint4*>(buf),
          recvPeer,
          bufSize / sizeof(uint4),
          groupIdx,
          nGroups);
    } else {
      ctranKernMultiStagedRecv<T>(
          recvPtr,
          count,
          reinterpret_cast<const T*>(buf),
          recvPeer,
          bufSize / sizeof(T),
          groupIdx,
          nGroups);
    }
  }
}

__global__ void __launch_bounds__(1024, 1) ncclKernelSendRecvStaged(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  // TODO: currently first args.numRecvBlocks blocks allocated for recv, and
  // rest for send. Sends and recvs will happen sequentially in allocated blocks
  // we will need better allocation of blocks based on send/recv sizes. May
  // define multiple worker groups as in
  // fbcode/comms/ctran/algos/AllToAllvDedup/ExecImpl.cu
  const auto groupType =
      blockIdx.x < args.numRecvBlocks ? GROUP_RECV : GROUP_SEND;
  const auto groupIdx =
      groupType == GROUP_RECV ? blockIdx.x : blockIdx.x - args.numRecvBlocks;

  if (groupType == GROUP_RECV) {
    recvImpl<uint8_t>(args.recvs, args.numRecvs, groupIdx);
  } else {
    sendImpl<uint8_t>(args.sends, args.numSends, groupIdx);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
