// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <stdio.h>
#include <cstddef>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

template <typename T>
__device__ __forceinline__ void sendImpl(
    const T* sendbuff,
    KernelElem* sendElemsList,
    int groupIdx,
    int ngroups) {
  size_t bufSize = shmDevState.bufSize;

  // Host algorithm already schedules send and receives with different peer to
  // avoid P2P congistion. Thus, kernel just runs following the list sequence
  KernelElem* elem = sendElemsList;

  // We preload the next work item out of the linked list so we can have the
  // next work item available with lower latency (instead of waiting to load
  // at the top of the iteration). Rewriting KernelElem to use an array or
  // ring buffer instead of a linked list would solve this problem instead.
  decltype(elem->staged) staged;
  KernelElem* elemNext;

  if (elem) {
    staged = elem->staged;
    elemNext = elem->next;
  }

  while (elem) {
    auto sendPeer = staged.peerRank;
    auto count = staged.count;
    auto displ = staged.displ;
    elem = elemNext;

    // preload next work item
    if (elemNext) {
      staged = elemNext->staged;
      elemNext = elemNext->next;
    }

    // get shared buffer
    void* buf = shmDevState.remoteStagingBufsMap[sendPeer];
    const T* sendPtr = sendbuff + displ;

    bool canCopy16Byte = canCopy16(sendPtr, count);

    if (canCopy16Byte) {
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
recvImpl(T* recvbuff, KernelElem* recvElemsList, int groupIdx, int ngroups) {
  size_t bufSize = shmDevState.bufSize;

  // Host algorithm already schedules send and receives with different peer to
  // avoid P2P congistion. Thus, kernel just runs following the list sequence
  KernelElem* elem = recvElemsList;

  // We preload the next work item out of the linked list so we can have the
  // next work item available with lower latency (instead of waiting to load
  // at the top of the iteration). Rewriting KernelElem to use an array or
  // ring buffer instead of a linked list would solve this problem instead.
  decltype(elem->staged) staged;
  KernelElem* elemNext;

  if (elem) {
    staged = elem->staged;
    elemNext = elem->next;
  }

  while (elem) {
    auto recvPeer = staged.peerRank;
    auto count = staged.count;
    auto displ = staged.displ;
    elem = elemNext;

    // preload next work item
    if (elemNext) {
      staged = elemNext->staged;
      elemNext = elemNext->next;
    }

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
__global__ void ncclKernelAllToAllv(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllToAllvArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  const T* sendbuff = reinterpret_cast<const T*>(args.sendbuff);
  T* recvbuff = reinterpret_cast<T*>(args.recvbuff);
  size_t selfCount = args.selfCount;
  size_t selfSendDispl = args.selfSendDispl;
  size_t selfRecvDispl = args.selfRecvDispl;
  KernelElem* sendElemsList = args.sendElemsList;
  KernelElem* recvElemsList = args.recvElemsList;

  // All blocks are involved in self D2D copy
  ctranKernCopy<T>(
      sendbuff + selfSendDispl,
      recvbuff + selfRecvDispl,
      selfCount,
      blockIdx.x,
      gridDim.x);

  // Partition blocks into a set of send groups and a set of receive groups
  // Let even blocks handle NVL sends, and odd blocks handle NVL receives,
  // and assign groupIdx 0, 1, 2... for block{0,2,4...}@sender and
  // block{1,3,5...}@receiver. The same groupIdx on sender and receiver
  // coordinates to finish a pair of send-receive.
  const auto ngroups = gridDim.x / 2;
  const auto groupIdx = blockIdx.x / 2;
  const bool groupType = blockIdx.x % 2 == 0 ? GROUP_SEND : GROUP_RECV;

  if (groupType == GROUP_RECV) {
    recvImpl(recvbuff, recvElemsList, groupIdx, ngroups);
  } else {
    sendImpl(sendbuff, sendElemsList, groupIdx, ngroups);
  }

  elemsFreeListByGroup(
      groupType == GROUP_RECV ? recvElemsList : sendElemsList, groupIdx, true);

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

#define DECL_CTRAN_ALLTOALLV_KERN(T)               \
  template __global__ void ncclKernelAllToAllv<T>( \
      int* flag,                                   \
      CtranAlgoDeviceState* devState,              \
      CtranKernelAllToAllvArgs args)
