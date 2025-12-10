// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <stdio.h>
#include <cstddef>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

// - Half blocks handle send, and the other handle receive
// - Used in p2p elem to ensure ngroups number of inuse flags are checked when
// reclaiming. This avoids cross-block sync in kernel

// Ensure each rank sends to different peer at a time to avoid alltoone P2P
// write congestion. For example, with localRanks = 4, the following
// schedule is used:
// - Round0:
// rank0: s(1)r(3); rank1: s(2)r(0); rank2: s(3)r(1); rank3: s(0)r(2)
// - Round1:
// rank0: s(2)r(2); rank1: s(3)r(3); rank2: s(0)r(0); rank3: s(1)r(1)
// - Round2:
// rank0: s(3)r(1); rank1: s(0)r(2); rank2: s(1)r(3); rank3: s(2)r(0)

enum ALGOTYPE { DYNAMIC, DYNAMIC_SPLIT, DYNAMIC_SPLIT_NON_CONTIG };
enum { GROUP_SEND, GROUP_RECV };

template <typename T>
__device__ __forceinline__ void sendData(
    const T* sendPtr,
    size_t count,
    int sendPeer,
    int localRank,
    int groupIdx,
    int ngroups) {
  void* buf = shmDevState.remoteStagingBufsMap[sendPeer];
  size_t bufSize = shmDevState.bufSize;

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

template <typename T>
__device__ __forceinline__ void recvData(
    T* recvPtr,
    size_t count,
    int recvPeer,
    int localRank,
    int groupIdx,
    int ngroups) {
  size_t bufSize = shmDevState.bufSize;
  const void* buf = shmDevState.localStagingBufsMap[recvPeer];

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

template <typename T>
__device__ __forceinline__ void sendImplNonContig(
    const T* const* sendbuffs,
    const size_t* sendcounts,
    size_t sendcountsLength,
    const size_t* inputChunkIndices,
    const size_t* inputChunkCountPerRank,
    size_t maxInputChunkCountPerRank,
    int groupIdx,
    int ngroups) {
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  for (int r = 0; r < nLocalRanks - 1; r++) {
    auto sendPeer = (localRank + r + 1) % nLocalRanks;
    auto sendPeerGlobal = statex->localRankToRank(sendPeer);

    // get shared buffer
    int* sync =
        &(devSyncGetLoc<LOCAL>(sendPeer)->syncs[groupIdx].stepOnSameBlockIdx);
    size_t* sendcountsPeerAllToAllvDynamicBufsMap =
        reinterpret_cast<size_t*>(
            shmDevState.peerAllToAllvDynamicBufsMap[sendPeer]) +
        (localRank * ngroups + groupIdx) *
            (sendcountsLength + 1 + maxInputChunkCountPerRank);
    size_t* sendIndicesPeerAllToAllvDynamicBufsMap =
        sendcountsPeerAllToAllvDynamicBufsMap + sendcountsLength;

    auto curSendIndicesLength = 1;
    auto startSendIndex = 0;
    curSendIndicesLength = inputChunkCountPerRank[sendPeerGlobal];
    for (int i = 0; i < sendPeerGlobal; i++) {
      startSendIndex += inputChunkCountPerRank[i];
    }

    // send the count to the remote process.  Each thread block sends
    // a copy of the count to the corresponding block on the peer
    // process.
    devSyncWaitStep(sync, groupIdx, CTRAN_ALGO_STEP_RESET);
    if (threadIdx.x == 0) {
      for (int i = 0; i < sendcountsLength; i++) {
        sendcountsPeerAllToAllvDynamicBufsMap[i] = sendcounts[i];
      }
      sendIndicesPeerAllToAllvDynamicBufsMap[0] = curSendIndicesLength;
      for (int i = 0; i < curSendIndicesLength; i++) {
        sendIndicesPeerAllToAllvDynamicBufsMap[i + 1] =
            inputChunkIndices[startSendIndex + i];
      }
    }
    devSyncSetStep(sync, groupIdx, 0);

    for (int i = 0; i < curSendIndicesLength; i++) {
      size_t curIndex = inputChunkIndices[startSendIndex + i];
      if (sendcounts[curIndex] > 0) {
        sendData(
            sendbuffs[curIndex],
            sendcounts[curIndex],
            sendPeer,
            localRank,
            groupIdx,
            ngroups);
      }
    }
  }
}

template <typename T>
__device__ __forceinline__ void recvImplNonContig(
    T* const* recvbuffs,
    size_t* recvCountsTmpbufGPU,
    size_t maxInputChunkCountPerRank,
    size_t sendcountsLength,
    int groupIdx,
    int ngroups,
    size_t maxRecvcount,
    bool combine) {
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  for (int r = 0; r < nLocalRanks - 1; r++) {
    auto recvPeer = (localRank + nLocalRanks - r - 1) % nLocalRanks;
    auto recvPeerGlobal = statex->localRankToRank(recvPeer);

    // get shared buffer
    int* sync =
        &(devSyncGetLoc<REMOTE>(recvPeer)->syncs[groupIdx].stepOnSameBlockIdx);
    size_t* recvcountsPeerAllToAllvDynamicBufsMap =
        reinterpret_cast<size_t*>(
            shmDevState.peerAllToAllvDynamicBufsMap[localRank]) +
        (recvPeer * ngroups + groupIdx) *
            (sendcountsLength + 1 + maxInputChunkCountPerRank);
    size_t* recvIndicesPeerAllToAllvDynamicBufsMap =
        recvcountsPeerAllToAllvDynamicBufsMap + sendcountsLength;
    auto mySendIndicesBlockLength = 0;

    // receive the count of data.  Each thread reads the count from
    // the shared buffer, but only one thread (in all blocks included)
    // writes it to the recvCountsTmpbufGPU buffer.
    devSyncWaitStep(sync, groupIdx, 0);
    mySendIndicesBlockLength = recvIndicesPeerAllToAllvDynamicBufsMap[0];
    if (threadIdx.x == 0 && groupIdx == 0 && !combine) {
      for (int i = 0; i < sendcountsLength; i++) {
        recvCountsTmpbufGPU[recvPeerGlobal * sendcountsLength + i] =
            recvcountsPeerAllToAllvDynamicBufsMap[i];
      }
    }
    devSyncSetStep(sync, groupIdx, CTRAN_ALGO_STEP_RESET);

    size_t recvOffsets = 0, lastRecvIndex = 0;
    if (combine) {
      lastRecvIndex = sendcountsLength * statex->rank() / statex->nRanks();
    }
    for (int i = 0; i < mySendIndicesBlockLength; i++) {
      size_t curRecvIndex = recvIndicesPeerAllToAllvDynamicBufsMap[i + 1];
      for (int j = lastRecvIndex; j < curRecvIndex; j++) {
        recvOffsets += recvcountsPeerAllToAllvDynamicBufsMap[j];
      }
      size_t count = recvcountsPeerAllToAllvDynamicBufsMap[curRecvIndex];

      if (count > 0) {
        T* recvPtr = recvbuffs[recvPeerGlobal] + recvOffsets;

        recvData(recvPtr, count, recvPeer, localRank, groupIdx, ngroups);
      }
      lastRecvIndex = curRecvIndex;
    }
  }
}

template <typename T>
__device__ __forceinline__ void sendImplContig(
    const T* const* sendbuffs,
    const size_t* sendcounts,
    int groupIdx,
    int ngroups) {
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  for (int r = 0; r < nLocalRanks - 1; r++) {
    auto sendPeer = (localRank + r + 1) % nLocalRanks;
    auto sendPeerGlobal = statex->localRankToRank(sendPeer);

    // get shared buffer
    int* sync =
        &(devSyncGetLoc<LOCAL>(sendPeer)->syncs[groupIdx].stepOnSameBlockIdx);
    size_t* peerAllToAllvDynamicBufsMap = reinterpret_cast<size_t*>(
        shmDevState.peerAllToAllvDynamicBufsMap[sendPeer]);

    auto count = sendcounts[sendPeerGlobal];
    const T* sendPtr = sendbuffs[sendPeerGlobal];

    // send the count to the remote process.  Each thread block sends
    // a copy of the count to the corresponding block on the peer
    // process.
    devSyncWaitStep(sync, groupIdx, CTRAN_ALGO_STEP_RESET);
    if (threadIdx.x == 0) {
      peerAllToAllvDynamicBufsMap[localRank * ngroups + groupIdx] = count;
    }
    devSyncSetStep(sync, groupIdx, 0);

    sendData(sendPtr, count, sendPeer, localRank, groupIdx, ngroups);
  }
}

template <typename T>
__device__ __forceinline__ void recvImplContig(
    T* const* recvbuffs,
    size_t* recvCountsTmpbufGPU,
    int groupIdx,
    int ngroups) {
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  for (int r = 0; r < nLocalRanks - 1; r++) {
    auto recvPeer = (localRank + nLocalRanks - r - 1) % nLocalRanks;
    auto recvPeerGlobal = statex->localRankToRank(recvPeer);

    // get shared buffer
    int* sync =
        &(devSyncGetLoc<REMOTE>(recvPeer)->syncs[groupIdx].stepOnSameBlockIdx);
    size_t* peerAllToAllvDynamicBufsMap = reinterpret_cast<size_t*>(
        shmDevState.peerAllToAllvDynamicBufsMap[localRank]);

    T* recvPtr = recvbuffs[recvPeerGlobal];
    size_t count;

    // receive the count of data.  Each thread reads the count from
    // the shared buffer, but only one thread (in all blocks included)
    // writes it to the recvCountsTmpbufGPU buffer.
    devSyncWaitStep(sync, groupIdx, 0);
    count = peerAllToAllvDynamicBufsMap[recvPeer * ngroups + groupIdx];
    if (threadIdx.x == 0 && groupIdx == 0) {
      recvCountsTmpbufGPU[recvPeerGlobal] = count;
    }
    devSyncSetStep(sync, groupIdx, CTRAN_ALGO_STEP_RESET);

    recvData(recvPtr, count, recvPeer, localRank, groupIdx, ngroups);
  }
}

template <typename T>
__device__ __forceinline__ void selfCopyNonContig(
    const T* const* sendbuffs,
    T* const* recvbuffs,
    const size_t* sendcounts,
    const size_t* inputChunkIndices,
    const size_t* inputChunkCountPerRank,
    size_t sendcountsLength,
    size_t maxInputChunkCountPerRank,
    size_t* recvCountsTmpbufGPU,
    int rank,
    int nRanks,
    int groupIdx,
    bool groupType,
    size_t maxRecvcount,
    bool combine) {
  // Now we calculate the startSendIndex on-the-fly,
  // which may not be efficient. If the inputChunkCountPerRank can be
  // on CPU, we can calculate it on CPU and pass it to GPU.
  // Or we could create a shared buffer (but will have to be per-block)
  // to store the pre-calculated startSendIndex.
  // We can optimize this if we see performance issue.
  auto startSendIndex = 0, recvOffsets = 0, curOffsetIndex = 0;

  for (int i = 0; i < rank; i++) {
    startSendIndex += inputChunkCountPerRank[i];
  }

  if (!combine && groupIdx == 0 && groupType == GROUP_RECV) {
    ctranKernCopy<size_t>(
        sendcounts,
        recvCountsTmpbufGPU + rank * sendcountsLength,
        sendcountsLength,
        0,
        1);
  }

  if (combine) {
    curOffsetIndex = sendcountsLength * rank / nRanks;
  }

  for (int i = 0; i < inputChunkCountPerRank[rank]; i++) {
    auto curSendIndex = inputChunkIndices[startSendIndex + i];

    for (int j = curOffsetIndex; j < curSendIndex; j++) {
      recvOffsets += sendcounts[j];
    }

    if (sendcounts[curSendIndex] > 0) {
      ctranKernCopy<T>(
          sendbuffs[curSendIndex],
          recvbuffs[rank] + recvOffsets,
          sendcounts[curSendIndex],
          blockIdx.x,
          gridDim.x);
    }

    curOffsetIndex = curSendIndex;
  }
}

template <typename T>
__device__ __forceinline__ void selfCopyContig(
    const T* const* sendbuffs,
    T* const* recvbuffs,
    const size_t* sendcounts,
    size_t* recvCountsTmpbufGPU,
    int rank,
    int groupIdx,
    bool groupType) {
  if (threadIdx.x == 0 && groupIdx == 0 && groupType == GROUP_RECV) {
    recvCountsTmpbufGPU[rank] = sendcounts[rank];
  }
  ctranKernCopy<T>(
      sendbuffs[rank],
      recvbuffs[rank],
      sendcounts[rank],
      blockIdx.x,
      gridDim.x);
}

// ncclKernelAllToAllvDynamic steps to handle SendCounts
// 1. Put the SendCounts locally to sendCountsTmpbufCPU and sendCountsTmpbufGPU
// 2. Put the SendCounts to local proccesses to their recvCountsTmpbufGPU
// 3. Put the SendBuf to local processes to their RecvBuf
// 4. Wait for the CPU to finish and receive the RecvCounts from remote
// processes (in different nodes)
// 5. Put the recvCountsTmpbufGPU to actualRecvcounts

template <typename T>
__device__ __forceinline__ void ncclKernelAllToAllvDynamicCommon(
    int* flag,
    CtranKernelAllToAllvDynamicArgs args,
    ALGOTYPE algoType,
    bool combine = false) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto rank = statex->rank();
  const auto nRanks = statex->nRanks();
  const auto nLocalRanks = statex->nLocalRanks();

  const T* const* sendbuffs = (algoType == DYNAMIC)
      ? (T**)args.nonSplit.sendbuffsPtrGPU
      : (T**)args.split.sendbuffsPtrShmDev;
  T* const* recvbuffs = (T**)args.recvbuffsPtrGPU;

  // Sendcounts:
  // - sendcounts is the data passed by user
  // - sendCountsTmpbufCPU is on CPU copied from sendcounts and to be used by
  // CPU
  // - sendCountsTmpbufGPU is registered GPU buffer copied from sendcounts and
  // to be used by CPU
  const size_t* sendcounts = reinterpret_cast<const size_t*>(args.sendcounts);
  const size_t sendcountsLength = args.sendcountsLength;
  size_t* sendCountsTmpbufCPU =
      reinterpret_cast<size_t*>(args.sendCountsTmpbufCPU);
  size_t* sendCountsTmpbufGPU =
      reinterpret_cast<size_t*>(args.sendCountsTmpbufGPU);
  size_t* recvCountsTmpbufGPU =
      reinterpret_cast<size_t*>(args.recvCountsTmpbufGPU);

  // only use one block to do the copy so that we can sync inside the block
  // and then signal the GPE thread
  if (blockIdx.x == 0) {
    ctranKernCopy<size_t>(
        sendcounts, sendCountsTmpbufCPU, sendcountsLength, blockIdx.x, 1);
    ctranKernCopy<size_t>(
        sendcounts, sendCountsTmpbufGPU, sendcountsLength, blockIdx.x, 1);
    __syncthreads();
    // Memory fence to ensure the writes to sendCountsTmpbufCPU are visible to
    // GPE thread
    __threadfence_system();
  }

  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = false;
  if (flag && gtIdx == 0) {
    ctran::device::KernelStartGpe(flag);
  }

  // Partition blocks into a set of send groups and a set of receive groups
  // Let even blocks handle NVL sends, and odd blocks handle NVL receives,
  // and assign groupIdx 0, 1, 2... for block{0,2,4...}@sender and
  // block{1,3,5...}@receiver. The same groupIdx on sender and receiver
  // coordinates to finish a pair of send-receive.
  const auto ngroups = gridDim.x / 2;
  const auto groupIdx = blockIdx.x / 2;
  const bool groupType = blockIdx.x % 2 == 0 ? GROUP_SEND : GROUP_RECV;

  // All blocks first involved in self D2D copy, then use NVL to exchange
  // counts/data with peers
  if (algoType == DYNAMIC_SPLIT_NON_CONTIG) {
    const size_t* inputChunkIndices = args.nonContig.inputChunkIndices;
    const size_t* inputChunkCountPerRank =
        args.nonContig.inputChunkCountPerRank;
    size_t maxInputChunkCountPerRank = args.nonContig.maxInputChunkCountPerRank;

    selfCopyNonContig(
        sendbuffs,
        recvbuffs,
        sendcounts,
        inputChunkIndices,
        inputChunkCountPerRank,
        sendcountsLength,
        maxInputChunkCountPerRank,
        recvCountsTmpbufGPU,
        rank,
        nRanks,
        groupIdx,
        groupType,
        args.nonContig.maxRecvcount,
        combine);
    if (groupType == GROUP_RECV) {
      recvImplNonContig(
          recvbuffs,
          recvCountsTmpbufGPU,
          maxInputChunkCountPerRank,
          sendcountsLength,
          groupIdx,
          ngroups,
          args.nonContig.maxRecvcount,
          combine);
    } else {
      sendImplNonContig(
          sendbuffs,
          sendcounts,
          sendcountsLength,
          inputChunkIndices,
          inputChunkCountPerRank,
          maxInputChunkCountPerRank,
          groupIdx,
          ngroups);
    }
  } else {
    selfCopyContig(
        sendbuffs,
        recvbuffs,
        sendcounts,
        recvCountsTmpbufGPU,
        rank,
        groupIdx,
        groupType);
    if (groupType == GROUP_RECV) {
      recvImplContig(recvbuffs, recvCountsTmpbufGPU, groupIdx, ngroups);
    } else {
      sendImplContig(sendbuffs, sendcounts, groupIdx, ngroups);
    }
  }

  // Use blockId = 1 (groupIdx = 0, groupTyple = GROUP_RECV) to sync with the
  // CPU.
  if (nLocalRanks != nRanks && groupIdx == 0 && groupType == GROUP_RECV) {
    bool revoked = false;
    elemWaitPostOrRevokeByGroup(args.kElem, blockIdx.x, &revoked);
    elemCompleteByGroup(args.kElem, blockIdx.x);
  }

  // Copy back to recvcounts for DYNAMIC and DYNAMIC_SPLIT
  // or if it is first a2a for DYNAMIC_SPLIT_NON_CONTIG
  if (groupIdx == 0 && groupType == GROUP_RECV &&
      (algoType != DYNAMIC_SPLIT_NON_CONTIG || !combine)) {
    ctranKernCopy<size_t>(
        recvCountsTmpbufGPU,
        reinterpret_cast<size_t*>(args.actualRecvcounts),
        ((algoType == DYNAMIC_SPLIT_NON_CONTIG) ? sendcountsLength : 1) *
            nRanks,
        0,
        1);
  }

  if (flag && groupIdx == 0 && groupType == GROUP_RECV && threadIdx.x == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

template <typename T>
__device__ __forceinline__ void generateSendbuffs(
    CtranKernelAllToAllvDynamicArgs& args,
    bool combine = false) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t* sendSplitLengths = (size_t*)args.sendcounts;
  args.split.sendbuffsPtrShmDev =
      shmDevState.alltoallvDynamicSendbuffsMap[blockIdx.x];
  T** sendbuffsGPU = (T**)args.split.sendbuffsPtrShmDev;
  T** sendbuffsCPU = (T**)args.sendbuffsPtrTmpbufCPU;

  int numCountsPerRank = int(args.sendcountsLength / statex->nRanks());

  if (threadIdx.x == 0) {
    sendbuffsGPU[0] = (T*)args.split.sendbuff;
    if (gtIdx == 0) {
      sendbuffsCPU[0] = sendbuffsGPU[0];
    }
    for (int i = 1; i < args.sendcountsLength; i++) {
      // If it is combine (2nd) a2a, we have all-gathered sendcounts
      // and hence need to reset the sendbuff offset.
      // The length of each rank is equal to maxsendcounts/ranks.
      // i / numCountsPerRank is the rank number.
      if (combine && (i % numCountsPerRank == 0)) {
        sendbuffsGPU[i] = sendbuffsGPU[0] +
            (args.nonContig.maxSendcount / statex->nRanks()) *
                (i / numCountsPerRank);
      } else {
        sendbuffsGPU[i] = sendSplitLengths[i - 1] + sendbuffsGPU[i - 1];
      }
      if (gtIdx == 0) {
        sendbuffsCPU[i] = sendbuffsGPU[i];
      }
    }
  }
  __syncthreads();
}

template <typename T>
__global__ void ncclKernelAllToAllvDynamic(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllToAllvDynamicArgs args) {
  devStateLoadToShm(devState);

  ncclKernelAllToAllvDynamicCommon<T>(flag, args, DYNAMIC);
}

template <typename T>
__global__ void ncclKernelAllToAllvDynamicSplit(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllToAllvDynamicArgs args) {
  devStateLoadToShm(devState);

  generateSendbuffs<T>(args);

  ncclKernelAllToAllvDynamicCommon<T>(flag, args, DYNAMIC_SPLIT);
}

template <typename T>
__global__ void ncclKernelAllToAllvDynamicSplitNonContig(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllToAllvDynamicArgs args) {
  devStateLoadToShm(devState);

  int totalSendIndicesLength = 0;
  for (int i = 0; i < statex->nRanks(); i++) {
    totalSendIndicesLength += args.nonContig.inputChunkCountPerRank[i];
  }

  bool combine = args.nonContig.combine;

  generateSendbuffs<T>(args, combine);

  ctranKernCopy<size_t>(
      args.nonContig.inputChunkIndices,
      reinterpret_cast<size_t*>(args.nonContig.inputChunkIndicesTmpbufCPU),
      totalSendIndicesLength,
      blockIdx.x,
      gridDim.x);
  ctranKernCopy<size_t>(
      args.nonContig.inputChunkCountPerRank,
      reinterpret_cast<size_t*>(args.nonContig.inputChunkCountPerRankTmpbufCPU),
      statex->nRanks(),
      blockIdx.x,
      gridDim.x);

  if (blockDim.x * blockIdx.x + threadIdx.x == 0) {
    for (int i = 0; i < statex->nRanks(); i++) {
      if (args.nonContig.inputChunkCountPerRank[i] >
          args.nonContig.maxInputChunkCountPerRank) {
        printf(
            "[AllToAllvDynamic Contig Kernel] The inputChunkCountPerRank %lu on rank %d to peer %d is larger than allowed (%lu)\n",
            args.nonContig.inputChunkCountPerRank,
            statex->rank(),
            i,
            args.nonContig.maxInputChunkCountPerRank);
        trap();
      }
    }
  }

  ncclKernelAllToAllvDynamicCommon<T>(
      flag, args, DYNAMIC_SPLIT_NON_CONTIG, combine);
}

#define DECL_CTRAN_ALLTOALLVDYNAMIC_KERN(T)               \
  template __global__ void ncclKernelAllToAllvDynamic<T>( \
      int* flag,                                          \
      CtranAlgoDeviceState* devState,                     \
      CtranKernelAllToAllvDynamicArgs args)

#define DECL_CTRAN_ALLTOALLVDYNAMIC_SPLIT_KERN(T)              \
  template __global__ void ncclKernelAllToAllvDynamicSplit<T>( \
      int* flag,                                               \
      CtranAlgoDeviceState* devState,                          \
      CtranKernelAllToAllvDynamicArgs args)

#define DECL_CTRAN_ALLTOALLVDYNAMIC_SPLITNONCONTIG_KERN(T)              \
  template __global__ void ncclKernelAllToAllvDynamicSplitNonContig<T>( \
      int* flag,                                                        \
      CtranAlgoDeviceState* devState,                                   \
      CtranKernelAllToAllvDynamicArgs args)
