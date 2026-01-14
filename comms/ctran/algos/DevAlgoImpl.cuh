// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/algos/bcast.cuh"
#include "comms/ctran/algos/localReduce.cuh"
#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/utils/DevUtils.cuh"

/**
 * Algorithm module functions
 */

// D2D copy data between buffers within the same GPU.
template <typename T>
__device__ __forceinline__ void ctranKernCopy(
    const T* sendbuff,
    T* recvbuff,
    size_t count,
    int groupIdx,
    int ngroups) {
  if (sendbuff != recvbuff) {
    if (canCopy16(sendbuff, recvbuff, count)) {
      copy<uint4>(
          reinterpret_cast<uint4*>(recvbuff),
          reinterpret_cast<const uint4*>(sendbuff),
          count * sizeof(T) / sizeof(uint4),
          groupIdx,
          ngroups);
    } else {
      copy<T>(recvbuff, sendbuff, count, groupIdx, ngroups);
    }
  }
}

// Staging copy based send via pre-allocated IPC buffer in shmDevState.
// - If count is larger than the internal buffer, it is split into multiple
// steps.
// - In each step, the local GPU copies data from local send buffer to the
// remote GPU's IPC buffer.
// - It performs producer-consumer style sync with the remote GPU via the
// pre-allocated sync flag in shmDevState. It is used to ensure the receiver has
// copied out the data before the sender overwrites the buffer for the next
// step.
template <typename T>
__device__ __forceinline__ void ctranKernMultiStagedSend(
    const T* sendbuff,
    size_t count,
    T* buf,
    int peerLocalRank,
    size_t bufCount,
    int groupIdx,
    int ngroups) {
  size_t offset = 0;
  int step = 0;

  // Sync will be acquire loaded and release stored to ensure memory
  // consistency.
  int* sync = &(
      devSyncGetLoc<LOCAL>(peerLocalRank)->syncs[groupIdx].stepOnSameBlockIdx);

  while (offset < count) {
    size_t pendingSendCount = count - offset;
    size_t stepCount = min(bufCount, pendingSendCount);
    const T* srcPtr = sendbuff + offset;

    // Update sendPeer's devState for localRank->sendPeer NVL copy.
    // - First wait for CTRAN_ALGO_STEP_RESET ensures the completion of previous
    // step
    devSyncWaitStep(sync, groupIdx, CTRAN_ALGO_STEP_RESET);

    // - P2P from local src to remote shared region
    copy<T>(buf, srcPtr, stepCount, groupIdx, ngroups);

    // - Last update with the current step
    devSyncSetStep(sync, groupIdx, step);

    offset += stepCount;
    step++;
  }
}

// Staging copy based receive via pre-allocated IPC buffer in shmDevState.
// See function description of ctranKernMultiStagedSend.
template <typename T>
__device__ __forceinline__ void ctranKernMultiStagedRecv(
    T* recvbuff,
    size_t count,
    const T* buf,
    int peerLocalRank,
    size_t bufCount,
    int groupIdx,
    int ngroups) {
  size_t offset = 0;
  int step = 0;

  // Sync will be acquire loaded and release stored to ensure memory
  // consistency.
  int* sync = &(
      devSyncGetLoc<REMOTE>(peerLocalRank)->syncs[groupIdx].stepOnSameBlockIdx);

  while (offset < count) {
    size_t pendingRecvCount = count - offset;
    size_t stepCount = min(bufCount, pendingRecvCount);
    T* dstPtr = recvbuff + offset;

    // Watch local devState for recvPeer->local NVL copy.
    // - First wait for the current step
    devSyncWaitStep(sync, groupIdx, step);

    // - D2D from local shared region to local dst
    copy<T>(dstPtr, buf, stepCount, groupIdx, ngroups);

    // - Last reset with CTRAN_ALGO_STEP_RESET to indicate the step has been
    // acked so that recvPeer can update the next step
    devSyncSetStep(sync, groupIdx, CTRAN_ALGO_STEP_RESET);
    offset += stepCount;
    step++;
  }
}

template <bool Complete, bool Free>
__device__ __forceinline__ void ctranKernMultiPutNotify(KernelElem* elemList) {
  const auto groupIdx = blockIdx.x;

  // Traverse each element, handle posted put one by one; exit once
  // completed all
  KernelElem* elem = elemList;
  while (elem != nullptr) {
    bool revoked = false;
    uint64_t recvbuffAddr =
        elemWaitPostOrRevokeByGroupForMultiPut(elem, groupIdx, &revoked);

    // Skip if entire elem has revoked (e.g., GPE thread found remote buffer
    // not allocated by cumem) or the current put is small and doesn't need use
    // all groups
    if (revoked || groupIdx >= elem->putNotify.ngroups) {
      elem = elem->next;
      continue;
    }

    const char* sendbuff =
        reinterpret_cast<const char*>(elem->putNotify.sendbuff);
    char* recvbuff = reinterpret_cast<char*>(recvbuffAddr);
    size_t nbytes = elem->putNotify.nbytes;

    if (threadIdx.x == 0) {
      CTRAN_DEV_TRACE(
          "copy elem %p %p->%p len %ld notify %d\n",
          elem,
          sendbuff,
          recvbuff,
          nbytes,
          elem->putNotify.notify);
    }

    // Copy - P2P from local buffer to remote buffer
    if (sendbuff != recvbuff && nbytes > 0) {
      if (canCopy16(sendbuff, recvbuff, nbytes)) {
        copy<uint4>(
            reinterpret_cast<uint4*>(recvbuff),
            reinterpret_cast<const uint4*>(sendbuff),
            nbytes / sizeof(uint4),
            groupIdx,
            elem->putNotify.ngroups);
      } else {
        copy<char>(
            recvbuff, sendbuff, nbytes, groupIdx, elem->putNotify.ngroups);
      }
    }

    // Optionally notify remote peer
    if (elem->putNotify.notify) {
      CtranAlgoDeviceSync* sync =
          devSyncGetLoc<REMOTE>(elem->putNotify.peerLocalRank);
      devSyncSetNotify(sync, groupIdx);
    }

    // Optionally mark completed per element to sync with GPE thread
    if (Complete) {
      elemCompleteByGroup(elem, groupIdx);
    }
    elem = elem->next;
  }

  // Free elements for host pool to reclaim.
  // If Free == false, free only revoked elements.
  elemsFreeListByGroup(elemList, groupIdx, Free);
}

template <bool Complete, bool Free>
__device__ __forceinline__ void ctranKernMultiNotifyOnly(KernelElem* elemList) {
  const auto groupIdx = blockIdx.x;

  // Traverse each element, handle posted put one by one; exit once
  // completed all
  KernelElem* elem = elemList;
  while (elem != nullptr) {
    // Optionally notify remote peer
    if (elem->putNotify.notify) {
      CtranAlgoDeviceSync* sync =
          devSyncGetLoc<REMOTE>(elem->putNotify.peerLocalRank);
      devSyncSetNotify(sync, groupIdx);
    }
    // Optionally mark completed per element to sync with GPE thread
    if (Complete) {
      elemCompleteByGroup(elem, groupIdx);
    }

    elem = elem->next;
  }

  // Free elements for host pool to reclaim.
  // If Free == false, free only revoked elements.
  elemsFreeListByGroup(elemList, groupIdx, Free);
}

template <int Complete, int Free>
__device__ __forceinline__ void ctranKernMultiWaitNotifyOnly(
    KernelElem* elemList) {
  const auto groupIdx = blockIdx.x;

  // Traverse each element, wait remote notification one by one; exit till
  // completed all
  KernelElem* elem = elemList;
  while (elem != nullptr) {
    CtranAlgoDeviceSync* sync =
        devSyncGetLoc<LOCAL>(elem->waitNotify.peerLocalRank);
    devSyncWaitNotify(sync, elem->waitNotify.ngroups);

    // Optionally mark completed per element to sync with GPE thread
    if (Complete) {
      elemCompleteByGroup(elem, groupIdx);
    }

    elem = elem->next;
  }

  // Free elements for host pool to reclaim.
  // If Free == false, free only revoked elements.
  elemsFreeListByGroup(elemList, groupIdx, Free);
}

template <int Complete, int Free>
__device__ __forceinline__ void ctranKernMultiWaitNotify(KernelElem* elemList) {
  // waitNotify should use only 1 thread block
  const auto groupIdx = blockIdx.x;
  if (groupIdx != 0) {
    return;
  }

  // Traverse each element, wait remote notification one by one; exit till
  // completed all
  KernelElem* elem = elemList;
  while (elem != nullptr) {
    bool revoked = false;
    elemWaitPostOrRevokeByGroup(elem, groupIdx, &revoked);

    // Skip it if has revoked
    if (revoked) {
      elem = elem->next;
      continue;
    }

    if (threadIdx.x == 0) {
      CTRAN_DEV_TRACE(
          "waitNotify elem %p peerLocalRank=%d ngroups=%d\n",
          elem,
          elem->waitNotify.peerLocalRank,
          elem->waitNotify.ngroups);
    }

    CtranAlgoDeviceSync* sync =
        devSyncGetLoc<LOCAL>(elem->waitNotify.peerLocalRank);
    devSyncWaitNotify(sync, elem->waitNotify.ngroups);

    // Optionally mark completed per element to sync with GPE thread
    if (Complete) {
      elemCompleteByGroup(elem, groupIdx);
    }
    elem = elem->next;
  }

  // Free elements for host pool to reclaim.
  // If Free == false, free only revoked elements.
  elemsFreeListByGroup(elemList, groupIdx, Free);
}

template <typename T, commRedOp_t RedOp>
__device__ __forceinline__ void ctranKernReduce(
    CtranAlgoDevReduceArg& redArg,
    KernelElem* elemOnHost,
    int stepId = 0) {
  const T* srcs[CTRAN_MAX_NVL_PEERS];
  T* dsts[CTRAN_MAX_NVL_PEERS];
  for (int i = 0; i < redArg.nsrcs; i++) {
    srcs[i] = reinterpret_cast<const T*>(redArg.srcs[i]);
  }
  for (int i = 0; i < redArg.ndsts; i++) {
    dsts[i] = reinterpret_cast<T*>(redArg.dsts[i]);
  }

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "reduce elem %p (redArg %p) nsrcs %ld ndsts %ld count %ld at step %d flushMem %d isFinal %d barrier %d\n",
        elemOnHost,
        &redArg,
        redArg.nsrcs,
        redArg.ndsts,
        redArg.count,
        stepId,
        redArg.flushMem,
        redArg.isFinal,
        redArg.barrier);
    for (int i = 0; i < redArg.nsrcs; i++) {
      CTRAN_DEV_TRACE("  srcs[%d] %p\n", i, srcs[i]);
    }
    for (int i = 0; i < redArg.ndsts; i++) {
      CTRAN_DEV_TRACE("  dsts[%d] %p\n", i, dsts[i]);
    }
  }

  if (RedOp == commAvg) {
    // If this is the last round of last reduce for a given data segment,
    // now dst should have final sum. Thus, apply average. Note that
    // we intentionally embed average into the last reduce with exactlly
    // same partition, thus it avoids cross thread-block sync.
    if (redArg.isFinal) {
      const auto nRanks = statex->nRanks();
      localReduce<T, RedOp>(
          redArg.nsrcs, srcs, redArg.ndsts, dsts, redArg.count, nRanks);
    } else {
      // For intermediate steps, sum for AVG
      localReduce<T, commSum>(
          redArg.nsrcs, srcs, redArg.ndsts, dsts, redArg.count);
    }
  } else {
    localReduce<T, RedOp>(redArg.nsrcs, srcs, redArg.ndsts, dsts, redArg.count);
  }

  if (redArg.flushMem) {
    // Flush memory so that data becomes visible to peer GPUs and NIC
    __threadfence_system();
  }

  // Perform a barrier with nvectors of peers
  if (redArg.barrier) {
    barrier(statex->localRank(), redArg.nsrcs);
  }
}

template <typename T, commRedOp_t RedOp, bool Complete, bool Free>
__device__ __forceinline__ void ctranKernMultiReduce(
    KernelElem* elemList,
    int stepId = 0) {
  const auto groupIdx = blockIdx.x;

  // Traverse each element, handle posted put one by one; exit till
  // completed all
  KernelElem* elem = elemList;
  while (elem != nullptr) {
    bool revoked = false;
    elemWaitPostOrRevokeByGroup(elem, groupIdx, &revoked);

    // Skip if entire elem has revoked
    if (revoked) {
      elem = elem->next;
      continue;
    }

    // Load reduce argument from host pinned memory
    CtranAlgoDevReduceArg redArg;
    // FIXME: sizeof(CtranAlgoDevReduceArg) is 1.2KB, load from host-pinned can
    // cause 40us overhead
    loadAlgoDevArg<CtranAlgoDevReduceArg>(redArg, &elem->reduce);

    // Perform reduce op with loaded local argument
    ctranKernReduce<T, RedOp>(redArg, elem, stepId);

    // Optionally mark completed per element to sync with GPE thread
    if (Complete) {
      // Only thread 0 to label the finished step and the complete check will
      // ensure all groups have finished when host side sees the completion with
      // the stepId.
      if (threadIdx.x == 0) {
        elem->stepDone = stepId;
      }
      elemCompleteByGroup(elem, groupIdx);
    }
    elem = elem->next;
  }

  // Free elements for host pool to reclaim.
  // If Free == false, free only revoked elements.
  elemsFreeListByGroup(elemList, groupIdx, Free);
}

template <typename T, commRedOp_t RedOp, bool Complete, bool Free>
__device__ __forceinline__ void ctranKernMultiStridedReduce(
    KernelElem* elemList,
    bool finalReduce,
    int stepId = 0) {
  const auto groupIdx = blockIdx.x;

  // Traverse each element, handle posted put one by one; exit till
  // completed all
  KernelElem* elem = elemList;
  while (elem != nullptr) {
    bool revoked = false;
    elemWaitPostOrRevokeByGroup(elem, groupIdx, &revoked);

    // Skip if entire elem has revoked
    if (revoked) {
      elem = elem->next;
      continue;
    }

    if (threadIdx.x == 0) {
      CTRAN_DEV_TRACE(
          "stridedReduce elem %p dst %p srcs %p numBlocks %d stride %ld, blockCount %ld inplaceBlockIdx %d at step %d, finalReduce %d\n",
          elem,
          elem->stridedReduce.dst,
          elem->stridedReduce.stridedSrc,
          elem->stridedReduce.numBlocks,
          elem->stridedReduce.stride,
          elem->stridedReduce.blockCount,
          elem->stridedReduce.inplaceBlockIdx,
          stepId,
          finalReduce);
    }

    const T* srcVec[CTRAN_MAX_NVL_PEERS];
    T* dst = reinterpret_cast<T*>(elem->stridedReduce.dst);
    T* stridedSrc = reinterpret_cast<T*>(elem->stridedReduce.stridedSrc);
    int numBlocks = elem->stridedReduce.numBlocks;
    size_t stride = elem->stridedReduce.stride;
    int inplaceBlockIdx = elem->stridedReduce.inplaceBlockIdx;
    const auto nRanks = statex->nRanks();

    // Need multiple rounds of local reduce if total num of blocks is more than
    // CTRAN_MAX_NVL_PEERS - 1.
    int vecIdx = 0;
    int srcsIdx = 0;
    while (srcsIdx < numBlocks) {
      // Special handling for inplace block: assign dst as src if set. Default
      // -1 to skip. Note that we need be careful not overwriting dst before it
      // is reduced. See redDst below.
      if (inplaceBlockIdx == srcsIdx) {
        srcVec[vecIdx] = dst;
      } else {
        size_t offset = srcsIdx * stride; // element-wise offset
        srcVec[vecIdx] = stridedSrc + offset;
      }
      vecIdx++;
      srcsIdx++;

      if (vecIdx == CTRAN_MAX_NVL_PEERS || srcsIdx == numBlocks) {
        T* redDst = dst;
        if (inplaceBlockIdx > 0 && srcsIdx - 1 < inplaceBlockIdx) {
          // If not yet arrive dst's index, use the unused block in
          // stridedSrc to avoid overwriting dst's value before it is
          // reduced
          redDst = stridedSrc + inplaceBlockIdx * stride;
        }
        if (RedOp == commAvg) {
          // If this is the last round of last reduce for a given data segment,
          // now redDst should have final sum. Thus, apply average. Note that
          // we intentionally embed average into the last reduce with exactlly
          // same partition, thus it avoids cross thread-block sync.
          if (finalReduce && srcsIdx == numBlocks) {
            localReduce<T, RedOp>(
                vecIdx, srcVec, redDst, elem->stridedReduce.blockCount, nRanks);
          } else {
            localReduce<T, commSum>(
                vecIdx, srcVec, redDst, elem->stridedReduce.blockCount);
          }
        } else {
          localReduce<T, RedOp>(
              vecIdx, srcVec, redDst, elem->stridedReduce.blockCount);
        }

        // Reset for next round; reserve vec[0] for intermediate result from the
        // finished round
        srcVec[0] = redDst;
        vecIdx = 1;
      }
    }

    if (elem->stridedReduce.flushMem) {
      // Flush memory so that data becomes visible to peer GPUs and NIC
      __threadfence_system();
    }

    // Optionally mark completed per element to sync with GPE thread
    if (Complete) {
      // Only thread 0 to label the finished step and the complete check will
      // ensure all groups have finished when host side sees the completion with
      // the stepId.
      if (threadIdx.x == 0) {
        elem->stepDone = stepId;
      }
      elemCompleteByGroup(elem, groupIdx);
    }
    elem = elem->next;
  }

  // Free elements for host pool to reclaim.
  // If Free == false, free only revoked elements.
  elemsFreeListByGroup(elemList, groupIdx, Free);
}

using BcastAlgo = enum { kDefaultBcast, kMultiPutBcast };

template <typename T, BcastAlgo Algo>
__device__ __forceinline__ void ctranKernBcast(
    CtranAlgoDevBcastArg& bcastArg,
    KernelElem* elemOnHost,
    const int groupIdx,
    const int nGroups,
    int stepId = 0) {
  T* dsts[CTRAN_MAX_NVL_PEERS];
  const T* src = (T*)bcastArg.src;
  for (int i = 0; i < bcastArg.nvectors; i++) {
    dsts[i] = reinterpret_cast<T*>(bcastArg.dsts[i]);
  }

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE(
        "%s bcast elem %p (redArg %p) nvectors %ld src %p count %ld at step %d flushMem %d barrier %d\n",
        Algo == kDefaultBcast ? "default" : "multiPut",
        elemOnHost,
        &bcastArg,
        bcastArg.nvectors,
        bcastArg.src,
        bcastArg.count,
        stepId,
        bcastArg.flushMem,
        bcastArg.barrier);
    for (int i = 0; i < bcastArg.nvectors; i++) {
      CTRAN_DEV_TRACE("  dsts[%d] %p\n", i, dsts[i]);
    }
  }

  if (Algo == kDefaultBcast) {
    bcast<T>(bcastArg.nvectors, src, dsts, bcastArg.count, groupIdx, nGroups);
  } else {
    multiPutBcast<T>(
        bcastArg.nvectors, src, dsts, bcastArg.count, groupIdx, nGroups);
  }

  if (bcastArg.flushMem) {
    // Flush memory so that data becomes visible to peer GPUs and NIC
    __threadfence_system();
  }

  // Perform a barrier with nvectors of peers
  if (bcastArg.barrier) {
    barrier(statex->localRank(), bcastArg.nvectors);
  }
}
