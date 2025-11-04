// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/DevShmState.cuh"
// FIXME [REBASE]: update the path once moved to fbcode/comms
#include "comms/ctran/gpe/tests/KernelElemPoolUTKernels.h"

__global__ void KElemConsumerKernel(KernelElem* elemList) {
  KernelElem* elem = elemList;
  while (elem) {
    elemFree(elem, blockIdx.x);
    elem = elem->next;
  }
}

__global__ void KElemPostRevokeKernel(KernelElem* elemList, int unuseIdx) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = false;
  KernelElem* elem = elemList;
  int i = 0;
  while (elem) {
    // Do nothing for unused elem;
    // no need host-kernel sync to mimic collective algorithm behavior (e.g.,
    // allReduceDirect)
    if (i == unuseIdx) {
      elem = elem->next;
      i++;
      continue;
    }

    bool revoked = false;
    elemWaitPostOrRevokeByGroup(elem, blockIdx.x, &revoked);

    // If revoked, free the elem
    if (revoked) {
      elemFreeByGroup(elem, blockIdx.x);
      // If posted, complete it so that host side knows kernel has consumed
    } else {
      elemCompleteByGroup(elem, blockIdx.x);
    }
    elem = elem->next;
    i++;
  }
}

__global__ void
KElemPostWaitKernel(KernelElem* elem, size_t count, int* vec1, int* vec2) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = false;
  bool revoked = false;
  elemWaitPostOrRevokeByGroup(elem, blockIdx.x, &revoked);

  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t idx = gtIdx; idx < count; idx += gridDim.x * blockDim.x) {
    int x = vec1[idx];
    int y = vec2[idx];
    vec1[idx] = x + y;
  }
  elemCompleteByGroup(elem, blockIdx.x);
}

// Post to multiple thread block groups via separate kernel element.
// Expect each group handles different subset of the data vectors independently,
// and thread blocks within each group parallize on the same subset
__global__ void KElemPostMultiGroupsKernel(
    KernelElem* elemList, // should contain nGroups number of elements
    size_t countPerGroupSet,
    int nGroupSets,
    int* vec1,
    int* vec2) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = false;
  bool revoked = false;
  auto nGroupsPerSet = gridDim.x / nGroupSets;
  auto groupSetId = blockIdx.x / nGroupsPerSet;
  auto groupId = blockIdx.x % nGroupsPerSet;

  if (threadIdx.x == 0) {
    printf(
        "blockIdx.x %d groupSetId %d/%d groupId %d nGroupsPerSet %d gridDim.x %d\n",
        blockIdx.x,
        groupSetId,
        nGroupSets,
        groupId,
        nGroupsPerSet,
        gridDim.x);
  }

  // Get the element corresponding to the groupSet
  KernelElem* elem = elemList;
  for (int i = 0; i < groupSetId; i++) {
    elem = elem->next;
  }

  elemWaitPostOrRevokeByGroup(elem, groupId, &revoked);

  // Workers in each group computes different subset of the vectors
  const auto gtIdx = blockDim.x * groupId + threadIdx.x;
  const auto gtInterval = nGroupsPerSet * blockDim.x;
  int* gVec1 = vec1 + groupSetId * countPerGroupSet;
  int* gVec2 = vec2 + groupSetId * countPerGroupSet;
  for (size_t idx = gtIdx; idx < countPerGroupSet; idx += gtInterval) {
    int x = gVec1[idx];
    int y = gVec2[idx];
    gVec1[idx] = x + y;
  }

  elemCompleteByGroup(elem, groupId);
}
