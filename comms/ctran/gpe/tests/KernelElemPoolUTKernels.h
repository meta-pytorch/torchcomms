// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/gpe/CtranGpeDev.h"

__global__ void KElemConsumerKernel(KernelElem* elemList);

__global__ void KElemPostRevokeKernel(KernelElem* elemList, int unuseIdx);

__global__ void
KElemPostWaitKernel(KernelElem* elem, size_t count, int* vec1, int* vec2);

__global__ void KElemPostMultiGroupsKernel(
    KernelElem* elemList, // should contain nGroups number of elements
    size_t countPerGroupSet,
    int nGroupSets,
    int* vec1,
    int* vec2);
