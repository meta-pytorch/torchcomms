// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

//------------------------------------------------------------------------------
// Test Kernel
//------------------------------------------------------------------------------
template <typename T>
__global__ void
copyKernel(const T* sendbuff, T* recvbuff, size_t count, int nRuns) {
  for (int i = 0; i < nRuns; i++) {
    ctranKernCopy<T>(sendbuff, recvbuff, count, blockIdx.x, gridDim.x);
  }
}

//------------------------------------------------------------------------------
// Explicit Template Instantiations
//------------------------------------------------------------------------------
template __global__ void copyKernel<uint8_t>(
    const uint8_t* sendbuff,
    uint8_t* recvbuff,
    size_t count,
    int nRuns);

//------------------------------------------------------------------------------
// Test Kernel: Dual-Destination Copy
//------------------------------------------------------------------------------
template <typename T>
__global__ void copyKernel2Dst(
    const T* sendbuff,
    T* recvbuff1,
    T* recvbuff2,
    size_t count,
    int nRuns) {
  for (int i = 0; i < nRuns; i++) {
    ctranKernCopyMultiDest<T>(
        sendbuff, recvbuff1, recvbuff2, count, blockIdx.x, gridDim.x);
  }
}

//------------------------------------------------------------------------------
// Explicit Template Instantiations: Dual-Destination Copy
//------------------------------------------------------------------------------
template __global__ void copyKernel2Dst<uint8_t>(
    const uint8_t* sendbuff,
    uint8_t* recvbuff1,
    uint8_t* recvbuff2,
    size_t count,
    int nRuns);
