// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/localReduce.cuh"
#include "comms/ctran/algos/tests/CtranDistAlgoDevUTKernels.h"

// Helper macro for launching kernels with dynamic shared memory
#define CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_RETURN(                           \
    kernel, grid, block, args, dynamicShmSize, stream)                        \
  [&]() -> cudaError_t {                                                      \
    cudaError_t res = cudaFuncSetAttribute(                                   \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicShmSize); \
    if (res != cudaSuccess)                                                   \
      return res;                                                             \
    return cudaLaunchKernel(                                                  \
        kernel, grid, block, args, dynamicShmSize, stream);                   \
  }()

template <bool Complete, bool Free>
__global__ void __launch_bounds__(1024, 1) testKernMultiPutNotify(
    KernelElem* elemList,
    CtranAlgoDeviceState* devState) {
  devStateLoadToShm(devState);
  ctranKernMultiPutNotify<Complete, Free>(elemList);
}

template <bool Complete, bool Free>
__global__ void __launch_bounds__(1024, 1) testKernMultiWaitNotify(
    KernelElem* elemList,
    CtranAlgoDeviceState* devState) {
  devStateLoadToShm(devState);
  ctranKernMultiWaitNotify<Complete, Free>(elemList);
}

template <bool Complete, bool Free, typename T, commRedOp_t redOp>
__global__ void __launch_bounds__(1024, 1) testKernMultiReduce(
    KernelElem* elemList,
    int nSteps,
    CtranAlgoDeviceState* devState) {
  devStateLoadToShm(devState);
  for (int i = 0; i < nSteps; i++) {
    ctranKernMultiReduce<T, redOp, Complete, Free>(elemList, i);
  }
}

template <typename T, commRedOp_t redOp>
__global__ void __launch_bounds__(1024, 1) testKernReduce(
    KernelElem* elemOnHost,
    CtranAlgoDeviceState* devState,
    int stepId = 0) {
  devStateLoadToShm(devState);
  KernelElem* elem = elemOnHost;
  while (elem != nullptr) {
    // Load reduce argument from host pinned memory
    CtranAlgoDevReduceArg redArg;
    loadAlgoDevArg<CtranAlgoDevReduceArg>(redArg, &elem->reduce);

    ctranKernReduce<T, redOp>(redArg, elemOnHost, stepId);
    elem = elem->next;
  }
}

template <typename T, typename RedT, commRedOp_t redOp>
__global__ void __launch_bounds__(1024, 1) testDequantizedAllToAllKernReduce(
    const T* sendbuff,
    T* recvbuff,
    size_t count,
    CtranAlgoDeviceState* devState) {
  devStateLoadToShm(devState);

  localReduceForDequantAllToAll<T, RedT, redOp>(
      sendbuff, recvbuff, count, statex->rank(), statex->nRanks());
}

/*
This kernel serves two primary use cases:
1) All blocks do a single bcast: In this scenario, the caller sets nBcastBlocks
equal to nGroups 2) Only `nBcastBlocks` blocks do each bcast: nBcastBlocks is
less than nGroups, the caller can specify different values for nBcastBlocks.
less than nGroups, the caller can specify bcast data for each bcastGroup.
*/
template <bool Complete, bool Free, typename T, BcastAlgo algo>
__global__ void __launch_bounds__(256, 1) testKernBcast(
    KernelElem* elemH,
    int nSteps,
    int nBcastBlocks,
    CtranAlgoDeviceState* devState) {
  devStateLoadToShm(devState);
  const auto groupIdx = blockIdx.x % nBcastBlocks;
  const auto bcastGroup = blockIdx.x / nBcastBlocks;

  // get bcastElem for bcastGroup
  for (int i = 0; i < bcastGroup; i++) {
    elemH = elemH->next;
  }

  for (int i = 0; i < nSteps; i++) {
    bool revoked = false;
    elemWaitPostOrRevokeByGroup(elemH, groupIdx, &revoked);

    // Skip if entire elem has revoked
    if (revoked) {
      continue;
    }

    // Load reduce argument from host pinned memory
    CtranAlgoDevBcastArg bArg;
    loadAlgoDevArg<CtranAlgoDevBcastArg>(bArg, &elemH->bcast);

    ctranKernBcast<T, algo>(bArg, elemH, groupIdx, nBcastBlocks, i);

    if (Complete) {
      if (threadIdx.x == 0) {
        elemH->stepDone = i;
      }

      elemCompleteByGroup(elemH, groupIdx);
    }
  }
  elemsFreeListByGroup(elemH, groupIdx, Free);
}

template <typename T>
__global__ void __launch_bounds__(512, 1) testKernCopy(
    const T* sendbuff,
    T* recvbuff,
    size_t count,
    int iters,
    CtranAlgoDeviceState* devState) {
  for (int i = 0; i < iters; i++) {
    ctranKernCopy<T>(sendbuff, recvbuff, count, blockIdx.x, gridDim.x);
  }
}

template <typename T>
__global__ void __launch_bounds__(512, 1) testNaiveKernCopy(
    const T* sendbuff,
    T* recvbuff,
    size_t count,
    int iters,
    CtranAlgoDeviceState* devState) {
  for (int i = 0; i < iters; i++) {
    for (auto j = blockIdx.x * blockDim.x + threadIdx.x; j < count;
         j += gridDim.x * blockDim.x) {
      recvbuff[j] = sendbuff[j];
    }
  }
}

cudaError_t testKernMultiPutNotifyWrapper(
    ElemTestType testType,
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream,
    KernelElem* elemList,
    CtranAlgoDeviceState* devState) {
  void* args[] = {&elemList, &devState};
  void* fn = nullptr;
  if (testType == kTestElemComplete) {
    fn = reinterpret_cast<void*>(testKernMultiPutNotify<true, false>);
  } else {
    fn = reinterpret_cast<void*>(testKernMultiPutNotify<false, true>);
  }

  return CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_RETURN(
      fn, grid, blocks, args, sizeof(CtranAlgoDeviceState), stream);
}

cudaError_t testKernMultiWaitNotifyWrapper(
    ElemTestType testType,
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream,
    KernelElem* elemList,
    CtranAlgoDeviceState* devState) {
  void* args[] = {&elemList, &devState};
  void* fn = nullptr;
  if (testType == kTestElemComplete) {
    fn = reinterpret_cast<void*>(testKernMultiWaitNotify<true, false>);
  } else {
    fn = reinterpret_cast<void*>(testKernMultiWaitNotify<false, true>);
  }

  return CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_RETURN(
      fn, grid, blocks, args, sizeof(CtranAlgoDeviceState), stream);
}

cudaError_t testKernMultiReduceWrapper(
    ElemTestType testType,
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream,
    KernelElem* elemList,
    int nSteps,
    CtranAlgoDeviceState* devState) {
  void* args[] = {&elemList, &nSteps, &devState};
  void* fn = nullptr;
  if (testType == kTestElemComplete || testType == kTestElemRepost) {
    fn =
        reinterpret_cast<void*>(testKernMultiReduce<true, false, int, commSum>);
  } else {
    fn =
        reinterpret_cast<void*>(testKernMultiReduce<false, true, int, commSum>);
  }

  return CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_RETURN(
      fn, grid, blocks, args, sizeof(CtranAlgoDeviceState), stream);
}

cudaError_t testKernBcastWrapper(
    BcastTestAlgo testAlgo,
    ElemTestType testType,
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream,
    KernelElem* elem,
    int nSteps,
    int nBcastBlocks,
    CtranAlgoDeviceState* devState) {
  void* args[] = {&elem, &nSteps, &nBcastBlocks, &devState};
  void* fn = nullptr;

  if (testType == kTestElemComplete || testType == kTestElemRepost) {
    if (testAlgo == kTestDefaultBcast) {
      fn = reinterpret_cast<void*>(
          testKernBcast<true, false, int, kDefaultBcast>);
    } else {
      fn = reinterpret_cast<void*>(
          testKernBcast<true, false, int, kMultiPutBcast>);
    }
  } else {
    if (testAlgo == kTestDefaultBcast) {
      fn = reinterpret_cast<void*>(
          testKernBcast<false, true, int, kDefaultBcast>);
    } else {
      fn = reinterpret_cast<void*>(
          testKernBcast<false, true, int, kMultiPutBcast>);
    }
  }

  return CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_RETURN(
      fn, grid, blocks, args, sizeof(CtranAlgoDeviceState), stream);
}

template <typename T>
void* getMultiReduceKernelFn(ElemTestType testType, commRedOp_t op) {
  if (testType == kTestElemComplete || testType == kTestElemRepost) {
    switch (op) {
      case commProd:
        return reinterpret_cast<void*>(testKernMultiReduce<
                                       true /*Complete*/,
                                       false /*Free*/,
                                       T,
                                       commProd>);
      case commMax:
        return reinterpret_cast<void*>(
            testKernMultiReduce<true /*Complete*/, false /*Free*/, T, commMax>);
      case commMin:
        return reinterpret_cast<void*>(
            testKernMultiReduce<true /*Complete*/, false /*Free*/, T, commMin>);
      case commAvg:
        return reinterpret_cast<void*>(
            testKernMultiReduce<true /*Complete*/, false /*Free*/, T, commAvg>);
      case commSum:
      default:
        return reinterpret_cast<void*>(
            testKernMultiReduce<true /*Complete*/, false /*Free*/, T, commSum>);
    }
  } else {
    switch (op) {
      case commProd:
        return reinterpret_cast<void*>(testKernMultiReduce<
                                       false /*Complete*/,
                                       true /*Free*/,
                                       T,
                                       commProd>);
      case commMax:
        return reinterpret_cast<void*>(
            testKernMultiReduce<false /*Complete*/, true /*Free*/, T, commMax>);
      case commMin:
        return reinterpret_cast<void*>(
            testKernMultiReduce<false /*Complete*/, true /*Free*/, T, commMin>);
      case commAvg:
        return reinterpret_cast<void*>(
            testKernMultiReduce<false /*Complete*/, true /*Free*/, T, commAvg>);
      case commSum:
      default:
        return reinterpret_cast<void*>(
            testKernMultiReduce<false /*Complete*/, true /*Free*/, T, commSum>);
    }
  }
  return nullptr;
}

template <typename T>
void* getReduceKernelFn(commRedOp_t op) {
  void* fn = nullptr;
  switch (op) {
    case commProd:
      return reinterpret_cast<void*>(testKernReduce<T, commProd>);
    case commMax:
      return reinterpret_cast<void*>(testKernReduce<T, commMax>);
    case commMin:
      return reinterpret_cast<void*>(testKernReduce<T, commMin>);
    case commAvg:
      return reinterpret_cast<void*>(testKernReduce<T, commAvg>);
    case commSum:
    default:
      return reinterpret_cast<void*>(testKernReduce<T, commSum>);
  }
}

template <typename T, typename RedT>
void* getDequantizedAllToAllReduceKernelFn(commRedOp_t op) {
  switch (op) {
    case commProd:
      return reinterpret_cast<void*>(
          testDequantizedAllToAllKernReduce<T, RedT, commProd>);
    case commMax:
      return reinterpret_cast<void*>(
          testDequantizedAllToAllKernReduce<T, RedT, commMax>);
    case commMin:
      return reinterpret_cast<void*>(
          testDequantizedAllToAllKernReduce<T, RedT, commMin>);
    case commAvg:
      return reinterpret_cast<void*>(
          testDequantizedAllToAllKernReduce<T, RedT, commSum>);
    default:
      return reinterpret_cast<void*>(
          testDequantizedAllToAllKernReduce<T, RedT, commSum>);
  }
}

template <typename T>
void* getKernCopyFn() {
  return reinterpret_cast<void*>(testKernCopy<T>);
}

template <typename T>
void* getNaiveKernCopyFn() {
  return reinterpret_cast<void*>(testNaiveKernCopy<T>);
}

template void* getReduceKernelFn<int>(commRedOp_t op);
template void* getMultiReduceKernelFn<int>(
    ElemTestType testType,
    commRedOp_t op);
template void* getDequantizedAllToAllReduceKernelFn<int, int>(commRedOp_t op);
template void* getKernCopyFn<int>();
template void* getNaiveKernCopyFn<int>();
