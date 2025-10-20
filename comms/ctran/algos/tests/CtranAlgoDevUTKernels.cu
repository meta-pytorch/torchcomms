// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/localReduce.cuh"
#include "comms/ctran/algos/tests/CtranAlgoDevUTKernels.cuh"

using namespace ctran;

template <bool Check>
__global__ void devStateLoadToShmTestKernel(
    CtranAlgoDeviceState* devStateIn,
    CtranAlgoDeviceState* devStateOut) {
  devStateLoadToShm(devStateIn);

  // Optionally copy out to check if the above function is correct
  if (threadIdx.x == 0 && Check) {
    memcpy(devStateOut, &shmDevState, sizeof(CtranAlgoDeviceState));
  }
}

// global function to wrap up device function we want to test
template <typename T>
__global__ void __launch_bounds__(1024, 1) testCtranLocalReduce(
    size_t nsrcs,
    const T** srcs,
    size_t ndsts,
    T** dst,
    size_t count,
    commRedOp_t redOp,
    int nranks) {
  if (redOp == commSum) {
    localReduce<T, commSum>(nsrcs, srcs, ndsts, dst, count, nranks);
  } else if (redOp == commMax) {
    localReduce<T, commMax>(nsrcs, srcs, ndsts, dst, count, nranks);
  } else if (redOp == commAvg) {
    localReduce<T, commAvg>(nsrcs, srcs, ndsts, dst, count, nranks);
  } else if (redOp == commMin) {
    localReduce<T, commMin>(nsrcs, srcs, ndsts, dst, count, nranks);
  } else if (redOp == commProd) {
    localReduce<T, commProd>(nsrcs, srcs, ndsts, dst, count, nranks);
  }
}

template __global__ void devStateLoadToShmTestKernel<true>(
    CtranAlgoDeviceState* devStateIn,
    CtranAlgoDeviceState* devStateOut);
template __global__ void devStateLoadToShmTestKernel<false>(
    CtranAlgoDeviceState* devStateIn,
    CtranAlgoDeviceState* devStateOut);

#define DECL_LOCALREDUCE_KERN(T)                    \
  template __global__ void testCtranLocalReduce<T>( \
      size_t nsrcs,                                 \
      const T** srcs,                               \
      size_t ndsts,                                 \
      T** dst,                                      \
      size_t count,                                 \
      commRedOp_t redOp,                            \
      int nranks)

DECL_LOCALREDUCE_KERN(char);
DECL_LOCALREDUCE_KERN(int8_t);
DECL_LOCALREDUCE_KERN(uint8_t);
DECL_LOCALREDUCE_KERN(int32_t);
DECL_LOCALREDUCE_KERN(uint32_t);
DECL_LOCALREDUCE_KERN(int64_t);
DECL_LOCALREDUCE_KERN(uint64_t);
DECL_LOCALREDUCE_KERN(half);
DECL_LOCALREDUCE_KERN(float);
DECL_LOCALREDUCE_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_LOCALREDUCE_KERN(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_LOCALREDUCE_KERN(__nv_fp8_e4m3);
DECL_LOCALREDUCE_KERN(__nv_fp8_e5m2);
#endif

// global function to wrap up device function we want to test
template <typename T>
__global__ void __launch_bounds__(1024, 1)
    testCtranLocalReduceSubsetThreadBlocks(
        size_t nsrcs,
        const T** srcs,
        size_t ndsts,
        T** dst,
        size_t count,
        commRedOp_t redOp,
        int nranks) {
  // Only the last n-1 thread blocks will do the reduction
  if (blockIdx.x == 0) {
    return;
  }
  const auto workerId = blockIdx.x - 1;
  const auto numWorkers = gridDim.x - 1;

  if (redOp == commSum) {
    localReduce<T, commSum>(
        nsrcs, srcs, ndsts, dst, count, workerId, numWorkers, nranks);
  } else if (redOp == commMax) {
    localReduce<T, commMax>(
        nsrcs, srcs, ndsts, dst, count, workerId, numWorkers, nranks);
  } else if (redOp == commAvg) {
    localReduce<T, commAvg>(
        nsrcs, srcs, ndsts, dst, count, workerId, numWorkers, nranks);
  } else if (redOp == commMin) {
    localReduce<T, commMin>(
        nsrcs, srcs, ndsts, dst, count, workerId, numWorkers, nranks);
  } else if (redOp == commProd) {
    localReduce<T, commProd>(
        nsrcs, srcs, ndsts, dst, count, workerId, numWorkers, nranks);
  }
}

#define DECL_LOCALREDUCESUBSET_KERN(T)                                \
  template __global__ void testCtranLocalReduceSubsetThreadBlocks<T>( \
      size_t nsrcs,                                                   \
      const T** srcs,                                                 \
      size_t ndsts,                                                   \
      T** dst,                                                        \
      size_t count,                                                   \
      commRedOp_t redOp,                                              \
      int nranks)

DECL_LOCALREDUCESUBSET_KERN(char);
DECL_LOCALREDUCESUBSET_KERN(int8_t);
DECL_LOCALREDUCESUBSET_KERN(uint8_t);
DECL_LOCALREDUCESUBSET_KERN(int32_t);
DECL_LOCALREDUCESUBSET_KERN(uint32_t);
DECL_LOCALREDUCESUBSET_KERN(int64_t);
DECL_LOCALREDUCESUBSET_KERN(uint64_t);
DECL_LOCALREDUCESUBSET_KERN(half);
DECL_LOCALREDUCESUBSET_KERN(float);
DECL_LOCALREDUCESUBSET_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_LOCALREDUCESUBSET_KERN(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_LOCALREDUCESUBSET_KERN(__nv_fp8_e4m3);
DECL_LOCALREDUCESUBSET_KERN(__nv_fp8_e5m2);
#endif

// global function to wrap up device function we want to test
template <typename T, typename RedT>
__global__ void __launch_bounds__(1024, 1) testDequantizedAllToAllLocalReduce(
    const T* src,
    T* dst,
    size_t count,
    int myRank,
    size_t nRanks,
    commRedOp_t redOp) {
  if (redOp == commSum) {
    localReduceForDequantAllToAll<T, RedT, commSum>(
        src, dst, count, myRank, nRanks);
  } else if (redOp == commMax) {
    localReduceForDequantAllToAll<T, RedT, commMax>(
        src, dst, count, myRank, nRanks);
  } else if (redOp == commAvg) {
    localReduceForDequantAllToAll<T, RedT, commAvg>(
        src, dst, count, myRank, nRanks);
  } else if (redOp == commMin) {
    localReduceForDequantAllToAll<T, RedT, commMin>(
        src, dst, count, myRank, nRanks);
  } else if (redOp == commProd) {
    localReduceForDequantAllToAll<T, RedT, commProd>(
        src, dst, count, myRank, nRanks);
  }
}

#define DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE(T, RedT)                  \
  template __global__ void testDequantizedAllToAllLocalReduce<T, RedT>( \
      const T* src,                                                     \
      T* dst,                                                           \
      size_t count,                                                     \
      int myRank,                                                       \
      size_t nRanks,                                                    \
      commRedOp_t redOp)

#define DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(T) \
  DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE(T, T)

DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(char);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(int8_t);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(uint8_t);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(int32_t);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(uint32_t);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(int64_t);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(uint64_t);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(half);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(float);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(__nv_fp8_e4m3);
DECL_DEQUANTIZED_ALLTOALL_LOCALREDUCE_SELF(__nv_fp8_e5m2);
#endif
