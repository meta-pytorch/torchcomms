// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/CtranAlgoDev.h"

template <bool Check>
__global__ void devStateLoadToShmTestKernel(
    CtranAlgoDeviceState* devStateIn,
    CtranAlgoDeviceState* devStateOut);

// global function to wrap up device function we want to test
template <typename T>
__global__ void testCtranLocalReduce(
    size_t nsrcs,
    const T** srcs,
    size_t ndsts,
    T** dsts,
    size_t count,
    commRedOp_t redOp,
    int nranks);

template <typename T>
__global__ void testCtranLocalReduceSubsetThreadBlocks(
    size_t nsrcs,
    const T** srcs,
    size_t ndsts,
    T** dsts,
    size_t count,
    commRedOp_t redOp,
    int nranks);

template <typename T, typename RedT>
__global__ void testDequantizedAllToAllLocalReduce(
    const T* src,
    T* dst,
    size_t count,
    int myRank,
    size_t nRanks,
    commRedOp_t redOp);

template <typename T, int NSrcs, int NDsts>
__global__ void testLocalReduceVectorizedTB(
    const T** srcs,
    T** dsts,
    size_t count,
    commRedOp_t redOp,
    int nranks);
