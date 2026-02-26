// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "collectives.h"
#include "enqueue.h"
#include "info.h"
#include "nccl.h"

#include "meta/wrapper/DataTypeStrUtils.h"

#include "comms/ctran/utils/ExtUtils.h"
#include "folly/logging/xlog.h"

__attribute__((visibility("default"))) ncclResult_t ncclReduceScatterQuantize(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t inputType,
    ncclDataType_t transportType,
    ncclRedOp_t op,
    uint64_t* seedPtr,
    ncclComm_t comm,
    cudaStream_t stream) {
  SetCudaDevRAII setCudaDev(comm->cudaDev);

  // Perform input param checks
  if (inputType != ncclFloat32) {
    XLOGF(
        ERR,
        "ncclReduceScatterQuantize: Unsupported input type: {}, input type must be FP32",
        ncclDatatypeToString(inputType));
    return ncclInvalidArgument;
  }

  if (transportType != ncclBfloat16) {
    XLOGF(
        ERR,
        "ncclReduceScatterQuantize: Unsupported transport type: {}, transport type must be BF16",
        ncclDatatypeToString(transportType));
    return ncclInvalidArgument;
  }

  if (op != ncclSum && op != ncclAvg) {
    XLOGF(
        ERR,
        "ncclReduceScatterQuantize: Unsupported reduction operation: {}",
        getRedOpStr(op));
    return ncclInvalidArgument;
  }

  // Validate that seedPtr points to GPU memory using CUDA APIs
  if (seedPtr != nullptr) {
    cudaPointerAttributes attr;
    auto err = cudaPointerGetAttributes(&attr, seedPtr);
#if CUDART_VERSION >= 10000
    bool isDevicePtr =
        (err == cudaSuccess) && (attr.type == cudaMemoryTypeDevice);
#else
    // For older CUDA versions, attr.memoryType is used
    bool isDevicePtr =
        (err == cudaSuccess) && (attr.memoryType == cudaMemoryTypeDevice);
#endif
    if (!isDevicePtr) {
      XLOGF(ERR, "ncclReduceScatterQuantize: seedPtr must point to GPU memory");
      return ncclInvalidArgument;
    }
  } else {
    XLOGF(ERR, "ncclReduceScatterQuantize: seedPtr is null");
    return ncclInvalidArgument;
  }

  auto info = ncclInfo{
      .coll = ncclFuncReduceScatter,
      .opName = "ReduceScatter",
      .sendbuff = sendbuff,
      .recvbuff = recvbuff,
      .count = recvcount,
      .datatype = inputType,
      .op = op,
      .root = 0,
      .comm = comm,
      .stream = stream, /* Args */
      .chunkSteps = REDUCESCATTER_CHUNKSTEPS,
      .sliceSteps = REDUCESCATTER_SLICESTEPS,
      .randomSeed = seedPtr,
      .transportType = transportType,
  };

  return ncclEnqueueCheck(&info);
}
