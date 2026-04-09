// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "collectives.h"
#include "enqueue.h"
#include "info.h"
#include "nccl.h"

#include "meta/wrapper/DataTypeStrUtils.h"

#include "comms/ctran/utils/ExtUtils.h"
#include "folly/logging/xlog.h"

// For any nccl version that supports ncclReduceScatterQuantize, it should
// define NCCL_REDUCE_SCATTER_QUANTIZE_SUPPORTED in the nccl.h header file.
#ifdef NCCL_REDUCE_SCATTER_QUANTIZE_SUPPORTED

// Shared input validation for ncclReduceScatterQuantize.
static ncclResult_t validateReduceScatterQuantizeArgs(
    ncclDataType_t inputType,
    ncclDataType_t transportType,
    ncclRedOp_t op,
    uint64_t* seedPtr) {
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

  return ncclSuccess;
}

#if NCCL_VERSION_CODE >= 22900
// v2.29+: Use InfoExt approach for algorithm/protocol selection.
// This bypasses the cost table and directly specifies PAT + SIMPLE via
// ncclInfoExt, following the same pattern as PAT AVG (see PatAvgHelper.h).
#include "meta/collectives/QuantizeHelper.h"

static ncclResult_t ncclReduceScatterQuantizeInfoExt(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t inputType,
    ncclDataType_t transportType,
    ncclRedOp_t op,
    uint64_t* seedPtr,
    ncclComm_t comm,
    cudaStream_t stream) {
  NCCLCHECK(
      validateReduceScatterQuantizeArgs(inputType, transportType, op, seedPtr));

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
  };

  size_t nBytes = recvcount * ncclTypeSize(inputType) * comm->nRanks;
  info.ext = ncclx::setupQuantizeInfoExt(comm, nBytes, seedPtr, transportType);

  return ncclEnqueueCheck(&info);
}

#else
// v2.27: Legacy approach using direct ncclInfo fields and cost table filtering.
// TODO: Migrate to InfoExt approach. For versions >= v2.29, the InfoExt path
// above is used instead, which bypasses cost table modifications and directly
// specifies the algorithm/protocol via ncclInfoExt.
static ncclResult_t ncclReduceScatterQuantizeLegacy(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t inputType,
    ncclDataType_t transportType,
    ncclRedOp_t op,
    uint64_t* seedPtr,
    ncclComm_t comm,
    cudaStream_t stream) {
  NCCLCHECK(
      validateReduceScatterQuantizeArgs(inputType, transportType, op, seedPtr));

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

#endif // NCCL_VERSION_CODE >= 22900

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

#if NCCL_VERSION_CODE >= 22900
  return ncclReduceScatterQuantizeInfoExt(
      sendbuff,
      recvbuff,
      recvcount,
      inputType,
      transportType,
      op,
      seedPtr,
      comm,
      stream);
#else
  return ncclReduceScatterQuantizeLegacy(
      sendbuff,
      recvbuff,
      recvcount,
      inputType,
      transportType,
      op,
      seedPtr,
      comm,
      stream);
#endif
}

#endif // NCCL_REDUCE_SCATTER_QUANTIZE_SUPPORTED
