// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "collectives.h"
#include "enqueue.h"
#include "info.h"
#include "nccl.h"

#include "meta/NcclxLogger.h"
#include "meta/wrapper/DataTypeStrUtils.h"
#include "meta/wrapper/MetaFactory.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/utils/ExtUtils.h"

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
    NCCLX_LOG(
        ERR,
        "ncclReduceScatterQuantize: Unsupported input type: {}, input type must be FP32",
        ncclDatatypeToString(inputType));
    return ncclInvalidArgument;
  }

  if (transportType != ncclBfloat16) {
    NCCLX_LOG(
        ERR,
        "ncclReduceScatterQuantize: Unsupported transport type: {}, transport type must be BF16",
        ncclDatatypeToString(transportType));
    return ncclInvalidArgument;
  }

  if (op != ncclSum && op != ncclAvg) {
    NCCLX_LOG(
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
      NCCLX_LOG(
          ERR, "ncclReduceScatterQuantize: seedPtr must point to GPU memory");
      return ncclInvalidArgument;
    }
  } else {
    NCCLX_LOG(ERR, "ncclReduceScatterQuantize: seedPtr is null");
    return ncclInvalidArgument;
  }

  return ncclSuccess;
}

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

  constexpr auto kDirectIbAlgo = NCCL_REDUCESCATTER_ALGO::ctdirect_ib;
  if (NCCL_REDUCESCATTER_QUANTIZED_ALGO ==
          NCCL_REDUCESCATTER_QUANTIZED_ALGO::ctdirect_ib &&
      comm->useCtran_ && op == ncclSum &&
      ctranReduceScatterSupport(comm->ctranComm_.get(), kDirectIbAlgo)) {
    return metaCommToNccl(ctranReduceScatterQuantize(
        sendbuff,
        recvbuff,
        recvcount,
        ncclToMetaComm(inputType),
        ncclToMetaComm(transportType),
        ncclToMetaComm(op),
        seedPtr,
        comm->ctranComm_.get(),
        stream,
        kDirectIbAlgo));
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
  };

  size_t nBytes = recvcount * ncclTypeSize(inputType) * comm->nRanks;
  info.ext = ncclx::setupQuantizeInfoExt(comm, nBytes, seedPtr, transportType);

  return ncclEnqueueCheck(&info);
}

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
}

#endif // NCCL_REDUCE_SCATTER_QUANTIZE_SUPPORTED
