// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "collectives.h"
#include "enqueue.h"
#include "info.h"
#include "nccl.h"

#include "meta/wrapper/DataTypeStrUtils.h"

#include "comms/ctran/utils/ExtUtils.h"
#include "folly/logging/xlog.h"

ncclResult_t ncclReduceScatterQuantize(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t inputType,
    ncclDataType_t transportType,
    ncclRedOp_t op,
    uint64_t seed,
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
      .randomSeed = seed,
  };

  return ncclEnqueueCheck(&info);
}
