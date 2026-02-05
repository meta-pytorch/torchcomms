// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/utils/commSpecs.h"

// Forward declaration
struct KernelElem;

#define ALLREDUCE_MAX_KERNEL_ELEMS (8)

namespace ctran::allreduce {

enum class KernElemRole {
  kIntraReduceScatter,
  kInterReduceScatter,
  kIntraAllGather,
  kRemIntraReduce,
  kRemIntraBcast,
  kRemInterReduce
};

struct KernelArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  commRedOp_t redOp;
  size_t count;
  size_t nSteps;
  void* tmpbuff;
  size_t tmpbuffSize;
  // IPC imported ptr to each of the local peers' tmpRecvBuff
  void* intraNodeRemoteTmpRecvBuffs[CTRAN_MAX_NVL_PEERS];
  // IPC imported ptr to each of the local peers' RecvBuff
  void* intraNodeRemoteRecvBuffs[CTRAN_MAX_NVL_PEERS];
  KernelElem* kernelElems[ALLREDUCE_MAX_KERNEL_ELEMS];
};

} // namespace ctran::allreduce
