// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h" // for CTRAN_ALGO_MAX_THREAD_BLOCKS macro
#include "comms/utils/commSpecs.h"

#ifdef CTRAN_DISABLE_TCPDM
#include "comms/ctran/backends/mock/CtranTcpDmBaseMock.h"
#else
#include "comms/tcp_devmem/unpack/batch_unpack_kernel.h"
#endif

// Forward declaration
struct KernelElem;

namespace ctran::broadcast {

struct KernelArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  size_t count;
  KernelElem* putNotifyList;
  KernelElem* waitNotifyList;
  SQueues unpack; // TCP Device Memory
};

} // namespace ctran::broadcast
