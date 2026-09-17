// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/commSpecs.h" // need for ncclDataType_t

#ifdef CTRAN_DISABLE_TCPDM
#include "comms/ctran/backends/mock/CtranTcpDmBaseMock.h"
#else
#include "comms/tcp_devmem/unpack/batch_unpack_kernel.h"
#endif

// Forward declaration
struct KernelElem;

namespace ctran::sendrecv {
struct KernelSendArgs {
  // List of send p2p elements each will be transferred via NVL copy
  KernelElem* putNotifyList;
  // used for checksum
  const void* sendbuff;
  commDataType_t datatype;
  size_t count;
};

struct KernelRecvArgs {
  KernelElem* waitNotifyList;
  // used for checksum
  const void* recvbuff;
  commDataType_t datatype;
  size_t count;
  SQueues unpack; // TCP Device Memory
};

struct KernelSendRecvArgs {
  KernelElem* putNotifyList;
  KernelElem* waitNotifyList;
  SQueues unpack; // TCP Device Memory
};

} // namespace ctran::sendrecv
