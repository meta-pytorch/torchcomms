// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/utils/commSpecs.h"

// Forward declarations
struct KernelElem;

namespace ctran {

namespace alltoall {

struct KernelArgs {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  commDataType_t datatype;
  // passing nLocalRanks so kernel can be no-op when nLocalRanks == 1
  int nLocalRanks;
};

} // namespace alltoall

namespace alltoallv {

struct KernelArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  // selfCount==0 && sendElemsList==nullptr && recvElemsList==nullptr signals
  // a no-op to the kernel: skip self-copy and NVL send/recv and only perform
  // GPE start/terminate sync. Set on the pure-IB path (no local NVL peers),
  // where the self-copy is issued via cudaMemcpyAsync from the host.
  size_t selfCount;
  size_t selfSendDispl;
  size_t selfRecvDispl;
  KernelElem* sendElemsList;
  KernelElem* recvElemsList;
};

} // namespace alltoallv

namespace alltoallp {
class AlgoImpl;
} // namespace alltoallp

} // namespace ctran
