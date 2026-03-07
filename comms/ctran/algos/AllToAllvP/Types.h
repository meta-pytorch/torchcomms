// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/utils/commSpecs.h"

using ctran::algos::GpeKernelSync;

struct CtranMapperRemoteAccessKey;
namespace ctran::alltoallvp {

struct PersistArgs {
  void* recvbuff;
  void* recvHdl;
  size_t maxRecvCount;
  commDataType_t datatype;
  std::vector<void*> remoteRecvBuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;

  // Per-peer receive buffers and handles for allToAllCtrl exchange
  std::vector<void*> recvBuffs;
  std::vector<void*> recvHdls;

  // Initialization offloads the remote handle exchange to GPE thread to avoid
  // potential deadlock on mapper epoch lock, if init is called again on the
  // main thread while there is an outstanding exec. Init returns without
  // waiting for the completion of async init. Any subsequent execution call
  // should wait for its completion via the initialized_ flag, before the main
  // thread can schedule copy engine copies
  std::atomic<bool> initialized{false};

  // Flag to signal that gpeFn() has completed the address exchange.
  // The main thread waits on this before calling nvlCeBcast() to ensure
  // remoteRecvBuffs contains valid exchanged addresses.
  std::atomic<bool> exchanged{false};
};

class AlgoImpl;
} // namespace ctran::alltoallvp
