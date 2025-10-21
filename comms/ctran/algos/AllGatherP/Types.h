// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/utils/commSpecs.h"

using ctran::algos::GpeKernelSync;

struct CtranMapperRemoteAccessKey;
namespace ctran::allgatherp {
struct PersistArgs {
  void* recvbuff;
  void* recvHdl;
  size_t maxRecvCount;
  commDataType_t datatype;
  std::vector<void*> remoteRecvBuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;

  // Initialization offloads the remote handle exchange to GPE thread to avoid
  // potential deadlock on mapper epoch lock, if init is called again on the
  // main thread while there is an outstanding exec. Init returns without
  // waiting for the completion of async init. Any subsequent execution call
  // should wait for its completion via the initialized_ flag, before the main
  // thread can schedule copy engine copies
  std::atomic<bool> initialized{false};
};

struct Resource {
  // Used in the pipeline algorithm. Sync object for GPE thread to notify the
  // wait kernel the completion of inter-node exchange, so that it can terminate
  // and kick off CE bcast to forward the received data to the other local ranks
  GpeKernelSync* pipeSync{nullptr};
};

struct PipeEndKernArgs {
  GpeKernelSync* pipeSync;
};

struct PipeSyncKernArgs {
  int stepId;
  GpeKernelSync* pipeSync;
};
} // namespace ctran::allgatherp
