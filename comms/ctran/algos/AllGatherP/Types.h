// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <atomic>
#include <memory>
#include <optional>
#include <vector>

#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::algos::GpeKernelSync;

struct CtranMapperRemoteAccessKey;

namespace ctran {
class ScopedIpcRegHdl;
class ScopedRegHdl;
} // namespace ctran

namespace ctran::allgatherp {
enum class InitState { kUninitialized, kSubmitted, kInitialized };

struct PersistArgs {
  void* recvbuff;
  size_t maxRecvCount;
  commDataType_t datatype;
  // Read only fields; ownership held by scoped regHdls.
  void* recvHdl;
  std::vector<void*> remoteRecvBuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;
  // Hold ownership of registered handles
  std::vector<ctran::ScopedIpcRegHdl> remoteIpcRegHdls_;
  std::unique_ptr<ctran::ScopedRegHdl> recvRegHdl_;

  // Initialization offloads the remote handle exchange to GPE thread to avoid
  // potential deadlock on mapper epoch lock, if init is called again on the
  // main thread while there is an outstanding exec. Init returns without
  // waiting for the completion of async init. Any subsequent execution call
  // should wait for its completion via the initState flag, before the main
  // thread can schedule copy engine copies
  std::atomic<InitState> initState{InitState::kUninitialized};

  // Set once the inter-node IB rkeys are populated in remoteRecvBuffs /
  // remoteAccessKeys by the gpeFn's first-exec peer exchange -- both eager and
  // graph defer the rkey exchange to first exec -- so later execs only re-sync.
  // GPE-thread-only, so not atomic.
  bool ibKeysExchanged{false};

  // Per-request AGP variant override. nullopt means "use the
  // NCCL_ALLGATHER_P_ALGO cvar" (preserves behavior for all existing callers);
  // ctwin sets it per-comm by topology since a single cvar cannot express
  // multiple comms with different topologies.
  std::optional<enum NCCL_ALLGATHER_P_ALGO> algo;
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
