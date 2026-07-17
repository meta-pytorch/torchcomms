// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
#include <atomic>
#include <memory>
#include <vector>

#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/utils/commSpecs.h"

struct CtranMapperRemoteAccessKey;

namespace ctran {
class ScopedIpcRegHdl;
class ScopedRegHdl;

namespace alltoallp {
enum class InitState { kUninitialized, kSubmitted, kInitialized };

struct PersistArgs {
  void* recvbuff;
  void* recvHdl;
  size_t maxRecvCount;
  commDataType_t datatype;
  bool skipCtrlMsg;
  bool ibKeysExchanged{false};
  std::vector<void*> remoteRecvBuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;
  // Hold ownership of registered handles
  std::vector<ctran::ScopedIpcRegHdl> remoteIpcRegHdls;
  std::unique_ptr<ctran::ScopedRegHdl> recvRegHdl;

  // Initialization offloads the remote handle exchange to GPE thread to avoid
  // potential deadlock on mapper epoch lock, if init is called again on the
  // main thread while there is an outstanding exec. Init returns without
  // waiting for the completion of async init. Any subsequent execution call
  // should wait for its completion via the initState flag, before the main
  // thread can schedule copy engine copies.
  std::atomic<InitState> initState{InitState::kUninitialized};
};

class AlgoImpl;
} // namespace alltoallp

} // namespace ctran
