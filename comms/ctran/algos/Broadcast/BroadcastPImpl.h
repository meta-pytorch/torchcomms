// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/utils/commSpecs.h"

namespace ctran::broadcastp {

struct PersistArgs {
  void* recvbuff;
  void* recvHdl;
  size_t maxRecvCount;
  commDataType_t datatype;
  bool skipCtrlMsg;
  std::vector<void*> remoteRecvBuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;
};

class AlgoImpl {
 public:
  PersistArgs pArgs;

  AlgoImpl(CtranComm* comm, cudaStream_t stream)
      : comm_(comm), stream_(stream) {};
  ~AlgoImpl() = default;

  commResult_t exec(const void* sendbuff, size_t count, int root);

 private:
  CtranComm* comm_{nullptr};
  cudaStream_t stream_{nullptr};
};

} // namespace ctran::broadcastp
