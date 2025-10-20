// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/synchronization/CallOnce.h>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::alltoallp {
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
      : comm_(comm), stream_(stream){};
  ~AlgoImpl(){};

  commResult_t init();

  commResult_t exec(const void* sendbuff, const size_t count);

  static inline const std::string algoName(enum NCCL_ALLTOALL_ALGO algo) {
    switch (algo) {
      case NCCL_ALLTOALL_ALGO::ctran:
        return "CtranAllToAllP";
      default:
        return "Unknown";
    }
  }

 private:
  CtranComm* comm_{nullptr};
  cudaStream_t stream_{nullptr};
};
} // namespace ctran::alltoallp
