// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/synchronization/CallOnce.h>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::alltoallvdynamicp {
struct PersistArgs {
  std::vector<void*> recvbuffs;
  std::vector<void*> recvHdls;
  size_t maxSendCount;
  size_t maxRecvCount;
  commDataType_t datatype;
  std::vector<void*> remoteRecvBuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;
};

class AlgoImpl {
 public:
  PersistArgs pArgs;

  AlgoImpl(CtranComm* comm, cudaStream_t stream)
      : comm_(comm), stream_(stream) {};
  ~AlgoImpl() {};

  commResult_t init();

  commResult_t updatePersistFuncAndOp(
      opFunc& opFunc,
      std::vector<std::unique_ptr<struct OpElem>>& opGroup,
      struct OpElem* op);

  static inline const std::string algoName(enum NCCL_ALLTOALL_ALGO algo) {
    switch (algo) {
      case NCCL_ALLTOALL_ALGO::ctran:
        return "CtranAllToAllvDynamicP";
      default:
        return "Unknown";
    }
  }

 private:
  CtranComm* comm_{nullptr};
  cudaStream_t stream_{nullptr};
};
} // namespace ctran::alltoallvdynamicp
