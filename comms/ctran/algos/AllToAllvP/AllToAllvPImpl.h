// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/AllToAllvP/Types.h"

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::alltoallvp {

class AlgoImpl {
 public:
  PersistArgs pArgs;

  AlgoImpl(CtranComm* comm, cudaStream_t stream)
      : comm_(comm), stream_(stream) {};
  ~AlgoImpl() {};

  commResult_t init();
  commResult_t exec(
      const void* sendbuff,
      const size_t sendcounts[],
      const size_t sdispls[],
      const size_t recvcounts[],
      const size_t rdispls[],
      const commDataType_t datatype);

  static inline const std::string algoName(enum NCCL_ALLTOALLV_P_ALGO algo) {
    switch (algo) {
      case NCCL_ALLTOALLV_P_ALGO::ctran:
        return "CtranAllToAllvP";
      default:
        return "Unknown";
    }
  }

 private:
  // Wait till either the async initialization is done or hit async error.
  // It is called before execution scheduling any CE copy to the stream.
  inline commResult_t waitInit() {
    while (!pArgs.initialized.load()) {
      FB_COMMCHECK(comm_->getAsyncResult());
    }
    return commSuccess;
  }

  inline commResult_t waitExchange() {
    while (!pArgs.exchanged.load()) {
      FB_COMMCHECK(comm_->getAsyncResult());
    }
    return commSuccess;
  }

  CtranComm* comm_{nullptr};
  cudaStream_t stream_{nullptr};
};
} // namespace ctran::alltoallvp
