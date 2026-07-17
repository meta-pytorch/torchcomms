// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/synchronization/CallOnce.h>
#include <atomic>
#include <memory>
#include <thread>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/HostTypes.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::alltoallp {
class AlgoImpl {
 public:
  PersistArgs pArgs;

  AlgoImpl(CtranComm* comm, cudaStream_t stream)
      : comm_(comm), stream_(stream) {};
  ~AlgoImpl();

  commResult_t exec(const void* sendbuff, const size_t count);

  static inline const std::string algoName(enum NCCL_ALLTOALL_ALGO algo) {
    switch (algo) {
      case NCCL_ALLTOALL_ALGO::ctran:
        return "CtranAllToAllP";
      case NCCL_ALLTOALL_ALGO::ctgraph:
        return "CtranAllToAllPCtgraph";
      default:
        return "Unknown";
    }
  }

  // Release internal resources; SW-only, safe on any teardown path.
  commResult_t destroy();

  // Wait till either the async initialization is done or hit async error.
  // Called before exec schedules any CE copy to the stream.
  inline commResult_t waitInit() {
    while (pArgs.initState.load() != InitState::kInitialized) {
      FB_COMMCHECK(comm_->getAsyncResult());
      std::this_thread::yield();
    }
    return commSuccess;
  }

 private:
  CtranComm* comm_{nullptr};
  cudaStream_t stream_{nullptr};
};

commResult_t createPersistentRequest(
    CtranComm* comm,
    cudaStream_t stream,
    void* recvbuff,
    size_t maxRecvCount,
    commDataType_t datatype,
    CtranPersistentRequest** out,
    bool waitForInit,
    bool skipCtrlMsg);

commResult_t destroyPersistentRequest(CtranPersistentRequest* const request);
} // namespace ctran::alltoallp
