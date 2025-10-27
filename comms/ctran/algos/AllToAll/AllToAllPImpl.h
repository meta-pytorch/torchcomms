// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/synchronization/CallOnce.h>
#include "Types.h"
#include "comms/ctran/CtranComm.h"
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
  ~AlgoImpl() {};

  commResult_t init();

  commResult_t exec(const void* sendbuff, const size_t count);

  inline commResult_t setPArgs(
      void* recvbuff,
      const size_t maxRecvCount,
      bool skipCtrlMsg,
      commDataType_t datatype) {
    size_t size = maxRecvCount * commTypeSize(datatype);
    void* regHdl{nullptr};
    bool localReg = false;
    // TODO: Pass-in a flag searchOnly to avoid dynamic register instead of reg
    // then deregister.
    FB_COMMCHECK(comm_->ctran_->mapper->searchRegHandle(
        recvbuff, size, &regHdl, &localReg));
    if (localReg) {
      comm_->ctran_->mapper->deregDynamic(regHdl);
      CLOGF(
          ERR,
          "recvbuff is not registered. Pointer: {} length: {}",
          recvbuff,
          size);
      return commInternalError;
    }

    pArgs = {
        .recvbuff = recvbuff,
        .recvHdl = regHdl,
        .maxRecvCount = maxRecvCount,
        .datatype = datatype,
        .skipCtrlMsg = skipCtrlMsg,
    };
    return commSuccess;
  }

  commResult_t updatePersistentFuncAndOp(opFunc& opFunc, struct OpElem* op);

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

commResult_t prepareCudagraphAwareAllToAll(
    opFunc& opFunc,
    struct OpElem* op,
    PersistentObj& pObj);
} // namespace ctran::alltoallp
