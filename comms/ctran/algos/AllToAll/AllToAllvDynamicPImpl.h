// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/synchronization/CallOnce.h>
#include "Types.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::alltoallvdynamicp {
class AlgoImpl {
 public:
  PersistArgs pArgs;

  AlgoImpl(CtranComm* comm, cudaStream_t stream)
      : comm_(comm), stream_(stream) {};
  ~AlgoImpl() {};

  commResult_t init();

  commResult_t updatePersistFuncAndOp(opFunc& opFunc, struct OpElem* op);

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

commResult_t prepareCudagraphAwareAllToAllvDynamic(
    opFunc& opFunc,
    struct OpElem* op,
    PersistentObj& pObj);
} // namespace ctran::alltoallvdynamicp
