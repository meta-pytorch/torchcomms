// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/utils/Checks.h"

namespace ctran::allgatherp {

class AlgoImpl {
 public:
  PersistArgs pArgs;

  AlgoImpl(CtranComm* comm, cudaStream_t stream)
      : comm_(comm), stream_(stream) {};
  ~AlgoImpl() {};

  commResult_t initialize();
  commResult_t destroy();

  // Execute the direct algorithm of allgatherP.
  // Each rank sends its own data to all other ranks. For intranode peers, the
  // send is done via IB backend. This is the most naive implementation of
  // allgatherP.
  commResult_t execDirect(
      const void* sendbuff,
      const size_t count,
      const commDataType_t datatype);

  // Execute the pipeline algorithm of allgatherP.
  // - Each rank sends its own data to other inter-node peers in the same rail
  //   via a Ring.
  // - Each rank sends its own data to other intra-node peers via NVL, and
  //   whenever receives a chunk from the inter-node peer, it broadcasts the
  //   chunk to all other intra-node peers via NVL.
  // - The inter-node put and intra-node broadcast are pipelined. The i-th chunk
  //   inter-node put may be overlapped with the (i-1)-th chunk intra-node
  //   broadcast.
  commResult_t execPipeline(
      const void* sendbuff,
      const size_t count,
      const commDataType_t datatype);

  static inline const std::string algoName(enum NCCL_ALLGATHER_P_ALGO algo) {
    switch (algo) {
      case NCCL_ALLGATHER_P_ALGO::ctdirect:
        return "CtranAllGatherPDirect";
      case NCCL_ALLGATHER_P_ALGO::ctpipeline:
        return "CtranAllGatherPPipeline";
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

  Resource resource_;
  CtranComm* comm_{nullptr};
  cudaStream_t stream_{nullptr};
  std::vector<cudaStream_t> ceStreams_;
};
} // namespace ctran::allgatherp
