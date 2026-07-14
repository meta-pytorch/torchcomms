// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/Checks.h"

namespace ctran::allgatherp {

class AlgoImpl {
 public:
  PersistArgs pArgs;

  AlgoImpl(CtranComm* comm, cudaStream_t stream)
      : comm_(comm), stream_(stream) {};
  ~AlgoImpl();

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

  // Execute the streamed recursive-doubling algorithm of allgatherP.
  // - Uses the ctsrd plan on logical node IDs. Each local rank owns one rail
  //   column, so a logical node chunk maps to recvbuff[node * nLocalRanks +
  //   localRank].
  // - Inter-node forwarding streams per chunk on the GPE thread.
  // - Intra-node NVL broadcast remains step-based: once all chunks for a step
  //   arrive, every local rank crosses one local barrier and broadcasts that
  //   step's received column chunks via CopyEngine.
  commResult_t execStreamedRecursiveDoubling(
      const void* sendbuff,
      const size_t count,
      const commDataType_t datatype);

  static inline const std::string algoName(enum NCCL_ALLGATHER_P_ALGO algo) {
    switch (algo) {
      case NCCL_ALLGATHER_P_ALGO::ctdirect:
        return "CtranAllGatherPDirect";
      case NCCL_ALLGATHER_P_ALGO::ctpipeline:
        return "CtranAllGatherPPipeline";
      case NCCL_ALLGATHER_P_ALGO::ctsrdpipeline:
        return "CtranAllGatherPStreamedRd";
      default:
        return "Unknown";
    }
  }

  // Allocate pipeSync and other internal resources.
  commResult_t initResources();

 private:
  // Wait till either the async initialization is done or hit async error.
  // It is called before execution scheduling any CE copy to the stream.
  inline commResult_t waitInit() {
    while (pArgs.initState.load() != InitState::kInitialized) {
      FB_COMMCHECK(comm_->getAsyncResult());
    }
    return commSuccess;
  }

  Resource resource_;
  CtranComm* comm_{nullptr};
  cudaStream_t stream_{nullptr};
};

commResult_t createPersistentRequest(
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest** out);

commResult_t destroyPersistentRequest(CtranPersistentRequest* const request);
} // namespace ctran::allgatherp
