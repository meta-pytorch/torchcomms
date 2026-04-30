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

  // Execute the recursive-doubling algorithm of allgatherP.
  // - Each rank copies its own chunk to every intra-node peer via NVL
  //   CopyEngine as an initial broadcast.
  // - Across the log2(nNodes) inter-node steps, each rank exchanges 2^step
  //   chunks with its rail peer at distance (nNodes / 2^(step+1)) * nLocalRanks
  //   using RDMA. Local ranks on a node stripe their IB traffic by column,
  //   so each rail link carries an equal share of the node-level exchange.
  // - After each inter-node step, the received chunks are broadcast across
  //   intra-node peers via NVL CopyEngine so that every local rank holds the
  //   union of data needed for the next step.
  // - Requires nNodes to be a power of 2. nLocalRanks == 1 (e.g. nolocal)
  //   skips the NVL broadcast stages.
  commResult_t execRecursiveDoubling(
      const void* sendbuff,
      const size_t count,
      const commDataType_t datatype);

  // Execute the PAT (Parallel Asynchronous Tiled) algorithm of allgatherP.
  // Rail-parallel butterfly (recursive doubling) with NVL CE broadcast.
  // Structurally identical to execRecursiveDoubling for v1, introduced as a
  // separate method to allow future divergence (non-power-of-two support,
  // step aggregation, auto-tuning).
  // - Requires nNodes to be a power of 2 and nRanks % nLocalRanks == 0.
  commResult_t execPat(
      const void* sendbuff,
      const size_t count,
      const commDataType_t datatype);

  static inline const std::string algoName(enum NCCL_ALLGATHER_P_ALGO algo) {
    switch (algo) {
      case NCCL_ALLGATHER_P_ALGO::ctdirect:
        return "CtranAllGatherPDirect";
      case NCCL_ALLGATHER_P_ALGO::ctpipeline:
        return "CtranAllGatherPPipeline";
      case NCCL_ALLGATHER_P_ALGO::ctrdpipeline:
        return "CtranAllGatherPRecDbl";
      case NCCL_ALLGATHER_P_ALGO::ctpat:
        return "CtranAllGatherPPat";
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
    while (!pArgs.initialized.load()) {
      FB_COMMCHECK(comm_->getAsyncResult());
    }
    return commSuccess;
  }

  Resource resource_;
  CtranComm* comm_{nullptr};
  cudaStream_t stream_{nullptr};
};
} // namespace ctran::allgatherp
