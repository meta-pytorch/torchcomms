// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>
#include <memory>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/Broadcast/BroadcastImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"

commResult_t ctranBroadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int root,
    CtranComm* comm,
    cudaStream_t stream) {
  auto algo = NCCL_BROADCAST_ALGO;

  CTRAN_COLL_INFO(
      broadcastAlgoName(algo).c_str(),
      sendbuff,
      recvbuff,
      count,
      datatype,
      root,
      comm,
      stream);

  if (algo == NCCL_BROADCAST_ALGO::ctran) {
    algo = NCCL_BROADCAST_ALGO::ctbtree;
  }

  switch (algo) {
    case NCCL_BROADCAST_ALGO::ctbtree:
      return ctranBroadcastBinomialTree(
          sendbuff, recvbuff, count, datatype, root, comm, stream);
    case NCCL_BROADCAST_ALGO::ctdirect:
    default:
      return ctranBroadcastDirect(
          sendbuff, recvbuff, count, datatype, root, comm, stream);
  }
}

bool CtranAlgo::supportBroadcast(
    std::optional<CtranMapperBackend> specifiedBackend) const {
  auto statex = comm_->statex_.get();
  // Check if all peers have the specified backend
  if (specifiedBackend.has_value()) {
    auto backend_ = specifiedBackend.value();
    for (int peer = 0; peer < statex->nRanks(); peer++) {
      if (peer != statex->rank() &&
          !ctran_->mapper->hasBackend(peer, backend_)) {
        return false;
      }
    }
    return true;
  }

  // Default assume all backends will be use
  return ctran_->mapper->hasBackend();
}

bool ctranBroadcastSupport(
    CtranComm* comm,
    std::optional<CtranMapperBackend> specifiedBackend) {
  return ctranInitialized(comm) &&
      comm->ctran_->algo->supportBroadcast(specifiedBackend);
}
