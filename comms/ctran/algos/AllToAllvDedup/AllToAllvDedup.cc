// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/json.h>
#include <cstddef>

#include "comms/ctran/algos/AllToAllvDedup/AlgoImpl.h"
#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/ArgCheck.h"

namespace ctran {
using namespace utils;

commResult_t allToAllvDedupInit(
    const int totalNumSendBlocks, // number of blocks (tokens) per batch
    const int blockCount, // number of elements per block (token)
    const int blockNumRecvBuckets, // number of receiving buckets for each
                                   // block (experts per token, topK)
    const int numRecvBuckets, // number of receiving buckets per rank (expert
                              // per rank)
    meta::comms::Hints hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request) {
  if (numRecvBuckets > alltoallvdedup::MAX_NUM_RECV_BUCKETS) {
    CLOGF_SUBSYS(
        ERR,
        INIT,
        "Unsupported numRecvBuckets value: {}. Must be less than {}",
        numRecvBuckets,
        alltoallvdedup::MAX_NUM_RECV_BUCKETS);
    return commResult_t::commInvalidArgument;
  }
  if (comm->statex_->nNodes() > alltoallvdedup::MAX_NUM_NODES) {
    CLOGF_SUBSYS(
        ERR,
        INIT,
        "Unsupported nNodes value: {}. Must be less than {}",
        comm->statex_->nNodes(),
        alltoallvdedup::MAX_NUM_NODES);
    return commResult_t::commInvalidArgument;
  }

  request = new CtranPersistentRequest(
      CtranPersistentRequest::Type::ALLTOALLV_DEDUP, comm, stream);

  auto statex = comm->statex_.get();

  // FIXME: avoid manual memory management for algo object
  auto algo =
      new alltoallvdedup::AlgoImpl(comm, statex, comm->ctran_.get(), stream);
  request->algo = algo;
  const auto myRank = statex->rank();
  algo->perfTracer = std::make_unique<perftrace::Tracer>(myRank);
  auto ts = std::make_unique<perftrace::Record>("allToAllvDedup", myRank);
  ts->startInterval("allToAllvDedupInit", 0, myRank);

  CLOGF_SUBSYS(INFO, INIT, "Rank {} allToAllvDedupInit START", myRank);

  algo->pArgs = {
      .totalNumSendBlocks = totalNumSendBlocks,
      .blockCount = blockCount,
      .blockNumRecvBuckets = blockNumRecvBuckets,
      .numRecvBuckets = numRecvBuckets,
      .datatype = datatype};

  auto guard = folly::makeGuard([algo, request] {
    if (algo) {
      delete algo;
    }
    if (request) {
      delete request;
    }
  });

  FB_COMMCHECK(algo->initialize());
  guard.dismiss();

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "Rank {} allToAllvDedupInit request {} COMPLETE",
      myRank,
      (void*)request);

  ts->endInterval("allToAllvDedupInit", 0);
  algo->perfTracer->addRecord(std::move(ts));
  return commSuccess;
}

commResult_t allToAllvDedupPrepare(
    const int blockRecvBuckets[],
    size_t numSendBlocks[],
    size_t numRecvBlocks[],
    size_t recvOffsets[],
    size_t numForwardBlocks[],
    size_t* totalNumRecvBlocks,
    // start arguments for external combine
    int xnodeInputSplits[],
    int xnodeOutputSplits[],
    int xnodeGatherIndices[],
    int localInputSplits[],
    int localOutputSplits[],
    int localGatherIndices[],
    int eGatherIndices[],
    // end arguments for external combine
    CtranPersistentRequest* request) {
  // TODO: this API is no longer needed. Will remove API and nccl callsite in
  // next DIFF
  return commSuccess;
}

commResult_t allToAllvDedupExec(
    const void* sendBuff,
    const int sendIdx[],
    const int fwdIdx[],
    const int recvIdx[],
    void* recvBuff,
    int recvBlockIds[],
    CtranPersistentRequest* request) {
  auto comm = request->comm_;
  auto stream = request->stream;
  auto algo = reinterpret_cast<alltoallvdedup::AlgoImpl*>(request->algo);

  auto ctran = comm->ctran_.get();
  auto opCount = ctran->getOpCount();

  auto execArgs = alltoallvdedup::ExecArgs{
      .sendBuff = sendBuff,
      .sendIdx = sendIdx,
      .fwdIdx = fwdIdx,
      .recvIdx = recvIdx,
      .recvBuff = recvBuff,
      .recvBlockIds = recvBlockIds};

  ARGCHECK_NULL_COMM(sendBuff, "sendBuff");
  ARGCHECK_NULL_COMM(sendIdx, "sendIdx");
  ARGCHECK_NULL_COMM(fwdIdx, "fwdIdx");
  ARGCHECK_NULL_COMM(recvIdx, "recvIdx");
  // It is possible that a rank has no recv blocks, so skip check recvBuff,
  // recvBlockIds

  CTRAN_COLL_INFO(
      alltoallvdedup::AlgoImpl::algoName(alltoallvdedup::AlgoImpl::Phase::kExec)
          .c_str(),
      execArgs.sendBuff,
      execArgs.recvBuff,
      0UL,
      algo->pArgs.datatype,
      -1,
      comm,
      stream);

  return algo->exec(execArgs, opCount);
}

commResult_t allToAllvDedupDestroy(CtranPersistentRequest* request) {
  auto comm = request->comm_;
  auto myRank = comm->statex_->rank();

  if (request->type != CtranPersistentRequest::Type::ALLTOALLV_DEDUP) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Unexpected PersistentRequest type {} called into allToAllvDedupDestroy",
        request->type);
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "Rank {} allToAllvDedupDestroy request {} START",
      myRank,
      (void*)request);

  if (request->algo) {
    delete reinterpret_cast<alltoallvdedup::AlgoImpl*>(request->algo);
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "Rank {} allToAllvDedupDestroy request {} COMPLETE",
      myRank,
      (void*)request);
  return commSuccess;
}

bool allToAllvDedupSupport(CtranComm* comm, meta::comms::Hints hints) {
  bool ctranSupport = false;
  const auto statex = comm->statex_.get();
  if (ctranInitialized(comm)) {
    ctranSupport = true;
    // Check if all remote peers are supported by ctran
    for (auto rank = 0; rank < statex->nRanks(); rank++) {
      if (comm->ctran_->mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
        ctranSupport = false;
        break;
      }
    }
  }

  return ctranSupport;
}
} // namespace ctran
