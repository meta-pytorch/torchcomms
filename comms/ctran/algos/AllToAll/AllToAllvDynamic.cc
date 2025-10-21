// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_fp16.h>
#include <cstddef>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicCommon.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/hints/Hints.h"

CTRAN_DATATYPE_TO_FUNC_MAPPER(
    alltoallvDynamicKerns,
    ncclKernelAllToAllvDynamic);

commResult_t ctranAllToAllvDynamic(
    const void* const* sendbuffs,
    const size_t* sendcounts,
    void* const* recvbuffs,
    size_t maxSendcount,
    size_t maxRecvcount,
    size_t* actualRecvcounts,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());

  // prepare kernel config for self and NVL copies
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALLV_DYNAMIC,
      stream,
      "ncclx::alltoallvDynamic",
      opCount);
  KernelElem* elem = nullptr;

  FB_COMMCHECK(setupKernelConfig(
      sendcounts,
      comm->statex_->nRanks(),
      recvbuffs,
      actualRecvcounts,
      datatype,
      comm,
      config,
      &elem));

  // Special parameter handling for ctranAllToAllvDynamic
  for (int i = 0; i < comm->statex_->nRanks(); i++) {
    config.args.collective.alltoallv_dynamic.nonSplit.sendbuffsPtrGPU[i] =
        sendbuffs[i];
  }
  // Initlize sendIndicesTmpbufCPU for contig buf/single-expert buf, so
  // it does not need special treatment in AllToAllvDynamicImpl.
  size_t* sendIndicesTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDINDICES_TMPBUF_CPU));
  size_t* sendIndicesBlockLengthsTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDINDICES_BLOCKLEN_TMPBUF_CPU));
  for (int i = 0; i < comm->statex_->nRanks(); i++) {
    sendIndicesTmpbufCPU[i] = i;
    sendIndicesBlockLengthsTmpbufCPU[i] = 1;
  }

  // prepare operation for IB path
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  FB_COMMCHECK(setupGpeOp(
      sendbuffs,
      recvbuffs,
      comm->statex_->nRanks(),
      maxSendcount,
      maxRecvcount,
      datatype,
      OpElem::opType::ALLTOALLV_DYNAMIC,
      comm,
      opCount,
      opGroup,
      elem));

  XCHECK(alltoallvDynamicKerns.contains(datatype))
      << "alltoallvDynamicKerns does not contain datatype " << datatype;
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      opIbImpl,
      config,
      alltoallvDynamicKerns.at(datatype)));

  return commSuccess;
}

commResult_t ctranAllToAllvDynamicSupport(
    CtranComm* comm,
    const meta::comms::Hints& hints,
    const size_t* maxSendcounts,
    const size_t* maxRecvcounts,
    commDataType_t datatype) {
  const auto statex = comm->statex_.get();

  if (!ctranInitialized(comm)) {
    FB_ERRORRETURN(commInvalidUsage, "CTRAN is not initialized on local rank");
  } else {
    // Check if all remote peers are supported by ctran
    // For intra-node peers, ctranAlgo supports copy based path;
    // for inter-node peers, we need a mapper backend to support.
    const int myNode = statex->node();
    for (int rank = 0; rank < statex->nRanks(); rank++) {
      if (statex->node(rank) != myNode &&
          comm->ctran_->mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
        FB_ERRORRETURN(
            commInvalidUsage,
            "CTRAN is not initialized on remote rank {}",
            rank);
      }
    }
  }

  // FIXME:
  // a proper way to do this check is first look up the registered buffer,
  // - if buffer is registered and registered buffer size <
  // CTRAN_MIN_REGISTRATION_SIZE, fail (regardless of maxRecvcounts here)
  // - if buffer is not registered then check with maxRecvcounts
  for (int rank = 0; rank < statex->nRanks(); rank++) {
    if (maxSendcounts[rank] * commTypeSize(datatype) <
        CTRAN_MIN_REGISTRATION_SIZE) {
      FB_ERRORRETURN(
          commInvalidUsage,
          "maxSendcounts[{}] {} is too small for CTRAN, expect minimal length {}",
          rank,
          maxSendcounts[rank],
          CTRAN_MIN_REGISTRATION_SIZE / commTypeSize(datatype));
    }
    if (maxRecvcounts[rank] * commTypeSize(datatype) <
        CTRAN_MIN_REGISTRATION_SIZE) {
      FB_ERRORRETURN(
          commInvalidUsage,
          "maxRecvcounts[{}] {} is too small for CTRAN, expect minimal length {}",
          rank,
          maxRecvcounts[rank],
          CTRAN_MIN_REGISTRATION_SIZE / commTypeSize(datatype));
    }
  }

  commResult_t res;
  std::string locationRes;

  res = hints.get("ncclx_alltoallv_dynamic_sendbuffs_location", locationRes);
  if (res == commSuccess && locationRes != "cpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_sendbuffs_location, supported values: cpu");
  }

  res = hints.get("ncclx_alltoallv_dynamic_recvbuffs_location", locationRes);
  if (res == commSuccess && locationRes != "cpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_recvbuffs_location, supported values: cpu");
  }

  res = hints.get("ncclx_alltoallv_dynamic_sendcounts_location", locationRes);
  if (res == commSuccess && locationRes != "gpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_sendcounts_location, supported values: gpu");
  }

  res =
      hints.get("ncclx_alltoallv_dynamic_max_sendcounts_location", locationRes);
  if (res == commSuccess && locationRes != "cpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_max_sendcounts_location, supported values: cpu");
  }

  res =
      hints.get("ncclx_alltoallv_dynamic_max_recvcounts_location", locationRes);
  if (res == commSuccess && locationRes != "cpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_max_recvcounts_location, supported values: cpu");
  }

  res = hints.get(
      "ncclx_alltoallv_dynamic_actual_recvcounts_location", locationRes);
  if (res == commSuccess && locationRes != "gpu") {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid ncclx_alltoallv_dynamic_actual_recvcounts_location, supported values: gpu");
  }

  return commSuccess;
}
