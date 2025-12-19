// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_fp16.h>
#include <cstddef>

#include "Types.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicCommon.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicPImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"

CTRAN_DATATYPE_TO_FUNC_MAPPER(
    alltoallvDynamicSplitNonContigKerns,
    ncclKernelAllToAllvDynamicSplitNonContig);

commResult_t ctranAlltoallvDynamicSplitNonContig(
    const void* sendbuff,
    const size_t* inputChunkSizes,
    size_t inputChunkSizesCount,
    const size_t* inputChunkIndices,
    const size_t* inputChunkCountPerRank,
    void* const* recvbuffs,
    void* recvbuff,
    size_t maxSendcount,
    size_t maxRecvcount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    bool combine,
    size_t* outputChunkSizesPerRank) {
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "Entered ctranAlltoallvDynamicSplitNonContig {}: myRank {} nRanks {}",
      combine ? "[combine]" : "[dispatch]",
      comm->statex_->rank(),
      comm->statex_->nRanks());

  auto opCount = comm->ctran_->getOpCount();
  FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());

  if (combine && recvbuff == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid AllToAllvDynamic argument: in combine mode, recvbuff cannot be nullptr");
  }

  if (!combine && recvbuffs == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Invalid AllToAllvDynamic argument: in dispatch mode, recvbuffs cannot be nullptr");
  }

  // prepare kernel config for self and NVL copies
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG,
      stream,
      "ncclx::alltoallvDynamicSplitNonContig",
      opCount);
  KernelElem* elem = nullptr;

  FB_COMMCHECK(setupKernelConfig(
      inputChunkSizes,
      inputChunkSizesCount,
      recvbuffs,
      outputChunkSizesPerRank,
      datatype,
      comm,
      config,
      &elem));

  // Special parameter handling for ctranAllToAllvDynamicSplitNonContig
  config.args.collective.alltoallv_dynamic.split.sendbuff = sendbuff;

  config.args.collective.alltoallv_dynamic.nonContig.inputChunkIndices =
      inputChunkIndices;
  config.args.collective.alltoallv_dynamic.nonContig
      .inputChunkIndicesTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDINDICES_TMPBUF_CPU));
  config.args.collective.alltoallv_dynamic.nonContig.inputChunkCountPerRank =
      inputChunkCountPerRank;
  config.args.collective.alltoallv_dynamic.nonContig
      .inputChunkCountPerRankTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDINDICES_BLOCKLEN_TMPBUF_CPU));
  config.args.collective.alltoallv_dynamic.nonContig.maxInputChunkCountPerRank =
      all2allvDynamicMaxNumSplitsPerRank;
  config.args.collective.alltoallv_dynamic.nonContig.maxRecvcount =
      maxRecvcount;
  config.args.collective.alltoallv_dynamic.nonContig.maxSendcount =
      maxSendcount;
  config.args.collective.alltoallv_dynamic.nonContig.combine = combine;

  if (recvbuff != nullptr) {
    for (int i = 0; i < comm->statex_->nRanks(); i++) {
      config.args.collective.alltoallv_dynamic.recvbuffsPtrGPU[i] = recvbuff;
    }
  }

  // prepare operation for IB path
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  FB_COMMCHECK(setupGpeOp(
      reinterpret_cast<void**>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDBUFFS_PTR_TMPBUF_CPU)),
      recvbuffs,
      inputChunkSizesCount,
      maxSendcount,
      maxRecvcount,
      datatype,
      OpElem::opType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG,
      comm,
      opCount,
      opGroup,
      elem,
      recvbuff,
      combine));

  XCHECK(alltoallvDynamicSplitNonContigKerns.contains(datatype))
      << "alltoallvDynamicSplitNonContigKerns does not contain datatype "
      << datatype;

  ctran::PreLaunchGraphPrepareFn graphPrepareFn = nullptr;
  if (NCCL_CTRAN_ALLTOALL_CUDAGRAPH_AWARE_ENABLE) {
    graphPrepareFn =
        ctran::alltoallvdynamicp::prepareCudagraphAwareAllToAllvDynamic;
  }
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      opIbImpl,
      config,
      alltoallvDynamicSplitNonContigKerns.at(datatype),
      std::nullopt, /* timeout */
      graphPrepareFn));

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "Enqueued AlltoallvDynamicSplitNonContig {}: myRank {} nRanks {}",
      combine ? "[combine]" : "[dispatch]",
      comm->statex_->rank(),
      comm->statex_->nRanks());

  return commSuccess;
}
