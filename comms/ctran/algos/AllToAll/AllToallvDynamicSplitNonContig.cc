// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_fp16.h>
#include <cstddef>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicCommon.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"

CTRAN_DATATYPE_TO_FUNC_MAPPER(
    alltoallvDynamicSplitNonContigKerns,
    ncclKernelAllToAllvDynamicSplitNonContig);

commResult_t ctranAlltoallvDynamicSplitNonContig(
    const void* sendbuff,
    const size_t* sendSplitLengths,
    size_t numSendSplitLengths,
    const size_t* sendIndices,
    const size_t* sendIndicesBlockLengths,
    void* const* recvbuffs,
    size_t* recvAllSplitLengths,
    size_t* recvIndices,
    size_t* recvIndicesBlockLengths,
    size_t maxSendcount,
    size_t maxRecvcount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());

  // prepare kernel config for self and NVL copies
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG,
      stream,
      "ncclx::alltoallvDynamicSplitNonContig",
      opCount);
  KernelElem* elem = nullptr;

  FB_COMMCHECK(setupKernelConfig(
      sendSplitLengths,
      numSendSplitLengths,
      recvbuffs,
      recvAllSplitLengths,
      datatype,
      comm,
      config,
      &elem));

  // Special parameter handling for ctranAllToAllvDynamicSplitNonContig
  config.args.collective.alltoallv_dynamic.split.sendbuff = sendbuff;

  config.args.collective.alltoallv_dynamic.nonContig.sendIndices = sendIndices;
  config.args.collective.alltoallv_dynamic.nonContig.sendIndicesTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDINDICES_TMPBUF_CPU));
  config.args.collective.alltoallv_dynamic.nonContig.sendIndicesBlockLengths =
      sendIndicesBlockLengths;
  config.args.collective.alltoallv_dynamic.nonContig
      .sendIndicesBlockLengthsTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDINDICES_BLOCKLEN_TMPBUF_CPU));
  config.args.collective.alltoallv_dynamic.nonContig.maxSendIndicesBlockLength =
      CTRAN_MAX_NUM_SPLITS_PER_RANK;
  config.args.collective.alltoallv_dynamic.nonContig.maxRecvcount =
      maxRecvcount;
  config.args.collective.alltoallv_dynamic.nonContig.maxSendcount =
      maxSendcount;

  // prepare operation for IB path
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  FB_COMMCHECK(setupGpeOp(
      reinterpret_cast<void**>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDBUFFS_PTR_TMPBUF_CPU)),
      recvbuffs,
      numSendSplitLengths,
      maxSendcount,
      maxRecvcount,
      datatype,
      OpElem::opType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG,
      comm,
      opCount,
      opGroup,
      elem));

  XCHECK(alltoallvDynamicSplitNonContigKerns.contains(datatype))
      << "alltoallvDynamicSplitNonContigKerns does not contain datatype "
      << datatype;
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      opIbImpl,
      config,
      alltoallvDynamicSplitNonContigKerns.at(datatype)));

  return commSuccess;
}
