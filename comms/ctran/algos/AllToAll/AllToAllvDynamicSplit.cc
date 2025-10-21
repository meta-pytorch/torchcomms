// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cuda_fp16.h>
#include <cstddef>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicCommon.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"

CTRAN_DATATYPE_TO_FUNC_MAPPER(
    alltoallvDynamicSplitKerns,
    ncclKernelAllToAllvDynamicSplit);

commResult_t ctranAlltoallvDynamicSplit(
    const void* sendbuff,
    const size_t* sendSplitLengths,
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
      KernelConfig::KernelType::ALLTOALLV_DYNAMIC_SPLIT,
      stream,
      "ncclx::alltoallvDynamicSplit",
      opCount);
  KernelElem* elem = nullptr;

  FB_COMMCHECK(setupKernelConfig(
      sendSplitLengths,
      comm->statex_->nRanks(),
      recvbuffs,
      actualRecvcounts,
      datatype,
      comm,
      config,
      &elem));

  // Special parameter handling for ctranAllToAllvDynamicSplit
  config.args.collective.alltoallv_dynamic.split.sendbuff = sendbuff;
  config.args.collective.alltoallv_dynamic.actualRecvcounts = actualRecvcounts;
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
      reinterpret_cast<void**>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDBUFFS_PTR_TMPBUF_CPU)),
      recvbuffs,
      comm->statex_->nRanks(),
      maxSendcount,
      maxRecvcount,
      datatype,
      OpElem::opType::ALLTOALLV_DYNAMIC_SPLIT,
      comm,
      opCount,
      opGroup,
      elem));

  XCHECK(alltoallvDynamicSplitKerns.contains(datatype))
      << "alltoallvDynamicSplitKerns does not contain datatype " << datatype;
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      opIbImpl,
      config,
      alltoallvDynamicSplitKerns.at(datatype)));

  return commSuccess;
}
