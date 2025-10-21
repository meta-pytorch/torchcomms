// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/cvars/nccl_cvars.h"

bool ctranAllGatherSupport(CtranComm* comm) {
  return ctranInitialized(comm) && comm->ctran_->mapper->hasBackend();
}

// Check if a specific algo is supported by CTRAN.
// If user sets a specific algo, it should check to avoid unexpected abort in
// ctranAllGather.
bool ctranAllGatherAlgoSupport(
    CtranComm* comm,
    const enum NCCL_ALLGATHER_ALGO algo) {
  if (!ctranAllGatherSupport(comm)) {
    return false;
  }

  const auto statex = comm->statex_.get();
  bool supported = false;
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctring:
      supported = statex->nLocalRanks() == 1;
      break;
    case NCCL_ALLGATHER_ALGO::ctbrucks:
      supported = statex->nLocalRanks() == 1;
      break;
    case NCCL_ALLGATHER_ALGO::ctrd:
      supported = statex->nLocalRanks() == 1;
      break;
    case NCCL_ALLGATHER_ALGO::ctdirect:
    case NCCL_ALLGATHER_ALGO::ctran:
      supported = true;
      break;
    case NCCL_ALLGATHER_ALGO::orig: // invalid query
      supported = false;
      break;
  }

  return supported;
}

commResult_t ctranAllGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  const auto statex = comm->statex_.get();
  // Set algo to global config
  auto algo = NCCL_ALLGATHER_ALGO;
  // Override algo if comm config is set
  if (ctranInitialized(comm)) {
    algo = comm->ctran_->algo->getAllGatherAlgo();
  }

  // Only ctdirect supports nLocalRanks>1 case.
  // Force to use ctdirect if nLocalRanks>1.
  if (algo == NCCL_ALLGATHER_ALGO::ctran) {
    if (statex->nLocalRanks() > 1) {
      algo = NCCL_ALLGATHER_ALGO::ctdirect;
    }
    // pick ctring for nLocalRanks=1 if user doesn't provide specific algo
    else if (statex->nLocalRanks() == 1) {
      algo = NCCL_ALLGATHER_ALGO::ctring;
    }
  }

  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctring:
      FB_CHECKABORT(
          comm->statex_->nLocalRanks() == 1,
          "CTRing only supports nLocalRanks=1");
      return ctranAllGatherRing(
          sendbuff, recvbuff, sendcount, datatype, comm, stream);

    case NCCL_ALLGATHER_ALGO::ctbrucks:
      FB_CHECKABORT(
          comm->statex_->nLocalRanks() == 1,
          "CTBrucks only supports nLocalRanks=1");
      return ctranAllGatherBrucksFF(
          sendbuff, recvbuff, sendcount, datatype, comm, stream);

    case NCCL_ALLGATHER_ALGO::ctrd:
      FB_CHECKABORT(
          comm->statex_->nLocalRanks() == 1,
          "CTRD only supports nLocalRanks=1");
      return ctranAllGatherRd(
          sendbuff, recvbuff, sendcount, datatype, comm, stream);
    case NCCL_ALLGATHER_ALGO::ctdirect:
    default:
      return ctranAllGatherDirect(
          sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }
}

// Util method for preparing out-of-place and small msg sizes before allgather
// collective set extraCopyBuff to be ctran internal pre-registered buffer if
// recvbuff is smaller than CTRAN_MIN_REGISTRATION_SIZE
commResult_t prepareAllGatherArgs(
    std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    KernelConfig& config,
    void** extraCopyBuff,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  std::unique_ptr<struct OpElem> op;
  auto opCount = comm->ctran_->getOpCount();
  const auto statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  const int typeSize = commTypeSize(datatype);
  bool outOfPlace =
      ((uintptr_t)recvbuff + rank * sendcount * typeSize !=
       (uintptr_t)sendbuff);

  // separate user send/recv buffers from ctran internal ones
  const void* sbuf = const_cast<void*>(sendbuff);
  void* dbuf = recvbuff;
  auto useCtranRegBuf = sendcount * typeSize * nRanks <
      CTRAN_MIN_REGISTRATION_SIZE; // IB verbs cannot register buffers <= page
                                   // size, see https://fburl.com/code/bp8m740o
  if (useCtranRegBuf) {
    FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());
    // only destination buffer needs memory registeration, sendbuff is only used
    // under ppn > 1 intraNode case
    dbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_DST_TMPBUF);
  }

  /* need to copy data from sendbuff if out-of-place or dbuf pointing to
   * pre-registered internode buffers  */
  if (outOfPlace || useCtranRegBuf) {
    FB_COMMCHECK(comm->ctran_->mapper->icopy(
        (void*)((uintptr_t)dbuf + rank * sendcount * typeSize),
        sbuf,
        sendcount * typeSize,
        stream));
  }

  op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::ALLGATHER, comm, opCount));
  op->allgather.sendbuff = sbuf;
  op->allgather.recvbuff = dbuf;
  op->allgather.sendcount = sendcount;
  op->allgather.datatype = datatype;
  opGroup.push_back(std::move(op));

  // kernel arguments are unused for now; needed for NVL path support
  ctranKernelSetAllGatherArgs(
      sbuf,
      dbuf,
      datatype,
      sendcount,
      comm->ctran_->algo->getDevState(),
      &config.args);
  if (useCtranRegBuf) {
    *extraCopyBuff = dbuf;
  }
  return commSuccess;
}
