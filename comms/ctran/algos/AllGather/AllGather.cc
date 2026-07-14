// Copyright (c) Meta Platforms, Inc. and affiliates.

// #include <mutex>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/utils/MathUtils.h"
#if defined(ENABLE_PRIMS)
#include "comms/prims/transport/MultiPeerTransport.h"
#endif
#include "comms/utils/cvars/nccl_cvars.h"

static bool isGraphAwareAlgo(enum NCCL_ALLGATHER_ALGO algo) {
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctgraph:
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
    case NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline:
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
    case NCCL_ALLGATHER_ALGO::ctgraph_rd:
      return true;
    case NCCL_ALLGATHER_ALGO::ctdirect:
    case NCCL_ALLGATHER_ALGO::ctsrd:
    case NCCL_ALLGATHER_ALGO::ctring:
    case NCCL_ALLGATHER_ALGO::ctbrucks:
    case NCCL_ALLGATHER_ALGO::cthierarchical_ring:
    case NCCL_ALLGATHER_ALGO::ctran:
    case NCCL_ALLGATHER_ALGO::orig:
      return false;
  }
  return false;
}

// Check if CTRAN is supported and if a specific algo is supported by CTRAN.
// If user sets a specific algo, it should check to avoid unexpected abort in
// ctranAllGather.
bool ctranAllGatherSupport(
    CtranComm* comm,
    enum NCCL_ALLGATHER_ALGO algo,
    cudaStream_t stream) {
  const bool pipesAlgo = algo == NCCL_ALLGATHER_ALGO::cthierarchical_ring;
  if (!ctranInitialized(comm) ||
      (!pipesAlgo && !comm->ctran_->mapper->hasBackend())) {
    return false;
  }

  const auto statex = comm->statex_.get();
  bool supported = false;
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctring:
    case NCCL_ALLGATHER_ALGO::ctbrucks:
    case NCCL_ALLGATHER_ALGO::ctsrd:
      supported = statex->nLocalRanks() == 1;
      if (!supported) {
        CLOGF_SUBSYS(
            WARN,
            COLL,
            "AllGather algorithm {} only support nLocalRanks=1. Falling back to baseline",
            allGatherAlgoName(algo));
      }
      break;
    case NCCL_ALLGATHER_ALGO::cthierarchical_ring:
#if defined(ENABLE_PRIMS)
      if (!comm->multiPeerTransport_) {
        CLOGF_SUBSYS(
            WARN,
            COLL,
            "AllGather {} requires MultiPeerTransport (NCCL_CTRAN_USE_PIPES=1)",
            allGatherAlgoName(algo));
        supported = false;
        break;
      }
      {
        const bool nvlAvailable =
            !comm->multiPeerTransport_->nvl_peer_ranks().empty();
        const int ibSize = nvlAvailable ? statex->nNodes() : statex->nRanks();
        const int nvlSize = nvlAvailable ? statex->nLocalRanks() : 1;
        if (statex->nRanks() <= 1 || ibSize <= 1) {
          CLOGF_SUBSYS(
              WARN,
              COLL,
              "AllGather {} requires multiple IB peers, got nRanks={} ibSize={}. Falling back to baseline",
              allGatherAlgoName(algo),
              statex->nRanks(),
              ibSize);
          supported = false;
          break;
        }
        if (nvlSize < 1 || statex->nRanks() != ibSize * nvlSize) {
          CLOGF_SUBSYS(
              WARN,
              COLL,
              "AllGather {} requires rectangular rank geometry, got nRanks={} ibSize={} nvlSize={}. Falling back to baseline",
              allGatherAlgoName(algo),
              statex->nRanks(),
              ibSize,
              nvlSize);
          supported = false;
          break;
        }
      }
      if (!NCCL_CTRAN_IBGDA_SENDRECV_ENABLE) {
        CLOGF_SUBSYS(
            WARN,
            COLL,
            "AllGather {} requires NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1",
            allGatherAlgoName(algo));
        supported = false;
        break;
      }
      supported = true;
#else
      supported = false;
#endif
      break;
    case NCCL_ALLGATHER_ALGO::ctdirect:
    case NCCL_ALLGATHER_ALGO::ctran:
      supported = true;
      break;
    case NCCL_ALLGATHER_ALGO::ctgraph:
    case NCCL_ALLGATHER_ALGO::ctgraph_pipeline:
    case NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline:
    case NCCL_ALLGATHER_ALGO::ctgraph_ring:
    case NCCL_ALLGATHER_ALGO::ctgraph_rd: {
      // Check stream capture status
      if (stream == nullptr) {
        supported = false;
        break;
      }
      ctran::utils::cudagraph::StreamCaptureInfo captureInfo;
      auto err =
          ctran::utils::cudagraph::getStreamCaptureInfo(stream, captureInfo);
      supported = (err == cudaSuccess) &&
          (captureInfo.status == cudaStreamCaptureStatusActive);
      if (!supported) {
        CLOGF_SUBSYS(
            INFO,
            COLL,
            "AllGather {}: not in capture mode. "
            "Falling back to baseline. "
            "commHash {:x} commDesc {} nRanks {} nLocalRanks {} nNodes {}",
            allGatherAlgoName(algo),
            statex->commHash(),
            statex->commDesc(),
            statex->nRanks(),
            statex->nLocalRanks(),
            statex->nNodes());
        break;
      }

      // Topology check for explicit algo variants
      if ((algo == NCCL_ALLGATHER_ALGO::ctgraph_ring ||
           algo == NCCL_ALLGATHER_ALGO::ctgraph_rd) &&
          statex->nLocalRanks() > 1) {
        CLOGF_SUBSYS(
            WARN,
            COLL,
            "AllGather {} requires nLocalRanks==1, got {}. "
            "Falling back to baseline",
            allGatherAlgoName(algo),
            statex->nLocalRanks());
        supported = false;
        break;
      }
      if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline &&
          !ctran::utils::isPowerOfTwo(statex->nNodes())) {
        CLOGF_SUBSYS(
            WARN,
            COLL,
            "AllGather {} requires power-of-2 nNodes, got {}. "
            "Falling back to baseline",
            allGatherAlgoName(algo),
            statex->nNodes());
        supported = false;
        break;
      }
      if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rd &&
          !ctran::utils::isPowerOfTwo(statex->nRanks())) {
        CLOGF_SUBSYS(
            WARN,
            COLL,
            "AllGather {} requires power-of-2 nRanks, got {}. "
            "Falling back to baseline",
            allGatherAlgoName(algo),
            statex->nRanks());
        supported = false;
        break;
      }
      break;
    }
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
    cudaStream_t stream,
    enum NCCL_ALLGATHER_ALGO algo) {
  // Cudagraph-aware optimization: when capturing and AGP is supported,
  // transparently convert to the persistent window-based AGP algorithm.
  if (isGraphAwareAlgo(algo)) {
    ctran::utils::cudagraph::StreamCaptureInfo captureInfo;
    FB_CUDACHECK(
        ctran::utils::cudagraph::getStreamCaptureInfo(stream, captureInfo));
    if (captureInfo.status == cudaStreamCaptureStatusActive) {
      return ctranAllGatherCudagraphAware(
          sendbuff, recvbuff, sendcount, datatype, comm, stream, algo);
    }
    FB_ERRORRETURN(
        commInvalidUsage,
        "AllGather {} called outside CUDA graph capture. "
        "ctranAllGatherSupport should have returned false.",
        allGatherAlgoName(algo));
  }

  const auto statex = comm->statex_.get();

  // Only ctdirect supports nLocalRanks>1 case.
  // Force to use ctdirect if nLocalRanks>1.
  if (algo == NCCL_ALLGATHER_ALGO::ctran) {
    if (statex->nLocalRanks() > 1) {
      algo = NCCL_ALLGATHER_ALGO::ctdirect;
      CLOGF_SUBSYS(
          INFO, COLL, "Running AllGather ctdirect algorithm for nLocalRanks>1");
    }
    // pick ctring for nLocalRanks=1 if user doesn't provide specific algo
    else if (statex->nLocalRanks() == 1) {
      algo = NCCL_ALLGATHER_ALGO::ctring;
      CLOGF_SUBSYS(
          INFO, COLL, "Running AllGather ctring algorithm for nLocalRanks=1");
    }
  }
  switch (algo) {
    case NCCL_ALLGATHER_ALGO::ctring:
      return ctranAllGatherRing(
          sendbuff, recvbuff, sendcount, datatype, comm, stream);

    case NCCL_ALLGATHER_ALGO::ctbrucks:
      return ctranAllGatherBrucksFF(
          sendbuff, recvbuff, sendcount, datatype, comm, stream);

    case NCCL_ALLGATHER_ALGO::ctsrd:
      return ctranAllGatherStreamedRd(
          sendbuff, recvbuff, sendcount, datatype, comm, stream);

    case NCCL_ALLGATHER_ALGO::cthierarchical_ring:
      return ctranAllGatherHierarchicalRing(
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
