// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>
#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#if defined(ENABLE_PIPES)
#include "comms/ctran/algos/AllToAll/DeviceAllToAllvPipesImpl.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/AllToAllv.h" // WarpReserveConfig
#endif
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/utils/cvars/nccl_cvars.h"

#if defined(ENABLE_PIPES)
template <PipeProtocol Proto>
extern __global__ void ncclKernelDeviceAllToAllvPipes(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::device_alltoallv_pipes::KernArgs args);
extern __global__ void ncclKernelDeviceAllToAllvPipesUnified(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::device_alltoallv_pipes::KernArgs args);
#endif

#define RETURN_ALLTOALLV_IB_IMPL(perfconfig) \
  return ctranAllToAllvIbImpl<perfconfig>(   \
      op->alltoall.sendbuff,                 \
      sendcounts,                            \
      sdispls,                               \
      op->alltoall.recvbuff,                 \
      recvcounts,                            \
      rdispls,                               \
      op->alltoall.datatype,                 \
      op->opCount,                           \
      comm,                                  \
      std::move(timestamp));

// This variable defaults to NCCL_ALLTOALL_ALGO::ctran, but if more algos are
// added, this implementation should be updated to support them instead of
// defaulting to ctran
static const auto myAlgo = NCCL_ALLTOALL_ALGO::ctran;

static commResult_t opIbImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = opGroup.front()->comm_;
  const auto statex = comm->statex_.get();

  std::vector<size_t> sendcounts(statex->nRanks(), 0);
  std::vector<size_t> sdispls(statex->nRanks(), 0);
  std::vector<size_t> recvcounts(statex->nRanks(), 0);
  std::vector<size_t> rdispls(statex->nRanks(), 0);

  CtranAlgoLogger logger(allToAllAlgoName(myAlgo), op->opCount, comm);

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allToAllAlgoName(myAlgo)));

  const int myNode = statex->node();
  for (int i = 0; i < statex->nRanks(); i++) {
    int peerNode = statex->node(i);
    // GPE thread handles only remote peers
    if (myNode != peerNode) {
      sendcounts[i] = op->alltoall.count;
      sdispls[i] = op->alltoall.count * i;
      recvcounts[i] = op->alltoall.count;
      rdispls[i] = op->alltoall.count * i;
    }
  }

  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    RETURN_ALLTOALLV_IB_IMPL(LowLatencyCollConfig)
  } else {
    RETURN_ALLTOALLV_IB_IMPL(DefaultPerfCollConfig)
  }
}

static inline commResult_t setupGpeOp(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    uint64_t opCount,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  const auto statex = comm->statex_.get();
  // Passing op only when remote peers are present
  if (statex->nNodes() > 1) {
    std::unique_ptr<struct OpElem> op = std::unique_ptr<struct OpElem>(
        new OpElem(OpElem::opType::ALLTOALL, stream, comm, opCount));
    op->alltoall.sendbuff = sendbuff;
    op->alltoall.recvbuff = recvbuff;
    op->alltoall.count = count;
    op->alltoall.datatype = datatype;
    opGroup.push_back(std::move(op));
  }

  return commSuccess;
}

commResult_t ctranAllToAll(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLTOALL_ALGO algo) {
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      allToAllAlgoName(algo).c_str(),
      sendbuff,
      recvbuff,
      count,
      datatype,
      -1,
      comm,
      stream);

  if (count == 0) {
    return commSuccess;
  }

  // TODO: alltoallKerns perform poorly on HCM due to lack of NVL connection
  // between some GPUs We need detect topology and switch to use IB transport in
  // such a case

  // prepare kernel config for self and NVL copies
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALL,
      stream,
      allToAllAlgoName(algo),
      opCount);
  FB_COMMCHECK(
      ctran::alltoall::setupKernelConfig(
          sendbuff, recvbuff, count, datatype, comm, stream, config));

  // prepare operation for IB path
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  FB_COMMCHECK(setupGpeOp(
      sendbuff, recvbuff, count, datatype, comm, stream, opCount, opGroup));
  ctran::PreLaunchGraphPrepareFn graphPrepareFn = nullptr;
  if (NCCL_CTRAN_ALLTOALL_CUDAGRAPH_AWARE_ENABLE) {
    graphPrepareFn = ctran::alltoallp::prepareCudagraphAwareAllToAll;
  }
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      opIbImpl,
      config,
      reinterpret_cast<void*>(ctran::alltoall::alltoallKerns[datatype]),
      std::nullopt, /* timeout */
      graphPrepareFn));

  return commSuccess;
}

bool ctranAllToAllSupport(
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    enum NCCL_ALLTOALL_ALGO algo) {
  // Currently there is only one ctran algo for alltoall, but we pass algo as a
  // parameter for future extension and consistency across collectives.
  // Currently just return false if algo is set to orig
  if (algo == NCCL_ALLTOALL_ALGO::orig) {
    return false;
  }
  bool ctranSupport = false;
  const auto statex = comm->statex_.get();
  if (ctranInitialized(comm)) {
    ctranSupport = true;
    // Check if all remote peers are supported by ctran
    // For intra-node peers, ctranAlgo supports copy based path;
    // for inter-node peers, we need a mapper backend to support.
    const int myNode = statex->node();
    for (int rank = 0; rank < statex->nRanks(); rank++) {
      if (statex->node(rank) != myNode &&
          comm->ctran_->mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
        ctranSupport = false;
        break;
      }
    }
  }

  if (ctranSupport &&
      commTypeSize(datatype) * count >= NCCL_CTRAN_ALLTOALL_THRESHOLD) {
    return true;
  } else {
    return false;
  }
}

#if defined(ENABLE_PIPES)
// ============================================================================
// Device AllToAllv (split sizes on device)
// NVLink domain only — all peers must be reachable via NVLink.
// IB support will be added in a follow-up via IBGDA (not CPU proxy).
// ============================================================================

commResult_t ctranDeviceAllToAllv(
    const void* sendbuff,
    void* recvbuff,
    const int64_t* sendcounts_d,
    const int64_t* recvcounts_d,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    int64_t sendcountsMultiplier,
    int64_t recvcountsMultiplier,
    const std::unordered_map<std::string, std::string>& hints) {
  if (NCCL_CTRAN_PIPES_DEVICE_ALLTOALLV_ENABLE) {
    auto opCount = comm->ctran_->getOpCount();
    auto* mpt = comm->multiPeerTransport_.get();
    auto deviceHandle = mpt->get_device_handle();

    KernelConfig config = KernelConfig(
        KernelConfig::KernelType::DEVICE_ALLTOALLV,
        stream,
        "DeviceAllToAllvPipesUnified",
        opCount);
    config.args.devState_d = comm->ctran_->algo->getDevState();

    const int nRanks = mpt->n_ranks();
    const size_t elementSize = commTypeSize(datatype);

    // Launch with max grid dimensions from autotune tables.
    // The kernel determines effective warps via kernel-side autotune
    // and excess warps early-return. This avoids cudaMemcpy D2H of
    // sendcounts/recvcounts which is illegal during CUDA graph capture.
    config.numBlocks = comms::pipes::kMaxAutotuneBlocks;
    config.numThreads = comms::pipes::kMaxAutotuneThreads;

    ctran::device_alltoallv_pipes::KernArgs kernArgs{};
    kernArgs.sendbuff = sendbuff;
    kernArgs.recvbuff = recvbuff;
    kernArgs.myRank = mpt->my_rank();
    kernArgs.nRanks = nRanks;
    kernArgs.elementSize = elementSize;
    kernArgs.sendcounts_d = sendcounts_d;
    kernArgs.recvcounts_d = recvcounts_d;
    kernArgs.sendcountsMultiplier = sendcountsMultiplier;
    kernArgs.recvcountsMultiplier = recvcountsMultiplier;
    kernArgs.transports = deviceHandle.transports.data();
    kernArgs.hasIbgdaPeers = !deviceHandle.ibgdaPeerRanks.empty();

    // Pass CVAR overrides to kernel for merge with per-msg autotune
    kernArgs.warpReserveCvars = comms::pipes::WarpReserveConfig{
        .nvlSendWarps =
            static_cast<int>(NCCL_CTRAN_PIPES_WARP_RESERVE_NVL_SEND),
        .nvlRecvWarps =
            static_cast<int>(NCCL_CTRAN_PIPES_WARP_RESERVE_NVL_RECV),
        .ibgdaSendWarps =
            static_cast<int>(NCCL_CTRAN_PIPES_WARP_RESERVE_IBGDA_SEND),
        .ibgdaRecvWarps =
            static_cast<int>(NCCL_CTRAN_PIPES_WARP_RESERVE_IBGDA_RECV),
        .selfWarps = static_cast<int>(NCCL_CTRAN_PIPES_WARP_RESERVE_SELF),
    };

    // Pass only peer info; kernel resolves boundaries via kernel-side
    // autotune + resolveWarpReserve logic
    kernArgs.warpReserve = {};
    kernArgs.warpReserve.nvlPeerRanks = deviceHandle.nvlPeerRanks.data();
    kernArgs.warpReserve.numNvlPeers =
        static_cast<uint32_t>(deviceHandle.nvlPeerRanks.size());
    kernArgs.warpReserve.ibgdaPeerRanks = deviceHandle.ibgdaPeerRanks.data();
    kernArgs.warpReserve.numIbgdaPeers =
        static_cast<uint32_t>(deviceHandle.ibgdaPeerRanks.size());
    kernArgs.warpReserve.maxChannelsPerPeer =
        static_cast<uint32_t>(mpt->maxChannelsPerPeer());

    config.algoArgs = &kernArgs;

    std::vector<std::unique_ptr<struct OpElem>> opGroup;
    FB_COMMCHECK(comm->ctran_->gpe->submit(
        std::move(opGroup),
        nullptr,
        config,
        reinterpret_cast<void*>(ncclKernelDeviceAllToAllvPipesUnified)));

    return commSuccess;
  }

  // Ctran kernel path: GPE, block scheduling, tunable hints.
  // Currently NVLink-only.
  auto opCount = comm->ctran_->getOpCount();

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::DEVICE_ALLTOALLV,
      stream,
      "DeviceAllToAllvPipes",
      opCount);

  // CollectiveConfig resolves all settings in its constructor:
  // per-collective hint > cvar > default
  ctran::device_alltoallv_pipes::CollectiveConfig collConfig(
      comm->statex_->nLocalRanks(), &hints);

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "DeviceAllToAllvPipes: opCount {} numBlocks {} numThreads {} "
      "blockScheduling {} ll128ThresholdBytes {} hasHints {} [nLocalRanks={}]",
      opCount,
      collConfig.numBlocks,
      collConfig.numThreads,
      collConfig.blockScheduling,
      collConfig.ll128ThresholdBytes,
      (!hints.empty()),
      comm->statex_->nLocalRanks());

  ctran::device_alltoallv_pipes::KernArgs kernArgs;
  FB_COMMCHECK(
      ctran::device_alltoallv_pipes::setupKernelConfig(
          sendbuff,
          recvbuff,
          sendcounts_d,
          recvcounts_d,
          datatype,
          comm,
          config,
          kernArgs,
          sendcountsMultiplier,
          recvcountsMultiplier,
          collConfig));

  // NVLink-only: no GPE op needed (no IB fallback)
  std::vector<std::unique_ptr<struct OpElem>> opGroup;

  auto* kernel = (collConfig.ll128ThresholdBytes > 0)
      ? ncclKernelDeviceAllToAllvPipes<PipeProtocol::LL128>
      : ncclKernelDeviceAllToAllvPipes<PipeProtocol::Simple>;

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup), nullptr, config, reinterpret_cast<void*>(kernel)));

  return commSuccess;
}

bool ctranDeviceAllToAllvSupport(CtranComm* comm) {
  if (!ctranInitialized(comm)) {
    return false;
  }

  // Require MultiPeerTransport (pipes)
  if (!comm->multiPeerTransport_) {
    return false;
  }

  // Verify ALL peers are reachable via a supported transport.
  // Supported: NVLink (P2P_NVL), IBGDA (P2P_IBGDA), or self (SELF).
  // Use host-side API — getMultiPeerTransportsPtr() returns a device pointer
  // that cannot be dereferenced on the host.
  const auto statex = comm->statex_.get();
  for (int rank = 0; rank < statex->nRanks(); rank++) {
    auto type = comm->multiPeerTransport_->get_transport_type(rank);
    if (type != comms::pipes::TransportType::P2P_NVL &&
        type != comms::pipes::TransportType::P2P_IBGDA &&
        type != comms::pipes::TransportType::SELF) {
      return false;
    }
  }

  return true;
}
#endif // ENABLE_PIPES

// Stubs when ENABLE_PIPES is not defined — prevents linker errors from
// unconditional declarations in Ctran.h.
#if !defined(ENABLE_PIPES)
commResult_t ctranDeviceAllToAllv(
    const void* /*sendbuff*/,
    void* /*recvbuff*/,
    const int64_t* /*sendcounts_d*/,
    const int64_t* /*recvcounts_d*/,
    commDataType_t /*datatype*/,
    CtranComm* /*comm*/,
    cudaStream_t /*stream*/,
    int64_t /*sendcountsMultiplier*/,
    int64_t /*recvcountsMultiplier*/,
    const std::unordered_map<std::string, std::string>& /*hints*/) {
  return commInternalError;
}

bool ctranDeviceAllToAllvSupport(CtranComm* /*comm*/) {
  return false;
}
#endif // !ENABLE_PIPES
