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
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/Transport.cuh"
#endif
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/utils/cvars/nccl_cvars.h"

#if defined(ENABLE_PIPES)
extern __global__ void ncclKernelDeviceAllToAllvPipes(
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
    int64_t recvcountsMultiplier) {
  auto opCount = comm->ctran_->getOpCount();

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::DEVICE_ALLTOALLV,
      stream,
      "DeviceAllToAllvPipes",
      opCount);

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
          recvcountsMultiplier));

  // NVLink-only: no GPE op needed (no IB fallback)
  std::vector<std::unique_ptr<struct OpElem>> opGroup;

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      nullptr,
      config,
      reinterpret_cast<void*>(ncclKernelDeviceAllToAllvPipes)));

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

  // NVLink domain only: verify ALL peers are reachable via NVLink (or self).
  // Reject communicators with any IB-only peers to prevent silent data loss.
  // Use host-side API — getMultiPeerTransportsPtr() returns a device pointer
  // that cannot be dereferenced on the host.
  const auto statex = comm->statex_.get();
  for (int rank = 0; rank < statex->nRanks(); rank++) {
    auto type = comm->multiPeerTransport_->get_transport_type(rank);
    if (type != comms::pipes::TransportType::P2P_NVL &&
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
    int64_t /*recvcountsMultiplier*/) {
  return commInternalError;
}

bool ctranDeviceAllToAllvSupport(CtranComm* /*comm*/) {
  return false;
}
#endif // !ENABLE_PIPES
