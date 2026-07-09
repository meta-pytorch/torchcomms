// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if defined(ENABLE_PRIMS)

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CollUtils.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterDirectIbConfig.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/prims/collectives/ReduceScatterDirectIbLauncher.h"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"

static const auto myAlgo = NCCL_REDUCESCATTER_ALGO::ctdirect_ib;

namespace {

bool checkedMultiply(size_t lhs, size_t rhs, size_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool checkedAdd(uintptr_t lhs, size_t rhs, uintptr_t& result) {
  if (rhs > std::numeric_limits<uintptr_t>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool reduceScatterByteSizes(
    size_t recvcount,
    int nRanks,
    commDataType_t datatype,
    size_t& recvBytes,
    size_t& totalBytes) {
  return checkedMultiply(recvcount, commTypeSize(datatype), recvBytes) &&
      checkedMultiply(recvBytes, static_cast<size_t>(nRanks), totalBytes);
}

bool rangesOverlap(
    uintptr_t lhs,
    size_t lhsBytes,
    uintptr_t rhs,
    size_t rhsBytes) {
  if (lhsBytes == 0 || rhsBytes == 0) {
    return false;
  }
  uintptr_t lhsEnd = 0;
  uintptr_t rhsEnd = 0;
  return checkedAdd(lhs, lhsBytes, lhsEnd) &&
      checkedAdd(rhs, rhsBytes, rhsEnd) && lhs < rhsEnd && rhs < lhsEnd;
}

bool isExactReduceScatterInPlace(
    uintptr_t sendbuff,
    uintptr_t recvbuff,
    size_t recvBytes,
    int rank) {
  size_t rankOffset = 0;
  uintptr_t rankSendbuff = 0;
  return checkedMultiply(static_cast<size_t>(rank), recvBytes, rankOffset) &&
      checkedAdd(sendbuff, rankOffset, rankSendbuff) &&
      rankSendbuff == recvbuff;
}

commResult_t validateDirectIbReduceScatter(
    const void* sendbuff,
    const void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm) {
  const auto* statex = comm->statex_.get();
  const int nRanks = statex->nRanks();

  if (datatype != commFloat32) {
    CLOGF(
        ERR,
        "ReduceScatter {} supports commFloat32 only; got {}",
        reduceScatterAlgoName(myAlgo),
        commDataTypeToString(datatype));
    return commInvalidArgument;
  }
  if (redOp != commSum) {
    CLOGF(
        ERR,
        "ReduceScatter {} supports commSum only; got {}",
        reduceScatterAlgoName(myAlgo),
        commOpToString(redOp));
    return commInvalidArgument;
  }
  if (nRanks <= 1) {
    CLOGF(
        ERR,
        "ReduceScatter {} requires multiple ranks, got nRanks={}",
        reduceScatterAlgoName(myAlgo),
        nRanks);
    return commInvalidArgument;
  }
  if (nRanks > comms::prims::kDirectReduceScatterIbMaxRanks) {
    CLOGF(
        ERR,
        "ReduceScatter {} nRanks={} exceeds max {}",
        reduceScatterAlgoName(myAlgo),
        nRanks,
        comms::prims::kDirectReduceScatterIbMaxRanks);
    return commInvalidArgument;
  }
  if (!comm->multiPeerTransport_) {
    CLOGF(
        ERR,
        "ReduceScatter {} requires MultiPeerTransport (NCCL_CTRAN_USE_PIPES=1)",
        reduceScatterAlgoName(myAlgo));
    return commInvalidArgument;
  }

  size_t recvBytes = 0;
  size_t totalBytes = 0;
  if (!reduceScatterByteSizes(
          recvcount, nRanks, datatype, recvBytes, totalBytes)) {
    CLOGF(
        ERR,
        "ReduceScatter {} byte size overflows size_t for recvcount={} nRanks={}",
        reduceScatterAlgoName(myAlgo),
        recvcount,
        nRanks);
    return commInvalidArgument;
  }
  const uintptr_t sendAddr = reinterpret_cast<uintptr_t>(sendbuff);
  const uintptr_t recvAddr = reinterpret_cast<uintptr_t>(recvbuff);
  uintptr_t sendEnd = 0;
  uintptr_t recvEnd = 0;
  if (!checkedAdd(sendAddr, totalBytes, sendEnd) ||
      !checkedAdd(recvAddr, recvBytes, recvEnd)) {
    CLOGF(
        ERR,
        "ReduceScatter {} buffer range overflows address space",
        reduceScatterAlgoName(myAlgo));
    return commInvalidArgument;
  }
  const bool inPlace = isExactReduceScatterInPlace(
      sendAddr, recvAddr, recvBytes, statex->rank());
  if (!inPlace && rangesOverlap(sendAddr, totalBytes, recvAddr, recvBytes)) {
    CLOGF(
        ERR,
        "ReduceScatter {} supports out-of-place buffers or exact ReduceScatter in-place aliasing only",
        reduceScatterAlgoName(myAlgo));
    return commInvalidArgument;
  }

  auto* mpt = comm->multiPeerTransport_.get();
  for (int peer = 0; peer < nRanks; ++peer) {
    if (peer == statex->rank()) {
      continue;
    }
    if (!mpt->has_ibgda(peer) || !mpt->prefers_ibgda(peer)) {
      CLOGF(
          ERR,
          "ReduceScatter {} requires preferred IBGDA transport for peer {}, has_ibgda={} prefers_ibgda={}",
          reduceScatterAlgoName(myAlgo),
          peer,
          mpt->has_ibgda(peer),
          mpt->prefers_ibgda(peer));
      return commInvalidArgument;
    }
  }

  return commSuccess;
}

} // namespace

commResult_t ctranReduceScatterDirectIb(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream) {
  CTRAN_COLL_INFO(
      reduceScatterAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      recvcount,
      datatype,
      -1,
      comm,
      stream);

  if (recvcount == 0) {
    comm->ctran_->updateOpCount();
    return commSuccess;
  }

  FB_COMMCHECK(validateDirectIbReduceScatter(
      sendbuff, recvbuff, recvcount, datatype, redOp, comm));

  const auto* statex = comm->statex_.get();
  const int nRanks = statex->nRanks();
  size_t recvBytes = 0;
  size_t totalBytes = 0;
  if (!reduceScatterByteSizes(
          recvcount, nRanks, datatype, recvBytes, totalBytes)) {
    return commInvalidArgument;
  }

  auto* mpt = comm->multiPeerTransport_.get();
  std::vector<int> peers;
  peers.reserve(static_cast<size_t>(nRanks - 1));
  for (int peer = 0; peer < nRanks; ++peer) {
    if (peer != statex->rank()) {
      peers.push_back(peer);
    }
  }

  try {
    mpt->materializePeers(peers);

    const int numBlocks =
        ctran::reducescatter::direct_ib::numBlocksForTotalBytes(totalBytes);

    comms::prims::DirectReduceScatterIbLaunchParams params{};
    params.my_rank = statex->rank();
    params.num_ranks = nRanks;
    params.chunk_elements = recvcount;
    params.signaling_data_size =
        ctran::reducescatter::direct_ib::signalingDataSize(recvBytes);
    params.input = static_cast<const float*>(sendbuff);
    params.output = static_cast<float*>(recvbuff);
    params.in_place = isExactReduceScatterInPlace(
        reinterpret_cast<uintptr_t>(sendbuff),
        reinterpret_cast<uintptr_t>(recvbuff),
        recvBytes,
        statex->rank());
    params.num_blocks = numBlocks;
    params.timeout_ms = ctran::reducescatter::direct_ib::kTimeoutMs;
    params.stream = stream;

    for (int peer : peers) {
      params.peers[peer] = comms::prims::P2pIbTransportDevice(
          mpt->get_p2p_ibgda_transport_device(peer));
    }

    comm->recordAlgoStats(
        "ReduceScatter", reduceScatterAlgoName(myAlgo), recvBytes);
    comms::prims::launch_direct_reduce_scatter_ib(params);
    FB_CUDACHECK(cudaGetLastError());
  } catch (const std::exception& e) {
    CLOGF(
        ERR,
        "ReduceScatter {} failed: {}",
        reduceScatterAlgoName(myAlgo),
        e.what());
    return commInternalError;
  }

  comm->ctran_->updateOpCount();
  return commSuccess;
}

#else // !ENABLE_PRIMS

#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/utils/logger/LogUtils.h"

commResult_t ctranReduceScatterDirectIb(
    const void* /*sendbuff*/,
    void* /*recvbuff*/,
    size_t /*recvcount*/,
    commDataType_t /*datatype*/,
    commRedOp_t /*redOp*/,
    CtranComm* /*comm*/,
    cudaStream_t /*stream*/) {
  CLOGF(ERR, "ReduceScatter CtranReduceScatterDirectIb requires ENABLE_PRIMS");
  return commInvalidArgument;
}

#endif // defined(ENABLE_PRIMS)
