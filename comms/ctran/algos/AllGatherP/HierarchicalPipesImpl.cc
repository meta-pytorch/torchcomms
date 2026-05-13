// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"

#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/utils/cvars/nccl_cvars.h"

#if defined(ENABLE_PIPES)

#include <algorithm>
#include <new>

#include "comms/ctran/algos/AllGatherP/HierarchicalPipes.cuh"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/collectives/DirectNvlTypes.h"

extern __global__ void ncclKernelAllGatherPHierarchicalPipes(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allgatherp::hierarchical_pipes::KernArgs args);

namespace {

constexpr const char* kAlgoName = "CtranAllGatherPHierarchicalPipes";
constexpr size_t kDirectFallbackMaxBytes = 4 * 1024 * 1024;

commResult_t validateAndFillRankGeometry(
    CtranComm* comm,
    comms::pipes::HierarchicalAllgatherFusedArgs& args) {
  auto* statex = comm->statex_.get();
  auto* transport = comm->multiPeerTransport_.get();
  if (transport == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "cthierarchical_pipes requires NCCL_CTRAN_USE_PIPES=1");
  }

  const int globalRank = statex->rank();
  const int nRanks = statex->nRanks();
  const int nvlSize = transport->nvl_n_ranks();
  const int nvlRank = transport->nvl_local_rank();
  if (nvlSize < 1 || nvlSize > comms::pipes::kDirectNvlMaxRanks) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "cthierarchical_pipes requires 1 <= nvl_size <= {} but got {}",
        comms::pipes::kDirectNvlMaxRanks,
        nvlSize);
  }
  if (nRanks % nvlSize != 0) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "cthierarchical_pipes requires rectangular rank geometry: nRanks {} "
        "must be divisible by nvl_size {}",
        nRanks,
        nvlSize);
  }
  if (globalRank % nvlSize != nvlRank) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "cthierarchical_pipes requires contiguous rank geometry: rank {} "
        "nvl_size {} implies nvl_rank {}, but pipes reports {}",
        globalRank,
        nvlSize,
        globalRank % nvlSize,
        nvlRank);
  }

  const int ibSize = nRanks / nvlSize;
  const int ibRank = globalRank / nvlSize;
  for (int peer = 0; peer < nvlSize; ++peer) {
    const int peerGlobal = ibRank * nvlSize + peer;
    try {
      if (transport->global_to_nvl_local(peerGlobal) != peer) {
        FB_ERRORRETURN(
            commInvalidArgument,
            "cthierarchical_pipes requires NVL local rank {} to map to "
            "global rank {}",
            peer,
            peerGlobal);
      }
    } catch (const std::exception& e) {
      FB_ERRORRETURN(
          commInvalidArgument,
          "cthierarchical_pipes rank geometry validation failed for global "
          "rank {}: {}",
          peerGlobal,
          e.what());
    }
  }

  args.ib_rank = ibRank;
  args.ib_size = ibSize;
  args.nvl_rank = nvlRank;
  args.nvl_size = nvlSize;
  return commSuccess;
}

commResult_t fillTransportArgs(
    CtranComm* comm,
    comms::pipes::HierarchicalAllgatherFusedArgs& args) {
  auto* transport = comm->multiPeerTransport_.get();
  if (transport == nullptr) {
    return ErrorStackTraceUtil::log(commInvalidArgument);
  }

  const int ibRank = args.ib_rank;
  const int ibSize = args.ib_size;
  const int nvlRank = args.nvl_rank;
  const int nvlSize = args.nvl_size;

  if (ibSize > 1) {
    const int prevIbRank = (ibRank - 1 + ibSize) % ibSize;
    const int nextIbRank = (ibRank + 1) % ibSize;
    const int prevGlobal = prevIbRank * nvlSize + nvlRank;
    const int nextGlobal = nextIbRank * nvlSize + nvlRank;
    if (!transport->has_ibgda(prevGlobal) ||
        !transport->has_ibgda(nextGlobal)) {
      FB_ERRORRETURN(
          commInvalidArgument,
          "cthierarchical_pipes requires IBGDA prev/next peers. "
          "prevGlobal={} has_ibgda={} nextGlobal={} has_ibgda={}",
          prevGlobal,
          transport->has_ibgda(prevGlobal),
          nextGlobal,
          transport->has_ibgda(nextGlobal));
    }
    args.ib_ring.prev_rank = prevIbRank;
    args.ib_ring.next_rank = nextIbRank;
    args.ib_ring.prev = transport->get_p2p_ibgda_transport_device(prevGlobal);
    args.ib_ring.next = transport->get_p2p_ibgda_transport_device(nextGlobal);
    if (args.ib_ring.prev == nullptr || args.ib_ring.next == nullptr) {
      FB_ERRORRETURN(
          commInvalidArgument,
          "cthierarchical_pipes received null IBGDA transport handle");
    }
  }

  for (int peer = 0; peer < nvlSize; ++peer) {
    if (peer == nvlRank) {
      continue;
    }
    const int peerGlobal = args.ib_rank * nvlSize + peer;
    if (!transport->is_nvl_peer(peerGlobal)) {
      FB_ERRORRETURN(
          commInvalidArgument,
          "cthierarchical_pipes requires global rank {} to be an NVL peer",
          peerGlobal);
    }
    new (&args.nvl_peers[peer]) comms::pipes::P2pNvlTransportDevice(
        transport->get_p2p_nvl_transport_device(peerGlobal));
  }

  return commSuccess;
}

} // namespace

#endif // ENABLE_PIPES

namespace ctran::allgatherp {

commResult_t AlgoImpl::execHierarchicalPipes(
    const void* sendbuff,
    const size_t count,
    const commDataType_t datatype) {
#if defined(ENABLE_PIPES)
  FB_COMMCHECK(waitInit());

  const size_t typeSize = commTypeSize(datatype);
  const size_t sendBytes = count * typeSize;
  if (sendBytes <= kDirectFallbackMaxBytes) {
    return execDirect(sendbuff, count, datatype);
  }

  hierarchical_pipes::KernArgs kernArgs{};
  auto& args = kernArgs.args;
  FB_COMMCHECK(validateAndFillRankGeometry(comm_, args));
  args.sendcount = sendBytes;
  args.ib_signaling_data_size = NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES;
  args.nvl_signaling_data_size = NCCL_CTRAN_HIER_AG_NVL_SIGNAL_BYTES;
  args.sendbuf = static_cast<const char*>(sendbuff);
  args.recvbuf = static_cast<char*>(pArgs.recvbuff);
  FB_COMMCHECK(fillTransportArgs(comm_, args));
  if (NCCL_CTRAN_HIER_AG_TIMEOUT_MS > 0) {
    int device = 0;
    FB_CUDACHECK(cudaGetDevice(&device));
    kernArgs.timeout = comms::pipes::makeTimeout(
        static_cast<uint32_t>(NCCL_CTRAN_HIER_AG_TIMEOUT_MS), device);
  }

  const auto opCount = comm_->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      kAlgoName, sendbuff, pArgs.recvbuff, count, datatype, -1, comm_, stream_);

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP, stream_, kAlgoName, opCount);
  config.numBlocks = static_cast<unsigned int>(
      std::max<int64_t>(1, NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS));
  config.numThreads = hierarchical_pipes::kBlockSize;
  config.dynamicSharedMemBytes = 1;
  config.args.devState_d = comm_->ctran_->algo->getDevState();
  config.algoArgs = &kernArgs;

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  FB_COMMCHECK(comm_->ctran_->gpe->submit(
      std::move(opGroup),
      nullptr,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherPHierarchicalPipes)));
  return commSuccess;
#else
  (void)sendbuff;
  (void)count;
  (void)datatype;
  return ErrorStackTraceUtil::log(commInvalidArgument);
#endif
}

} // namespace ctran::allgatherp
