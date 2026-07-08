// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if defined(ENABLE_PRIMS)

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/CtranPipes.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/prims/collectives/AllGatherLauncher.h"
#include "comms/prims/collectives/RingUtils.h"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

static const auto myAlgo = NCCL_ALLGATHER_ALGO::cthierarchical_ring;
using ::meta::comms::colltrace::CollTraceHandleTriggerState;

namespace {

constexpr size_t kHierarchicalAgOverlapChunk512KiB = 512 * 1024;
constexpr size_t kHierarchicalAgOverlapChunk1MiB = 1024 * 1024;
constexpr size_t kHierarchicalAgOverlapChunk2MiB = 2 * 1024 * 1024;
constexpr size_t kHierarchicalAgOverlapChunk4MiB = 4 * 1024 * 1024;
constexpr size_t kHierarchicalAgOverlapChunk8MiB = 8 * 1024 * 1024;

bool rangesOverlap(
    uintptr_t lhs,
    size_t lhsBytes,
    uintptr_t rhs,
    size_t rhsBytes) {
  if (lhsBytes == 0 || rhsBytes == 0) {
    return false;
  }
  return lhs < rhs + rhsBytes && rhs < lhs + lhsBytes;
}

bool isAllGatherInPlace(
    const void* sendbuff,
    const void* recvbuff,
    size_t sendBytes,
    int rank,
    int nRanks) {
  const uintptr_t sendPtr = reinterpret_cast<uintptr_t>(sendbuff);
  const uintptr_t recvPtr = reinterpret_cast<uintptr_t>(recvbuff);
  const size_t recvBytes = sendBytes * static_cast<size_t>(nRanks);
  const uintptr_t ownRecvPtr = recvPtr + static_cast<size_t>(rank) * sendBytes;
  const bool buffersOverlap =
      rangesOverlap(sendPtr, sendBytes, recvPtr, recvBytes);
  return buffersOverlap && sendPtr == ownRecvPtr;
}

commResult_t validateAllGatherBufferLayout(
    const void* sendbuff,
    const void* recvbuff,
    size_t sendBytes,
    int rank,
    int nRanks) {
  const uintptr_t sendPtr = reinterpret_cast<uintptr_t>(sendbuff);
  const uintptr_t recvPtr = reinterpret_cast<uintptr_t>(recvbuff);
  const size_t recvBytes = sendBytes * static_cast<size_t>(nRanks);
  const uintptr_t ownRecvPtr = recvPtr + static_cast<size_t>(rank) * sendBytes;

  if (rangesOverlap(sendPtr, sendBytes, recvPtr, recvBytes) &&
      sendPtr != ownRecvPtr) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} invalid buffer layout: sendbuff range [0x{:x}, 0x{:x}) overlaps recvbuff range [0x{:x}, 0x{:x}) but does not start at this rank's recv slot 0x{:x}. Use sendbuff=recvbuff+rank*sendcount*sizeof(datatype) for in-place allgather, or use a disjoint send buffer for out-of-place allgather.",
        allGatherAlgoName(myAlgo),
        sendPtr,
        sendPtr + sendBytes,
        recvPtr,
        recvPtr + recvBytes,
        ownRecvPtr);
    return commInvalidArgument;
  }

  return commSuccess;
}

size_t selectHierarchicalAgOverlapChunkBytes(size_t sendBytes) {
  const size_t targetChunkBytes = (sendBytes + 7) / 8;
  if (targetChunkBytes <= kHierarchicalAgOverlapChunk512KiB) {
    return kHierarchicalAgOverlapChunk512KiB;
  }
  if (targetChunkBytes <= kHierarchicalAgOverlapChunk1MiB) {
    return kHierarchicalAgOverlapChunk1MiB;
  }
  if (targetChunkBytes <= kHierarchicalAgOverlapChunk2MiB) {
    return kHierarchicalAgOverlapChunk2MiB;
  }
  if (targetChunkBytes <= kHierarchicalAgOverlapChunk4MiB) {
    return kHierarchicalAgOverlapChunk4MiB;
  }
  return kHierarchicalAgOverlapChunk8MiB;
}

commResult_t validateHierarchicalRingParams(CtranComm* comm, int numBlocks) {
  if (!NCCL_CTRAN_IBGDA_SENDRECV_ENABLE) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} requires NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1",
        allGatherAlgoName(myAlgo));
    return commInvalidArgument;
  }

  const auto dataBufferSize = NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE;
  if (dataBufferSize == 0) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} requires a positive IBGDA data-buffer size via NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE or per-communicator pipesIbgdaDataBufferSize",
        allGatherAlgoName(myAlgo));
    return commInvalidArgument;
  }
  if (numBlocks <= 0) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} requires positive NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS; got {}",
        allGatherAlgoName(myAlgo),
        numBlocks);
    return commInvalidArgument;
  }
  if (numBlocks > NCCL_CTRAN_IB_MAX_GROUPS) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} numBlocks={} exceeds NCCL_CTRAN_IB_MAX_GROUPS={}",
        allGatherAlgoName(myAlgo),
        numBlocks,
        NCCL_CTRAN_IB_MAX_GROUPS);
    return commInvalidArgument;
  }
  if (dataBufferSize / static_cast<uint64_t>(numBlocks) < 16) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} dataBufferSize={} is too small for numBlocks={}",
        allGatherAlgoName(myAlgo),
        dataBufferSize,
        numBlocks);
    return commInvalidArgument;
  }

  const auto* statex = comm->statex_.get();
  const int nRanks = statex->nRanks();
  const int nNodes = statex->nNodes();
  const int nLocalRanks = statex->nLocalRanks();
  if (nRanks <= 1 || nNodes <= 1) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} requires multiple nodes, got nRanks={} nNodes={}",
        allGatherAlgoName(myAlgo),
        nRanks,
        nNodes);
    return commInvalidArgument;
  }
  if (nLocalRanks < 1 || nRanks != static_cast<int64_t>(nNodes) * nLocalRanks) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} requires rectangular rank geometry, got nRanks={} nNodes={} nLocalRanks={}",
        allGatherAlgoName(myAlgo),
        nRanks,
        nNodes,
        nLocalRanks);
    return commInvalidArgument;
  }
  if (nLocalRanks > comms::prims::kDirectNvlMaxRanks) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} nLocalRanks={} exceeds direct NVLink peer capacity {}",
        allGatherAlgoName(myAlgo),
        nLocalRanks,
        comms::prims::kDirectNvlMaxRanks);
    return commInvalidArgument;
  }

  for (int node = 0; node < nNodes; ++node) {
    for (int localRank = 0; localRank < nLocalRanks; ++localRank) {
      const int expectedRank = node * nLocalRanks + localRank;
      const int rank = statex->localRankToRank(localRank, node);
      if (rank != expectedRank) {
        CLOGF_SUBSYS(
            WARN,
            COLL,
            "AllGather {} requires contiguous node-major rank layout; localRankToRank({}, {})={} expected {}",
            allGatherAlgoName(myAlgo),
            localRank,
            node,
            rank,
            expectedRank);
        return commInvalidArgument;
      }
    }
  }

  if (!comm->multiPeerTransport_) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} requires MultiPeerTransport (NCCL_CTRAN_USE_PIPES=1)",
        allGatherAlgoName(myAlgo));
    return commInvalidArgument;
  }

  auto* mpt = comm->multiPeerTransport_.get();
  const int myNode = statex->node();
  const int localRank = statex->localRank();
  auto rings = comms::prims::make_standard_rings(nNodes, myNode, 1);
  if (!rings.has_value()) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} cannot construct IB ring for {} nodes",
        allGatherAlgoName(myAlgo),
        nNodes);
    return commInvalidArgument;
  }
  const auto& ibRing = (*rings)[0];
  const int prevGlobal = ibRing.prev_rank * nLocalRanks + localRank;
  const int nextGlobal = ibRing.next_rank * nLocalRanks + localRank;
  if (!mpt->has_ibgda(prevGlobal) || !mpt->has_ibgda(nextGlobal)) {
    CLOGF_SUBSYS(
        WARN,
        COLL,
        "AllGather {} requires IBGDA prev/next transports; prev {} has_ibgda={} next {} has_ibgda={}",
        allGatherAlgoName(myAlgo),
        prevGlobal,
        mpt->has_ibgda(prevGlobal),
        nextGlobal,
        mpt->has_ibgda(nextGlobal));
    return commInvalidArgument;
  }

  for (int peerLocal = 0; peerLocal < nLocalRanks; ++peerLocal) {
    if (peerLocal == localRank) {
      continue;
    }
    const int peerGlobal = statex->localRankToRank(peerLocal);
    if (!mpt->is_nvl_peer(peerGlobal)) {
      CLOGF_SUBSYS(
          WARN,
          COLL,
          "AllGather {} requires NVLink transport to local peer globalRank={} localRank={}",
          allGatherAlgoName(myAlgo),
          peerGlobal,
          peerLocal);
      return commInvalidArgument;
    }
  }

  return commSuccess;
}

commResult_t ensureReadyCounterCapacity(
    CtranComm* comm,
    size_t counterCount,
    cudaStream_t stream) {
  if (counterCount == 0) {
    return commSuccess;
  }
  if (comm->hierarchicalAgReadyCounterCount_ >= counterCount &&
      comm->hierarchicalAgReadyCounters_ != nullptr) {
    return commSuccess;
  }

  if (comm->hierarchicalAgReadyCounters_ != nullptr) {
    FB_CUDACHECK(cudaFree(comm->hierarchicalAgReadyCounters_));
    comm->hierarchicalAgReadyCounters_ = nullptr;
    comm->hierarchicalAgReadyCounterCount_ = 0;
  }

  void* ptr = nullptr;
  FB_CUDACHECK(cudaMalloc(&ptr, counterCount * sizeof(uint64_t)));
  comm->hierarchicalAgReadyCounters_ = static_cast<uint64_t*>(ptr);
  comm->hierarchicalAgReadyCounterCount_ = counterCount;
  FB_CUDACHECK(cudaMemsetAsync(
      comm->hierarchicalAgReadyCounters_,
      0,
      counterCount * sizeof(uint64_t),
      stream));
  return commSuccess;
}

} // namespace

commResult_t ctranAllGatherHierarchicalRing(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  CTRAN_COLL_INFO(
      allGatherAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      -1,
      comm,
      stream);

  const auto* statex = comm->statex_.get();
  const int nRanks = statex->nRanks();
  const int nNodes = statex->nNodes();
  const int nLocalRanks = statex->nLocalRanks();
  const int rank = statex->rank();
  const int localRank = statex->localRank();
  const int node = statex->node();
  const size_t sendBytes = sendcount * commTypeSize(datatype);

  if (nRanks == 1) {
    if (sendBytes > 0 && sendbuff != recvbuff) {
      FB_CUDACHECK(cudaMemcpyAsync(
          recvbuff, sendbuff, sendBytes, cudaMemcpyDefault, stream));
    }
    comm->ctran_->updateOpCount();
    return commSuccess;
  }
  if (sendBytes == 0) {
    comm->ctran_->updateOpCount();
    return commSuccess;
  }

  FB_COMMCHECK(validateAllGatherBufferLayout(
      sendbuff, recvbuff, sendBytes, rank, nRanks));
  const bool inPlace =
      isAllGatherInPlace(sendbuff, recvbuff, sendBytes, rank, nRanks);

  const bool useOverlap = NCCL_CTRAN_HIER_AG_OVERLAP_ENABLE && nLocalRanks > 1;
  const int numBlocks = static_cast<int>(NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS);
  const int nvlNumBlocks =
      NCCL_CTRAN_MAX_NBLOCKS > 0 ? static_cast<int>(NCCL_CTRAN_MAX_NBLOCKS) : 1;
  FB_COMMCHECK(validateHierarchicalRingParams(comm, numBlocks));

  auto* mpt = comm->multiPeerTransport_.get();
  auto ibRings = comms::prims::make_standard_rings(nNodes, node, 1);
  if (!ibRings.has_value()) {
    return commInvalidArgument;
  }

  comms::prims::HierarchicalAllgatherLaunchParams params{};
  params.num_ranks = nRanks;
  params.ib_rank = node;
  params.ib_size = nNodes;
  params.nvl_rank = localRank;
  params.nvl_size = nLocalRanks;
  // Pipes allgather kernels operate on byte-addressed char buffers.
  params.sendcount = sendBytes;
  params.ib_signaling_data_size =
      static_cast<size_t>(NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES);
  params.nvl_signaling_data_size =
      static_cast<size_t>(NCCL_CTRAN_HIER_AG_NVL_SIGNAL_BYTES);
  params.sendbuf = static_cast<const char*>(sendbuff);
  params.recvbuf = static_cast<char*>(recvbuff);
  params.in_place = inPlace;
  params.ib_num_blocks = numBlocks;
  params.timeout_ms = NCCL_CTRAN_HIER_AG_TIMEOUT_MS;
  params.stream = stream;

  const auto& ibRing = (*ibRings)[0];
  const int prevGlobal = ibRing.prev_rank * nLocalRanks + localRank;
  const int nextGlobal = ibRing.next_rank * nLocalRanks + localRank;
  params.ib_ring.prev_rank = ibRing.prev_rank;
  params.ib_ring.next_rank = ibRing.next_rank;
  params.ib_ring.prev = mpt->get_p2p_ibgda_transport_device(prevGlobal);
  params.ib_ring.next = mpt->get_p2p_ibgda_transport_device(nextGlobal);

  for (int peerLocal = 0; peerLocal < nLocalRanks; ++peerLocal) {
    if (peerLocal == localRank) {
      continue;
    }
    const int peerGlobal = statex->localRankToRank(peerLocal);
    new (&params.nvl_peers[peerLocal]) comms::prims::P2pNvlTransportDevice(
        mpt->get_p2p_nvl_transport_device(peerGlobal));
  }

  KernelConfig config(
      KernelConfig::KernelType::ALLGATHER,
      stream,
      allGatherAlgoName(myAlgo),
      comm->ctran_->getOpCount());
  ctranKernelSetAllGatherArgs(
      sendbuff,
      recvbuff,
      datatype,
      sendcount,
      comm->ctran_->algo->getDevState(),
      &config.args);

  std::vector<std::unique_ptr<struct OpElem>> emptyOpGroup;
  auto colltraceHandle =
      meta::comms::colltrace::getCollTraceHandle(comm, emptyOpGroup, config);
  if (colltraceHandle != nullptr) {
    colltraceHandle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  }
  comm->recordAlgoStats(
      "AllGather",
      allGatherAlgoName(myAlgo),
      sendcount * commTypeSize(datatype));

  if (useOverlap) {
    comms::prims::HierarchicalAllgatherOverlapLaunchParams overlapParams{};
    overlapParams.num_ranks = params.num_ranks;
    overlapParams.ib_rank = params.ib_rank;
    overlapParams.ib_size = params.ib_size;
    overlapParams.nvl_rank = params.nvl_rank;
    overlapParams.nvl_size = params.nvl_size;
    overlapParams.sendcount = params.sendcount;
    overlapParams.ib_signaling_data_size = params.ib_signaling_data_size;
    overlapParams.nvl_signaling_data_size = params.nvl_signaling_data_size;
    const size_t configuredChunkBytes =
        static_cast<size_t>(NCCL_CTRAN_HIER_AG_OVERLAP_CHUNK_BYTES);
    overlapParams.chunk_bytes = configuredChunkBytes != 0
        ? configuredChunkBytes
        : selectHierarchicalAgOverlapChunkBytes(sendBytes);
    overlapParams.ready_sequence = comm->ctran_->getOpCount() + 1;
    overlapParams.sendbuf = params.sendbuf;
    overlapParams.recvbuf = params.recvbuf;
    overlapParams.in_place = params.in_place;
    overlapParams.ib_num_blocks = params.ib_num_blocks;
    overlapParams.nvl_num_blocks = nvlNumBlocks;
    overlapParams.timeout_ms = params.timeout_ms;
    overlapParams.stream = params.stream;
    overlapParams.ib_ring = params.ib_ring;
    for (int peer = 0; peer < nLocalRanks; ++peer) {
      if (peer == localRank) {
        continue;
      }
      new (&overlapParams.nvl_peers[peer])
          comms::prims::P2pNvlTransportDevice(params.nvl_peers[peer]);
    }

    const size_t totalChunks =
        (sendBytes + overlapParams.chunk_bytes - 1) / overlapParams.chunk_bytes;
    const size_t readyCounters = static_cast<size_t>(nNodes) * totalChunks;
    FB_COMMCHECK(
        ensureReadyCounterCapacity(comm, readyCounters, overlapParams.stream));
    overlapParams.ready_counters = comm->hierarchicalAgReadyCounters_;
    FB_COMMCHECK(ctran::ctranPreparePipesTrace(comm, overlapParams.trace));
    comms::prims::launch_hierarchical_allgather_overlap(overlapParams);
  } else {
    comms::prims::launch_hierarchical_allgather_fused(params);
  }
  FB_CUDACHECK(cudaGetLastError());

  if (colltraceHandle != nullptr) {
    colltraceHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
  }
  comm->ctran_->updateOpCount();

  return commSuccess;
}

#else // !ENABLE_PRIMS

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"

commResult_t ctranAllGatherHierarchicalRing(
    const void* /*sendbuff*/,
    void* /*recvbuff*/,
    size_t /*sendcount*/,
    commDataType_t /*datatype*/,
    CtranComm* /*comm*/,
    cudaStream_t /*stream*/) {
  return commInternalError;
}

#endif // ENABLE_PRIMS
