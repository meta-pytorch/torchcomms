// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <chrono>
#include <optional>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PIPES)

#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/collectives/RingAllReduceLauncher.h"
#include "comms/pipes/collectives/RingUtils.h"

commResult_t ctranAllReducePipesRing(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout) {
  if (!comm->multiPeerTransport_) {
    CLOGF(
        ERR,
        "PipesRingAllReduce requires MultiPeerTransport "
        "(NCCL_CTRAN_USE_PIPES=1)");
    return commInternalError;
  }

  if (datatype != commDataType_t::commFloat32) {
    CLOGF(
        WARN,
        "PipesRingAllReduce currently supports float32 only, got {}",
        static_cast<int>(datatype));
    return commInvalidArgument;
  }

  if (redOp != commRedOp_t::commSum) {
    CLOGF(
        WARN,
        "PipesRingAllReduce currently supports Sum only, got {}",
        static_cast<int>(redOp));
    return commInvalidArgument;
  }

  if (count == 0) {
    return commSuccess;
  }

  auto* statex = comm->statex_.get();
  const int nRanks = statex->nRanks();
  const int rank = statex->rank();

  if (nRanks < 2) {
    if (sendbuff != recvbuff) {
      auto err = cudaMemcpyAsync(
          recvbuff,
          sendbuff,
          count * sizeof(float),
          cudaMemcpyDeviceToDevice,
          stream);
      if (err != cudaSuccess) {
        CLOGF(
            ERR,
            "PipesRingAllReduce cudaMemcpyAsync failed: {}",
            cudaGetErrorString(err));
        return commInternalError;
      }
    }
    return commSuccess;
  }

  if (count % nRanks != 0) {
    CLOGF(
        ERR,
        "PipesRingAllReduce requires count ({}) divisible by nRanks ({})",
        count,
        nRanks);
    return commInvalidArgument;
  }

  const size_t messageBytes = count * sizeof(float);

  bool enableBidirAg = false;
  if (nRanks > 2) {
    int64_t bidirMinSize = NCCL_CTRAN_ALLREDUCE_PIPES_BIDIR_AG_MIN_SIZE;
    if (bidirMinSize == -1) {
      enableBidirAg = false;
    } else if (bidirMinSize == 0) {
      enableBidirAg = true;
    } else if (bidirMinSize == -2) {
      enableBidirAg = (messageBytes >= 16UL * 1024 * 1024);
    } else if (bidirMinSize > 0) {
      enableBidirAg = (static_cast<int64_t>(messageBytes) >= bidirMinSize);
    }
  }

  int numRings;
  int numBlocks;
  if (NCCL_CTRAN_ALLREDUCE_PIPES_NUM_RINGS > 0) {
    numRings = NCCL_CTRAN_ALLREDUCE_PIPES_NUM_RINGS;
  } else if (messageBytes > 32UL * 1024 * 1024) {
    numRings = 2;
  } else {
    numRings = 1;
  }

  if (NCCL_CTRAN_ALLREDUCE_PIPES_NUM_BLOCKS > 0) {
    numBlocks = NCCL_CTRAN_ALLREDUCE_PIPES_NUM_BLOCKS;
  } else if (messageBytes <= 1024 * 1024) {
    numBlocks = 4;
  } else if (messageBytes <= 32UL * 1024 * 1024) {
    numBlocks = 8;
  } else {
    numBlocks = 16;
  }

  auto ringsOpt = comms::pipes::make_standard_rings(nRanks, rank, numRings);
  if (!ringsOpt.has_value()) {
    CLOGF(
        ERR,
        "PipesRingAllReduce: failed to build {} rings for {} ranks",
        numRings,
        nRanks);
    return commInvalidArgument;
  }
  const auto& rings = *ringsOpt;

  auto* mpt = comm->multiPeerTransport_.get();

  comms::pipes::RingAllReduceLaunchParams params{};
  params.my_rank = rank;
  params.num_ranks = nRanks;
  params.count = count;
  params.signaling_data_size = 0;
  params.input = static_cast<const float*>(sendbuff);
  params.output = static_cast<float*>(recvbuff);
  params.num_blocks = numBlocks;
  params.num_rings = numRings;
  params.stream = stream;
  params.enable_bidir_ag = enableBidirAg;

  if (timeout.has_value()) {
    params.timeout_ms = static_cast<float>(timeout->count());
  }

  for (int r = 0; r < numRings; r++) {
    params.rings[r].prev_rank = rings[r].prev_rank;
    params.rings[r].next_rank = rings[r].next_rank;
    params.rings[r].prev =
        mpt->get_p2p_ibgda_transport_device(rings[r].prev_rank);
    params.rings[r].next =
        mpt->get_p2p_ibgda_transport_device(rings[r].next_rank);
  }

  comms::pipes::launch_ring_allreduce(params);

  return commSuccess;
}

#else

commResult_t ctranAllReducePipesRing(
    const void*,
    void*,
    size_t,
    commDataType_t,
    commRedOp_t,
    CtranComm*,
    cudaStream_t,
    std::optional<std::chrono::milliseconds>) {
  return commInternalError;
}

#endif // ENABLE_PIPES
