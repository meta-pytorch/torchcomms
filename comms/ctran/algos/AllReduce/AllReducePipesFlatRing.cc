// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PIPES)

#include "comms/ctran/algos/AllReduce/AllReducePipesFlatRing.cuh"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/collectives/RingUtils.h"

namespace {

bool isSupportedRingCount(int numRings) {
  return numRings == 1 || numRings == 2 || numRings == 4;
}

int getPipesFlatRingNumRings() {
  if (NCCL_CTRAN_ALLREDUCE_PIPES_FLAT_RING_NUM_RINGS == -1) {
    return 1;
  }
  return NCCL_CTRAN_ALLREDUCE_PIPES_FLAT_RING_NUM_RINGS;
}

int getPipesFlatRingNumBlocks() {
  if (NCCL_CTRAN_ALLREDUCE_PIPES_FLAT_RING_NUM_BLOCKS == -1) {
    return 16;
  }
  return NCCL_CTRAN_ALLREDUCE_PIPES_FLAT_RING_NUM_BLOCKS;
}

commResult_t makePipesTimeout(
    std::optional<std::chrono::milliseconds> timeout,
    comms::pipes::Timeout& pipesTimeout) {
  pipesTimeout = comms::pipes::Timeout();
  if (!timeout.has_value() || timeout->count() <= 0) {
    return commSuccess;
  }

  if (timeout->count() > std::numeric_limits<uint32_t>::max()) {
    CLOGF(
        WARN,
        "PipesFlatRingAllReduce timeout {}ms exceeds supported maximum",
        timeout->count());
    return commInvalidArgument;
  }

  int device = 0;
  const auto err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    CLOGF(
        ERR,
        "PipesFlatRingAllReduce cudaGetDevice failed while creating timeout: {}",
        cudaGetErrorString(err));
    return commInternalError;
  }

  try {
    pipesTimeout = comms::pipes::makeTimeout(
        static_cast<uint32_t>(timeout->count()), device);
  } catch (const std::exception& ex) {
    CLOGF(
        ERR,
        "PipesFlatRingAllReduce failed to create Pipes timeout: {}",
        ex.what());
    return commInternalError;
  }
  return commSuccess;
}

commResult_t submitPipesFlatRingPhase(
    CtranComm* comm,
    cudaStream_t stream,
    const char* phaseName,
    uint64_t opCount,
    ctran::allreduce::pipesflatring::KernArgs& kernArgs,
    const void* kernel,
    std::optional<std::chrono::milliseconds> timeout) {
  KernelConfig config(
      KernelConfig::KernelType::ALLREDUCE, stream, phaseName, opCount);
  config.numBlocks = static_cast<unsigned int>(kernArgs.numBlocks);
  config.numThreads = ctran::allreduce::pipesflatring::kBlockSize;
  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.algoArgs = &kernArgs;

  config.args.collective.allreduce.sendbuff = kernArgs.sendbuff;
  config.args.collective.allreduce.recvbuff = kernArgs.recvbuff;
  config.args.collective.allreduce.redOp = commRedOp_t::commSum;
  config.args.collective.allreduce.count = kernArgs.count;
  config.args.collective.allreduce.datatype = commDataType_t::commFloat32;

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup), nullptr, config, kernel, timeout));
  return commSuccess;
}

} // namespace

commResult_t ctranAllReducePipesFlatRing(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout) {
  if (comm == nullptr || comm->ctran_ == nullptr ||
      comm->ctran_->gpe == nullptr || comm->ctran_->algo == nullptr ||
      comm->statex_ == nullptr) {
    CLOGF(ERR, "PipesFlatRingAllReduce requires initialized Ctran state");
    return commInternalError;
  }

  if (!comm->multiPeerTransport_) {
    CLOGF(
        ERR,
        "PipesFlatRingAllReduce requires MultiPeerTransport "
        "(NCCL_CTRAN_USE_PIPES=1)");
    return commInternalError;
  }

  if (!NCCL_CTRAN_IBGDA_SENDRECV_ENABLE) {
    CLOGF(
        WARN,
        "PipesFlatRingAllReduce requires NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1");
    return commInvalidArgument;
  }

  if (datatype != commDataType_t::commFloat32) {
    CLOGF(
        WARN,
        "PipesFlatRingAllReduce currently supports float32 only, got {}",
        static_cast<int>(datatype));
    return commInvalidArgument;
  }

  if (redOp != commRedOp_t::commSum) {
    CLOGF(
        WARN,
        "PipesFlatRingAllReduce currently supports Sum only, got {}",
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
            "PipesFlatRingAllReduce cudaMemcpyAsync failed: {}",
            cudaGetErrorString(err));
        return commInternalError;
      }
    }
    return commSuccess;
  }

  const size_t divisibleCount = (count / nRanks) * nRanks;
  const size_t remainder = count - divisibleCount;

  if (divisibleCount > 0) {
    const int numRings = getPipesFlatRingNumRings();
    if (!isSupportedRingCount(numRings)) {
      CLOGF(
          WARN,
          "PipesFlatRingAllReduce unsupported num_rings={} (supported: 1, 2, 4)",
          numRings);
      return commInvalidArgument;
    }

    if (numRings > ctran::allreduce::pipesflatring::kMaxRings) {
      CLOGF(
          WARN,
          "PipesFlatRingAllReduce num_rings={} exceeds Ctran flat-ring max {}",
          numRings,
          ctran::allreduce::pipesflatring::kMaxRings);
      return commInvalidArgument;
    }

    const int numBlocks = getPipesFlatRingNumBlocks();
    if (numBlocks <= 0) {
      CLOGF(
          WARN,
          "PipesFlatRingAllReduce num_blocks must be positive, got {}",
          numBlocks);
      return commInvalidArgument;
    }

    if (NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS < numBlocks) {
      CLOGF(
          WARN,
          "PipesFlatRingAllReduce num_blocks={} exceeds NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS={}",
          numBlocks,
          NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS);
      return commInvalidArgument;
    }

    const auto dataBufferSize = NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE;
    if (dataBufferSize == 0) {
      CLOGF(
          WARN,
          "PipesFlatRingAllReduce requires positive NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE");
      return commInvalidArgument;
    }

    if (dataBufferSize / static_cast<uint64_t>(numBlocks) < 16) {
      CLOGF(
          WARN,
          "PipesFlatRingAllReduce dataBufferSize={} is too small for num_blocks={}",
          dataBufferSize,
          numBlocks);
      return commInvalidArgument;
    }

    auto ringsOpt = comms::pipes::make_standard_rings(nRanks, rank, numRings);
    if (!ringsOpt.has_value()) {
      CLOGF(
          ERR,
          "PipesFlatRingAllReduce: failed to build {} rings for {} ranks",
          numRings,
          nRanks);
      return commInvalidArgument;
    }
    const auto& rings = *ringsOpt;

    auto* mpt = comm->multiPeerTransport_.get();

    comms::pipes::Timeout pipesTimeout;
    FB_COMMCHECK(makePipesTimeout(timeout, pipesTimeout));

    ctran::allreduce::pipesflatring::KernArgs kernArgs{};
    kernArgs.rank = rank;
    kernArgs.nRanks = nRanks;
    kernArgs.count = divisibleCount;
    kernArgs.chunkElements = divisibleCount / nRanks;
    kernArgs.sendbuff = static_cast<const float*>(sendbuff);
    kernArgs.recvbuff = static_cast<float*>(recvbuff);
    kernArgs.numBlocks = numBlocks;
    kernArgs.numRings = numRings;
    kernArgs.signalingDataSize = 0;
    kernArgs.timeout = pipesTimeout;
    for (int r = 0; r < numRings; r++) {
      kernArgs.rings[r].prevRank = rings[r].prev_rank;
      kernArgs.rings[r].nextRank = rings[r].next_rank;
      kernArgs.rings[r].prev =
          mpt->get_p2p_ibgda_transport_device(rings[r].prev_rank);
      kernArgs.rings[r].next =
          mpt->get_p2p_ibgda_transport_device(rings[r].next_rank);
    }

    const auto opCount = comm->ctran_->getOpCount();
    FB_COMMCHECK(submitPipesFlatRingPhase(
        comm,
        stream,
        "PipesFlatRingReduceScatter",
        opCount,
        kernArgs,
        reinterpret_cast<void*>(ctranKernelAllReducePipesFlatRingReduceScatter),
        timeout));
    FB_COMMCHECK(submitPipesFlatRingPhase(
        comm,
        stream,
        "PipesFlatRingAllGather",
        opCount,
        kernArgs,
        reinterpret_cast<void*>(ctranKernelAllReducePipesFlatRingAllGather),
        timeout));
  }

  if (remainder > 0) {
    const size_t elemSize = sizeof(float);
    const void* remainderSendbuff =
        static_cast<const char*>(sendbuff) + divisibleCount * elemSize;
    void* remainderRecvbuff =
        static_cast<char*>(recvbuff) + divisibleCount * elemSize;
    return ctranAllReduceDirect(
        remainderSendbuff,
        remainderRecvbuff,
        remainder,
        datatype,
        redOp,
        comm,
        stream,
        timeout);
  }

  return commSuccess;
}

#else

commResult_t ctranAllReducePipesFlatRing(
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
