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

commResult_t ctranAllReducePipesFlatRing(
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
        "PipesFlatRingAllReduce requires MultiPeerTransport "
        "(NCCL_CTRAN_USE_PIPES=1)");
    return commInternalError;
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
    const int numRings = 1;

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

    comms::pipes::RingAllReduceLaunchParams params{};
    params.my_rank = rank;
    params.num_ranks = nRanks;
    params.count = divisibleCount;
    params.signaling_data_size = 0;
    params.input = static_cast<const float*>(sendbuff);
    params.output = static_cast<float*>(recvbuff);
    params.num_blocks = 16;
    params.num_rings = numRings;
    params.stream = stream;

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
