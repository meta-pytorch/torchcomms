// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <iostream>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::alltoallvp {
inline void* getPtr(void* base, size_t offset) {
  return (void*)((uintptr_t)base + offset);
}

inline commResult_t nvlBarrier(CtranComm* comm, cudaStream_t stream) {
  const auto statex = comm->statex_.get();
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  // FIXME: needs to add cudaGraph capture support

  // Barrier to make sure all local ranks is ready to start intranode comm
  std::array<void*, 3> kernelArgs;
  kernelArgs.at(0) = (void*)&localRank;
  kernelArgs.at(1) = (void*)&nLocalRanks;
  auto devState_d = comm->ctran_->algo->getDevState();
  kernelArgs.at(2) = (void*)&devState_d;
  dim3 grid = {1, 1, 1};
  dim3 blocks = {1, 1, 1};
  FB_CUDACHECK(cudaFuncSetAttribute(
      reinterpret_cast<void*>(ncclKernelNvlBarrier),
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(CtranAlgoDeviceState)));
  FB_CUDACHECK(cudaLaunchKernel(
      reinterpret_cast<void*>(ncclKernelNvlBarrier),
      grid,
      blocks,
      kernelArgs.data(),
      sizeof(CtranAlgoDeviceState),
      stream));
  return commSuccess;
}

inline commResult_t nvlCeBcast(
    CtranComm* comm,
    const void* sendBuff,
    const size_t sendCounts[],
    const size_t sDispls[],
    const commDataType_t datatype,
    PersistArgs& pArgs,
    cudaStream_t stream,
    bool barrier = true) {
  const auto statex = comm->statex_.get();
  const auto rank = statex->rank();
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  // Barrier to make sure all local ranks has arrived before CE bcast, to
  // avoid unwanted incast traffic congestion
  if (barrier) {
    nvlBarrier(comm, stream);
  }

  auto mapper = comm->ctran_->mapper.get();

  // Copy data to other local ranks
  for (auto r = 1; r < nLocalRanks; r++) {
    const auto localPeer = (localRank + r) % nLocalRanks;
    const auto peer = statex->localRankToRank(localPeer);

    auto sendSize = sendCounts[peer] * commTypeSize(datatype);
    auto sendOffset = sDispls[peer] * commTypeSize(datatype);

    if ((sendSize > 0) &&
        (pArgs.remoteAccessKeys[peer].backend == CtranMapperBackend::NVL)) {
      auto sendPtr = getPtr(const_cast<void*>(sendBuff), sendOffset);
      auto recvPtr = getPtr(pArgs.remoteRecvBuffs[peer], 0);

      CLOGF_TRACE(
          COLL,
          "Rank {} CE copy to peer {}, sendPtr {} (sendBuff {} + sendOffset {}) -> remoteRecvBuff {}, sendSize {}",
          rank,
          peer,
          sendPtr,
          sendBuff,
          sendOffset,
          recvPtr,
          sendSize);

      FB_COMMCHECK(mapper->icopy(recvPtr, sendPtr, sendSize, stream));
    }
  }
  return commSuccess;
}

// Copy data to self for out-of-place AllToAllv. No-op if it is an in-place
// AllToAllv.
inline commResult_t copyToSelf(
    CtranComm* comm,
    const void* sendBuff,
    void* recvBuff,
    const size_t sendCount,
    const size_t sendOffset,
    const size_t recvOffset,
    const commDataType_t datatype,
    cudaStream_t stream) {
  const auto statex = comm->statex_.get();
  const auto rank = statex->rank();

  // Copy data to self for out-of-place AllToAllv
  auto sendSize = sendCount * commTypeSize(datatype);
  auto sendPtr =
      getPtr(const_cast<void*>(sendBuff), sendOffset * commTypeSize(datatype));
  auto recvPtr = getPtr(recvBuff, recvOffset * commTypeSize(datatype));

  if (recvPtr != sendPtr) {
    CLOGF_TRACE(
        COLL,
        "Rank {} CE copy to self, sendPtr {} (sendBuff {} + sendOffset {}) -> recvPtr {} (recvBuff {} + recvOffset {}), sendSize {}",
        rank,
        sendPtr,
        sendBuff,
        sendOffset,
        recvPtr,
        recvBuff,
        recvOffset,
        sendSize);
    FB_COMMCHECK(
        comm->ctran_->mapper->icopy(recvPtr, sendPtr, sendSize, stream));
  }
  return commSuccess;
}
} // namespace ctran::alltoallvp
