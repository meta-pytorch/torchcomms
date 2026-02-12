// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <iostream>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/ExtUtils.h"

namespace ctran::allgatherp {
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
    const size_t sendSize,
    const size_t recvOffset,
    PersistArgs& pArgs,
    cudaStream_t stream,
    const std::vector<cudaStream_t>& ceStreams,
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

  // Copy data to other local ranks, each rank starts with the next rank as peer
  // and shift by 1 to avoid all-to-one incast traffic
  for (auto r = 1; r < nLocalRanks; r++) {
    // Use main stream for r=1, ceStreams[r-2] for r>=2
    cudaStream_t stream_ = (r == 1) ? stream : ceStreams[r - 2];

    const auto localPeer = (localRank + r) % nLocalRanks;
    const auto peer = statex->localRankToRank(localPeer);

    // FIXME: the location doesn't seem correct
    if (pArgs.remoteAccessKeys[peer].backend == CtranMapperBackend::NVL) {
      auto recvPtr = getPtr(pArgs.remoteRecvBuffs[peer], recvOffset);
      CLOGF_TRACE(
          COLL,
          "Rank {} CE copy to peer {}, sendBuff {} -> recvBuff {} ({} + recvOffset {}), sendSize {}",
          rank,
          peer,
          sendBuff,
          recvPtr,
          pArgs.remoteRecvBuffs[peer],
          recvOffset,
          sendSize);
      FB_COMMCHECK(mapper->icopy(recvPtr, sendBuff, sendSize, stream_));
    }
  }

  /*
  1) cudaDeviceSynchronize():
  CPU main thread:    [enqueue icopy x7] [BLOCKED...........]
  CPU GPE thread:     [IB PUT operations ALSO BLOCKED - same process]
  GPU ceStreams:      [icopy]──►  ←── still parallel on GPU
                      [icopy]──►
                      [icopy]──►
                                ↑ CPU waits here

  2) CUDA Events:
  CPU main thread:    [enqueue icopy x7] [return immediately]
  CPU GPE thread:     [IB PUT operations running concurrently...]
  GPU ceStreams:      [icopy]──►  ←── parallel
                      [icopy]──►
                      [icopy]──►
  GPU main stream:    ──────────[waits]──────────►
  */

  // FB_CUDACHECK(cudaDeviceSynchronize());

  // Use CUDA events to synchronize streams without blocking the host
  // Record an event on each CE stream, then make the main stream wait
  for (const auto& s : ceStreams) {
    cudaEvent_t event;
    FB_CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    FB_CUDACHECK(cudaEventRecord(event, s));
    FB_CUDACHECK(cudaStreamWaitEvent(stream, event, 0));
    FB_CUDACHECK(cudaEventDestroy(event));
  }

  // Second barrier to ensure all local ranks have finished their CE copies
  // before any rank proceeds to read the received data
  if (barrier) {
    nvlBarrier(comm, stream);
  }

  return commSuccess;
}

// Copy data to self for out-of-place AllGather. No-op if it is an in-place
// allgather.
inline commResult_t copyToSelf(
    CtranComm* comm,
    const void* sendBuff,
    const size_t sendSize,
    PersistArgs& pArgs,
    cudaStream_t stream) {
  const auto statex = comm->statex_.get();
  const auto rank = statex->rank();
  const auto recvOffset = rank * sendSize;

  // Copy data to self for out-of-place allgather
  auto recvPtr = getPtr(pArgs.recvbuff, recvOffset);
  if (recvPtr != sendBuff) {
    CLOGF_TRACE(
        COLL,
        "Rank {} CE copy to self, sendbuff {} -> recvbuff {} ({} + recvOffset {}), sendSize {}",
        rank,
        sendBuff,
        recvPtr,
        pArgs.recvbuff,
        recvOffset,
        sendSize);
    FB_COMMCHECK(
        comm->ctran_->mapper->icopy(recvPtr, sendBuff, sendSize, stream));
  }
  return commSuccess;
}
} // namespace ctran::allgatherp
