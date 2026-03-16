// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <iostream>
#include <vector>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::allgatherp {
inline void* getPtr(void* base, size_t offset) {
  return (void*)((uintptr_t)base + offset);
}

inline commResult_t scheduleNvlBarrier(CtranComm* comm, cudaStream_t stream) {
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
    bool barrier = true) {
  const auto statex = comm->statex_.get();
  const auto rank = statex->rank();
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  // Barrier to make sure all local ranks has arrived before CE bcast, to
  // avoid unwanted incast traffic congestion
  if (barrier) {
    scheduleNvlBarrier(comm, stream);
  }

  const size_t numOps = nLocalRanks - 1;
  std::vector<void*> dsts(numOps);
  std::vector<void*> srcs(numOps);
  std::vector<size_t> sizes(numOps);

  // Copy data to other local ranks, each rank starts with the next rank as peer
  // and shift by 1 to avoid all-to-one incast traffic
  for (auto r = 1; r < nLocalRanks; r++) {
    const auto localPeer = (localRank + r) % nLocalRanks;
    const auto peer = statex->localRankToRank(localPeer);

    if (pArgs.remoteAccessKeys[peer].backend != CtranMapperBackend::NVL) {
      FB_ERRORRETURN(
          commInvalidArgument,
          "Peer {} has non-NVL backend in nvlCeBcast",
          peer);
    }

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
    dsts.at(r - 1) = recvPtr;
    srcs.at(r - 1) = const_cast<void*>(sendBuff);
    sizes.at(r - 1) = sendSize;
  }

#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes attr = {};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;

#if CUDART_VERSION == 12080
  size_t failIdx = 0;
  FB_CUDACHECK(cudaMemcpyBatchAsync(
      dsts.data(), srcs.data(), sizes.data(), numOps, attr, &failIdx, stream));
#else
  FB_CUDACHECK(cudaMemcpyBatchAsync(
      dsts.data(), srcs.data(), sizes.data(), numOps, attr, stream));
#endif

#else
  auto mapper = comm->ctran_->mapper.get();
  for (size_t i = 0; i < numOps; i++) {
    FB_COMMCHECK(mapper->icopy(dsts.at(i), srcs.at(i), sizes.at(i), stream));
  }
#endif

  return commSuccess;
}

inline int resolveCeBatchSize(int nLocalRanks, size_t sendSize) {
  int batchSize = NCCL_CTRAN_ALLGATHER_P_DIRECT_BATCH_SIZE;
  if (batchSize == 0) {
    if (nLocalRanks <= 8 ||
        sendSize < static_cast<size_t>(
                       NCCL_CTRAN_ALLGATHER_P_DIRECT_BATCH_THRESHOLD)) {
      // H100 HGX or small messages (< 16MB per rank): unbatched.
      // Barrier overhead outweighs incast benefit for small messages.
      batchSize = nLocalRanks - 1;
    } else {
      // Large messages (>= 16MB per rank) on NVL36/NVL64/NVL72:
      // aggressive batching to bound NVSwitch incast.
      // Target ~8 concurrent CE copies per wave, cap at 16.
      const int nPeers = nLocalRanks - 1;
      batchSize = std::min((nPeers + 7) / 8, 16);
    }
  }
  return batchSize;
}

inline commResult_t nvlCeBcastBatched(
    CtranComm* comm,
    const void* sendBuff,
    const size_t sendSize,
    const size_t recvOffset,
    PersistArgs& pArgs,
    cudaStream_t stream,
    int batchSize,
    bool barrier = true) {
  const auto statex = comm->statex_.get();
  const auto rank = statex->rank();
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  if (nLocalRanks <= 1) {
    return commSuccess;
  }

  const int nPeers = nLocalRanks - 1;

  // If batchSize is not set or >= nPeers, fall back to unbatched nvlCeBcast
  if (batchSize <= 0 || batchSize >= nPeers) {
    return nvlCeBcast(
        comm, sendBuff, sendSize, recvOffset, pArgs, stream, barrier);
  }

  // ── PRE-BARRIER ──
  // Ensures all local ranks have arrived before any CE writes begin.
  if (barrier) {
    FB_COMMCHECK(scheduleNvlBarrier(comm, stream));
  }

  const int nWaves = (nPeers + batchSize - 1) / batchSize;

  for (int wave = 0; wave < nWaves; wave++) {
    const int waveStart = wave * batchSize;
    const int waveEnd = std::min(waveStart + batchSize, nPeers);
    const size_t waveOps = waveEnd - waveStart;

    // Build dst/src/size vectors for this wave's CE copies.
    // Anti-diagonal round-robin: offset r = i+1, peer = (localRank+r)%N.
    // Within each wave, every source targets a unique destination set,
    // so max incast at any destination = batchSize.
    std::vector<void*> dsts(waveOps);
    std::vector<void*> srcs(waveOps);
    std::vector<size_t> sizes(waveOps);

    for (int i = waveStart; i < waveEnd; i++) {
      const int r = i + 1;
      const auto localPeer = (localRank + r) % nLocalRanks;
      const auto peer = statex->localRankToRank(localPeer);

      if (pArgs.remoteAccessKeys[peer].backend != CtranMapperBackend::NVL) {
        FB_ERRORRETURN(
            commInvalidArgument,
            "Peer {} has non-NVL backend in nvlCeBcastBatched",
            peer);
      }

      auto recvPtr = getPtr(pArgs.remoteRecvBuffs[peer], recvOffset);
      CLOGF_TRACE(
          COLL,
          "Rank {} CE copy to peer {} (wave {}/{}, batch idx {}), "
          "sendBuff {} -> recvBuff {} ({} + recvOffset {}), sendSize {}",
          rank,
          peer,
          wave,
          nWaves,
          i - waveStart,
          sendBuff,
          recvPtr,
          pArgs.remoteRecvBuffs[peer],
          recvOffset,
          sendSize);
      dsts.at(i - waveStart) = recvPtr;
      srcs.at(i - waveStart) = const_cast<void*>(sendBuff);
      sizes.at(i - waveStart) = sendSize;
    }

    // Issue this wave's CE copies as a batch
#if CUDART_VERSION >= 12080
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;

#if CUDART_VERSION == 12080
    size_t failIdx = 0;
    FB_CUDACHECK(cudaMemcpyBatchAsync(
        dsts.data(),
        srcs.data(),
        sizes.data(),
        waveOps,
        attr,
        &failIdx,
        stream));
#else
    FB_CUDACHECK(cudaMemcpyBatchAsync(
        dsts.data(), srcs.data(), sizes.data(), waveOps, attr, stream));
#endif

#else
    {
      auto mapper = comm->ctran_->mapper.get();
      for (size_t j = 0; j < waveOps; j++) {
        FB_COMMCHECK(
            mapper->icopy(dsts.at(j), srcs.at(j), sizes.at(j), stream));
      }
    }
#endif

    // ── INTER-WAVE BARRIER ──
    // Synchronize all GPUs between waves. This bounds incast:
    //   - All GPUs finish wave W before any GPU starts wave W+1
    //   - The barrier kernel won't execute until this wave's CE copies
    //     complete on this GPU (stream ordering)
    //   - The barrier then waits for all other GPUs' barriers too
    //
    // Skip after the last wave — matches nvlCeBcast which has no
    // post-barrier (caller handles post-synchronization).
    if (wave < nWaves - 1) {
      FB_COMMCHECK(scheduleNvlBarrier(comm, stream));
    }
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
