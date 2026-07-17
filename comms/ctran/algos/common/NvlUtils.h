// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/CudaGraphUtils.h"

extern __global__ void
ncclKernelNvlBarrier(int rank, int nLocalRanks, CtranAlgoDeviceState* devState);

namespace ctran::algos {

// Launch the shared single-thread barrier kernel so all local ranks arrive
// before any rank writes into a peer's recvbuf via CE copies. Shared verbatim
// by AGP (nvlCeBcast) and A2AP (nvlCeAllToAll).
inline commResult_t nvlBarrier(CtranComm* comm, cudaStream_t stream) {
  const auto statex = comm->statex_.get();
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

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

// Issue a batch of intra-node NVL copy-engine copies on `stream`, dispatching
// using cudaMemcpyBatchAsync in eager mode (CUDA >= 12.8), falling back to
// per-op cudaMemcpyAsync under CUDA graph capture (where the batch API is not
// permitted) or on older CUDA.
inline commResult_t nvlCeMemcpyBatch(
    const std::vector<void*>& dsts,
    const std::vector<void*>& srcs,
    const std::vector<size_t>& sizes,
    cudaStream_t stream) {
  const size_t numOps = dsts.size();
  if (numOps == 0) {
    return commSuccess;
  }

  bool batchCopied = false;
#if CUDART_VERSION >= 12080
  ctran::utils::cudagraph::StreamCaptureInfo captureInfo;
  FB_CUDACHECK(
      ctran::utils::cudagraph::getStreamCaptureInfo(stream, captureInfo));
  if (captureInfo.status != cudaStreamCaptureStatusActive) {
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
#if CUDART_VERSION < 13000
    size_t failIdx = 0;
    FB_CUDACHECK(cudaMemcpyBatchAsync(
        dsts.data(),
        srcs.data(),
        sizes.data(),
        numOps,
        attr,
        &failIdx,
        stream));
#else
    FB_CUDACHECK(cudaMemcpyBatchAsync(
        dsts.data(), srcs.data(), sizes.data(), numOps, attr, stream));
#endif
    batchCopied = true;
  }
#endif

  // Fallback to individual copies if with older CUDA or in graph capture mode
  if (!batchCopied) {
    for (size_t i = 0; i < numOps; i++) {
      FB_CUDACHECK(cudaMemcpyAsync(
          dsts.at(i), srcs.at(i), sizes.at(i), cudaMemcpyDefault, stream));
    }
  }

  return commSuccess;
}

// Copy `size` bytes from `src` to `dst` on the CE stream, unless they alias
// (in-place). Used for the self chunk in AGP/A2AP.
inline commResult_t copyToSelf(
    CtranComm* comm,
    const void* src,
    void* dst,
    const size_t size,
    cudaStream_t stream) {
  if (dst != src) {
    CLOGF_TRACE(
        COLL,
        "Rank {} CE copy to self, src {} -> dst {}, size {}",
        comm->statex_->rank(),
        src,
        dst,
        size);
    FB_CUDACHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
  }
  return commSuccess;
}

// Broadcast `sendBuff` (sendSize bytes) to every other local rank's recvbuf at
// `recvOffset` via NVL CE copies. Optionally barriers first to avoid incast
// congestion.
inline commResult_t nvlCeBcast(
    CtranComm* comm,
    const void* sendBuff,
    const size_t sendSize,
    const size_t recvOffset,
    const std::vector<void*>& remoteRecvBuffs,
    const std::vector<CtranMapperRemoteAccessKey>& remoteAccessKeys,
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

  const size_t numOps = nLocalRanks - 1;
  std::vector<void*> dsts;
  std::vector<void*> srcs;
  std::vector<size_t> sizes;

  // Copy data to other local ranks, each rank starts with the next rank as peer
  // and shift by 1 to avoid all-to-one incast traffic
  dsts.reserve(numOps);
  srcs.reserve(numOps);
  sizes.reserve(numOps);
  for (auto r = 1; r < nLocalRanks; r++) {
    const auto localPeer = (localRank + r) % nLocalRanks;
    const auto peer = statex->localRankToRank(localPeer);

    if (remoteAccessKeys[peer].backend != CtranMapperBackend::NVL) {
      FB_ERRORRETURN(
          commInvalidArgument,
          "Peer {} has non-NVL backend in nvlCeBcast",
          peer);
    }

    void* recvPtr = static_cast<char*>(remoteRecvBuffs[peer]) + recvOffset;
    CLOGF_TRACE(
        COLL,
        "Rank {} CE copy to peer {}, sendBuff {} -> recvBuff {} ({} + recvOffset {}), sendSize {}",
        rank,
        peer,
        sendBuff,
        recvPtr,
        remoteRecvBuffs[peer],
        recvOffset,
        sendSize);
    dsts.push_back(recvPtr);
    srcs.push_back(const_cast<void*>(sendBuff));
    sizes.push_back(sendSize);
  }

  FB_COMMCHECK(nvlCeMemcpyBatch(dsts, srcs, sizes, stream));
  return commSuccess;
}

// Copy each intra-node peer's distinct src slice into that peer's recvbuf via
// NVL CE copies on `stream`. rank r's chunk for peer P is sendbuff[P*chunkSize]
// stored at recvbuf[r*chunkSize]. Barriers first to avoid incast congestion.
inline commResult_t nvlCeAllToAll(
    CtranComm* comm,
    const void* sendbuff,
    const size_t chunkSize,
    const std::vector<void*>& remoteRecvBuffs,
    const std::vector<CtranMapperRemoteAccessKey>& remoteAccessKeys,
    cudaStream_t stream,
    bool barrier = true) {
  const auto statex = comm->statex_.get();
  const auto myRank = statex->rank();
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  // Barrier to make sure all local ranks have arrived before CE copies, to
  // avoid unwanted incast traffic congestion.
  if (barrier) {
    FB_COMMCHECK(nvlBarrier(comm, stream));
  }

  const size_t recvOffset = myRank * chunkSize;

  std::vector<void*> dsts;
  std::vector<void*> srcs;
  std::vector<size_t> sizes;
  dsts.reserve(nLocalRanks - 1);
  srcs.reserve(nLocalRanks - 1);
  sizes.reserve(nLocalRanks - 1);

  // Start with the next local rank and shift by 1 to avoid all-to-one incast.
  for (int r = 1; r < nLocalRanks; r++) {
    const auto localPeer = (localRank + r) % nLocalRanks;
    const auto peer = statex->localRankToRank(localPeer);
    if (remoteAccessKeys[peer].backend != CtranMapperBackend::NVL) {
      FB_ERRORRETURN(
          commInvalidArgument,
          "Peer {} has non-NVL backend in nvlCeAllToAll",
          peer);
    }
    auto* dst = static_cast<char*>(remoteRecvBuffs[peer]) + recvOffset;
    auto* src = const_cast<char*>(
        static_cast<const char*>(sendbuff) + peer * chunkSize);
    CLOGF_TRACE(
        COLL,
        "Rank {} CE copy to peer {}, sendBuff {} -> recvBuff {} ({} + recvOffset {}), chunkSize {}",
        myRank,
        peer,
        (void*)src,
        (void*)dst,
        remoteRecvBuffs[peer],
        recvOffset,
        chunkSize);
    dsts.push_back(dst);
    srcs.push_back(src);
    sizes.push_back(chunkSize);
  }

  FB_COMMCHECK(nvlCeMemcpyBatch(dsts, srcs, sizes, stream));
  return commSuccess;
}

} // namespace ctran::algos
