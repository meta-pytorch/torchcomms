// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/transport/ib/tests/HostTestKernels.h"

#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/transport/ib/HostTransportDev.cuh"
#include "comms/prims/CopyUtils.cuh"
#include "comms/prims/ThreadGroup.cuh"

namespace ctran::transport::ib {

namespace {

// Device-side D2D copy state machine.
//
// Walks every chunk of `userBuf`: waits for the host's per-chunk
// GpeKernelSync post (via the standalone variant that doesn't need
// the CTRAN host-abort plumbing), does the vectorized memcpy between
// `userBuf` and the staging slot in the direction dictated by
// `isSend`, then signals complete back to the host so the next round
// can fire.
__device__ __forceinline__ void ibCopyStagingChunked(
    HostTransportDev* dt,
    char* userBuf,
    size_t totalSize,
    bool isSend,
    int myBlockIdx,
    int numBlocks) {
  auto block = comms::prims::make_block_group();

  const size_t chunkSize = dt->chunkSize;
  const int pipelineDepth = dt->pipelineDepth;
  const int totalChunks =
      static_cast<int>((totalSize + chunkSize - 1) / chunkSize);

  for (int c = 0; c < totalChunks; ++c) {
    const int slot = c % pipelineDepth;
    const int round = c / pipelineDepth;

    DeviceChunkDesc& desc =
        isSend ? dt->sendChunks[slot] : dt->recvChunks[slot];

    ctran::algos::GpeKernelSyncDev::waitPostWithReset(
        desc.sync, myBlockIdx, round);

    const size_t offset = static_cast<size_t>(c) * chunkSize;
    const size_t len =
        (offset + chunkSize <= totalSize) ? chunkSize : (totalSize - offset);

    const size_t bytesPerBlock = (len + static_cast<size_t>(numBlocks) - 1) /
        static_cast<size_t>(numBlocks);
    const size_t myStart =
        min(static_cast<size_t>(myBlockIdx) * bytesPerBlock, len);
    const size_t myEnd = min(myStart + bytesPerBlock, len);
    const size_t myLen = myEnd - myStart;

    if (myLen > 0) {
      char* staging = desc.stagingSlot;
      char* user = userBuf + offset;
      if (isSend) {
        comms::prims::memcpy_vectorized(
            staging + myStart, user + myStart, myLen, block);
      } else {
        comms::prims::memcpy_vectorized(
            user + myStart, staging + myStart, myLen, block);
      }
    }

    ctran::algos::GpeKernelSyncDev::complete(desc.sync, myBlockIdx, round);
  }
}

} // namespace

__global__ void ibStagingCopyTestKernel(IbStagingCopyTestKernelArgs args) {
  // First args.sendBlocks blocks do send; remaining do recv. Either
  // side may be empty (sendBlocks == 0 or recvBlocks == 0), in which
  // case only the other side's branch fires.
  if (blockIdx.x < args.sendBlocks) {
    if (args.sendBuf != nullptr) {
      ibCopyStagingChunked(
          args.devTransport,
          args.sendBuf,
          args.sendTotalSize,
          /*isSend=*/true,
          blockIdx.x,
          args.sendBlocks);
    }
  } else {
    if (args.recvBuf != nullptr) {
      const auto recvBlocks = gridDim.x - args.sendBlocks;
      ibCopyStagingChunked(
          args.devTransport,
          args.recvBuf,
          args.recvTotalSize,
          /*isSend=*/false,
          blockIdx.x - args.sendBlocks,
          recvBlocks);
    }
  }
}

void launchIbStagingCopyTestKernel(
    IbStagingCopyTestKernelArgs args,
    cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int numBlocks = args.sendBlocks + args.recvBlocks;
  if (numBlocks <= 0) {
    return;
  }
  ibStagingCopyTestKernel<<<numBlocks, kBlockSize, 0, stream>>>(args);
}

} // namespace ctran::transport::ib
