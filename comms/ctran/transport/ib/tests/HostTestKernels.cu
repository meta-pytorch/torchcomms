// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/transport/ib/tests/HostTestKernels.h"

#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/transport/ib/HostTransportDev.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

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

// Per-slot recv-only kernel. Each block owns a single slot and walks
// all chunks bound to that slot across rounds. Slots progress
// independently, so a stranded slot only blocks its own block.
//
// IMPORTANT: each slot's GpeKernelSync must have nworkers == 1 (set via
// setKernelNumBlocks(send=*, recv=1)). The host's processData calls
// sync->post(round) which writes postFlag[0..nworkers-1]; only block
// `s` reads postFlag[0] for slot s, but only one block touches each
// slot. The host's isProcessDone scans completeFlag[0..nworkers-1];
// only block s writes completeFlag[0] for slot s, so nworkers=1 keeps
// the check coherent. Launch dim is hard-coded to
// kDeviceMaxPipelineDepth blocks; the kernel only walks chunks bound
// to slot < dt->pipelineDepth, so any extra blocks exit immediately.
__global__ void ibRecvPerSlotKernel(IbRecvPerSlotKernelArgs args) {
  HostTransportDev* dt = args.devTransport;
  const auto slot = blockIdx.x;
  const int pipelineDepth = dt->pipelineDepth;
  if (slot >= pipelineDepth) {
    return;
  }
  const size_t chunkSize = dt->chunkSize;
  const int totalChunks = args.totalChunks;

  auto block = comms::prims::make_block_group();
  DeviceChunkDesc& desc = dt->recvChunks[slot];

  for (int c = slot; c < totalChunks; c += pipelineDepth) {
    const int round = c / pipelineDepth;
    // workerId=0 because each slot's sync has nworkers=1.
    ctran::algos::GpeKernelSyncDev::waitPostWithReset(desc.sync, 0, round);

    const size_t offset = static_cast<size_t>(c) * chunkSize;
    const size_t len = (offset + chunkSize <= args.recvTotalSize)
        ? chunkSize
        : (args.recvTotalSize - offset);
    if (len > 0) {
      comms::prims::memcpy_vectorized(
          args.recvBuf + offset, desc.stagingSlot, len, block);
    }
    ctran::algos::GpeKernelSyncDev::complete(desc.sync, 0, round);
  }
}

void launchIbRecvPerSlotKernel(
    IbRecvPerSlotKernelArgs args,
    cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  ibRecvPerSlotKernel<<<kDeviceMaxPipelineDepth, kBlockSize, 0, stream>>>(args);
}

} // namespace ctran::transport::ib
