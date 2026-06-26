// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cuda_runtime.h>
#include "comms/ctran/transport/ib/HostTransportDev.cuh"

// Standalone test kernel that drives the IB host CB transport's
// host↔device staging-copy contract from the GPU side. Used by the
// dist UT (`HostTransportDistUT.cc`) to provide the D2D copy that the
// CB transport's state machine cooperates with. Not part of the
// production transport / algorithm code path.

namespace ctran::transport::ib {

// Args for the unified staging-copy test kernel.
//
// Handles three cases via one launch:
//   - send-only :  recvBuf == nullptr, recvBlocks == 0
//   - recv-only :  sendBuf == nullptr, sendBlocks == 0
//   - bidir     :  both non-null. First sendBlocks GPU blocks drive
//                  the send-side D2D, remaining (gridDim.x - sendBlocks)
//                  drive the recv-side D2D. Caller launches
//                  gridDim.x = sendBlocks + recvBlocks.
//
// Bidirectional CB with two separate kernels on the same stream
// deadlocks because the recv kernel cannot start until the send kernel
// finishes, but the send kernel needs credits from the peer's recv
// state machine — which can only be signalled after the peer's recv
// kernel runs (and that one is also queued behind its send kernel).
// One combined kernel avoids the deadlock while keeping a single
// CUDA stream.
struct IbStagingCopyTestKernelArgs {
  HostTransportDev* devTransport;
  char* sendBuf{nullptr};
  size_t sendTotalSize{0};
  int sendBlocks{0};
  char* recvBuf{nullptr};
  size_t recvTotalSize{0};
  int recvBlocks{0};
};

// Launches the kernel with gridDim.x = sendBlocks + recvBlocks.
// No-op if both are zero.
void launchIbStagingCopyTestKernel(
    IbStagingCopyTestKernelArgs args,
    cudaStream_t stream);

// Args for the per-slot parallel recv-only test kernel. Used by the
// pollRecvNotifications wrap-around regression test. Each GPU block
// handles a SINGLE slot's full sequence of chunks across rounds —
// e.g. block s walks chunks s, s+pipelineDepth, s+2*pipelineDepth,
// ... up to totalChunks. This decouples slot processing so a stranded
// slot N can NOT block the kernel from reaching slot M's wait (m≠n)
// — which is the precise pattern needed to make a buggy
// pollRecvNotifications attribute round-(R+1) postFlag to a slot
// before sender's iput-to-that-slot-round-(R+1) has actually
// landed, causing the kernel to copy stale staging into the user
// buffer.
struct IbRecvPerSlotKernelArgs {
  HostTransportDev* devTransport;
  char* recvBuf;
  size_t recvTotalSize;
  int totalChunks;
};

// Launches the per-slot parallel recv kernel. gridDim.x = pipelineDepth
// (from devTransport). Caller must have setKernelNumBlocks(any,
// pipelineDepth) so each slot's GpeKernelSync has nworkers ==
// pipelineDepth and processData/post signals the right block.
void launchIbRecvPerSlotKernel(
    IbRecvPerSlotKernelArgs args,
    cudaStream_t stream);

} // namespace ctran::transport::ib
