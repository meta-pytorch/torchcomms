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

} // namespace ctran::transport::ib
