// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/tests/P2pIbTransportDeviceAbortTest.cuh"

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/transport/P2pIbTransportDevice.cuh"

namespace comms::prims::test {

namespace {

__global__ void waitSignalWithDisabledAbortKernel(
    P2pIbTransportDevice transport,
    bool runWait) {
  auto group = make_block_group();
  comms::fault_tolerance::AbortDevice abort;
  if (runWait) {
    transport.wait_signal(group, 0, 0, abort);
  }
}

} // namespace

void launchIbWrapperWaitSignalAbortCompileCheck() {
  P2pIbTransportDevice transport{};
  waitSignalWithDisabledAbortKernel<<<1, 32>>>(transport, false);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
