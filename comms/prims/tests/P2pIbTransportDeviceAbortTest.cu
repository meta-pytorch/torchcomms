// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/tests/P2pIbTransportDeviceAbortTest.cuh"

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/transport/P2pIbTransportDevice.cuh"
#include "comms/prims/transport/ibrc/P2pIbrcTransportDevice.cuh"

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

__global__ void waitIbrcSignalWithDisabledAbortKernel(
    uint64_t* signal,
    bool* success,
    bool runWait) {
  auto group = make_block_group();
  IbgdaLocalBuffer localSignal{signal, NetworkLKeys{}};
  P2pIbrcTransportDevice transport(
      DeviceSpan<IbrcCmdQueueDevice>{},
      /*nics=*/0,
      /*maxChannels=*/0,
      /*qpsPerConnection=*/0,
      DeviceSpan<IbLocalChannel>{},
      IbgdaRemoteBuffer{},
      localSignal,
      IbgdaLocalBuffer{},
      IbgdaLocalBuffer{},
      /*numSignalSlots=*/1);
  comms::fault_tolerance::AbortDevice abort;
  if (runWait) {
    transport.wait_signal(group, /*signalId=*/0, /*expected=*/0, abort);
  }
  if (group.is_leader()) {
    *success = true;
  }
}

} // namespace

void launchIbWrapperWaitSignalAbortCompileCheck() {
  P2pIbTransportDevice transport{};
  waitSignalWithDisabledAbortKernel<<<1, 32>>>(transport, false);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcWaitSignalWithDisabledAbort(uint64_t* signal, bool* success) {
  waitIbrcSignalWithDisabledAbortKernel<<<1, 32>>>(
      signal, success, /*runWait=*/false);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
