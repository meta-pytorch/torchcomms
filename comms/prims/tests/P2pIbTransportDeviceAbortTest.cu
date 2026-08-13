// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/tests/P2pIbTransportDeviceAbortTest.cuh"

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/transport/P2pIbTransportDevice.cuh"
#include "comms/prims/transport/ibrc/P2pIbrcTransportDevice.cuh"

namespace comms::prims::test {

namespace {

__device__ P2pIbrcTransportDevice
makeLocalIbrcTransport(IbrcCmdQueueDevice* queues, IbLocalChannel* channels) {
  return P2pIbrcTransportDevice(
      DeviceSpan<IbrcCmdQueueDevice>{queues, 2},
      /*nics=*/1,
      /*maxChannels=*/1,
      /*qpsPerConnection=*/1,
      DeviceSpan<IbLocalChannel>{channels, 1});
}

__global__ void waitSignalWithDisabledAbortKernel(
    uint64_t* signal,
    bool* success) {
  auto group = make_block_group();
  IbrcCmdQueueDevice queues[2]{};
  IbLocalChannel channels[1]{};
  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(queues, channels);
  P2pIbTransportDevice transport(&ibrc);
  IbgdaLocalBuffer localSignal{signal, NetworkLKeys{}};
  comms::fault_tolerance::AbortDevice abort;
  const bool ok =
      transport.wait_signal(group, localSignal, /*expected=*/0, abort);
  if (group.is_leader()) {
    *success = ok;
  }
}

__global__ void waitIbrcSignalWithDisabledAbortKernel(
    uint64_t* signal,
    bool* success) {
  auto group = make_block_group();
  IbrcCmdQueueDevice queues[2]{};
  IbLocalChannel channels[1]{};
  P2pIbrcTransportDevice transport = makeLocalIbrcTransport(queues, channels);
  IbgdaLocalBuffer localSignal{signal, NetworkLKeys{}};
  comms::fault_tolerance::AbortDevice abort;
  const bool ok =
      transport.wait_signal(group, localSignal, /*expected=*/0, abort);
  if (group.is_leader()) {
    *success = ok;
  }
}

__global__ void waitSignalWithPreAbortedSkipKernel(
    uint64_t* signal,
    bool* success,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  IbrcCmdQueueDevice queues[2]{};
  IbLocalChannel channels[1]{};
  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(queues, channels);
  P2pIbTransportDevice transport(&ibrc);
  IbgdaLocalBuffer localSignal{signal, NetworkLKeys{}};
  const bool ok =
      transport.wait_signal(group, localSignal, /*expected=*/1, abort);
  if (group.is_leader()) {
    *success = !ok;
  }
}

__global__ void waitIbrcSignalWithPreAbortedSkipKernel(
    uint64_t* signal,
    bool* success,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  IbrcCmdQueueDevice queues[2]{};
  IbLocalChannel channels[1]{};
  P2pIbrcTransportDevice transport = makeLocalIbrcTransport(queues, channels);
  IbgdaLocalBuffer localSignal{signal, NetworkLKeys{}};
  const bool ok =
      transport.wait_signal(group, localSignal, /*expected=*/1, abort);
  if (group.is_leader()) {
    *success = !ok;
  }
}

} // namespace

void launchIbWrapperWaitSignalAbortCompileCheck(
    uint64_t* signal,
    bool* success) {
  waitSignalWithDisabledAbortKernel<<<1, 32>>>(signal, success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcWaitSignalWithDisabledAbort(uint64_t* signal, bool* success) {
  waitIbrcSignalWithDisabledAbortKernel<<<1, 32>>>(signal, success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbWrapperWaitSignalWithPreAbortedSkip(
    uint64_t* signal,
    bool* success,
    comms::fault_tolerance::AbortDevice abort) {
  waitSignalWithPreAbortedSkipKernel<<<1, 32>>>(signal, success, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcWaitSignalWithPreAbortedSkip(
    uint64_t* signal,
    bool* success,
    comms::fault_tolerance::AbortDevice abort) {
  waitIbrcSignalWithPreAbortedSkipKernel<<<1, 32>>>(signal, success, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
