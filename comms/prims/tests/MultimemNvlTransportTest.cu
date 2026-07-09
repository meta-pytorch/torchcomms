// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/MultimemNvlTransportTest.cuh"

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"

namespace comms::prims::test {

namespace {

__global__ void setUserSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal(group, signalId, SignalOp::SIGNAL_SET, value);
}

__global__ void setInternalSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal_internal(group, signalId, SignalOp::SIGNAL_SET, value);
}

__global__ void addUserSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal(group, signalId, SignalOp::SIGNAL_ADD, value);
}

__global__ void addInternalSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal_internal(group, signalId, SignalOp::SIGNAL_ADD, value);
}

__global__ void waitAndReadUserSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out) {
  auto group = make_warp_group();
  transport.wait_signal_until(group, signalId, op, expected);
  if (group.is_leader()) {
    *out = transport.read_signal(signalId);
  }
}

__global__ void waitAndReadInternalSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out) {
  auto group = make_warp_group();
  transport.wait_internal_signal_until(group, signalId, op, expected);
  if (group.is_leader()) {
    *out = transport.read_internal_signal(signalId);
  }
}

__global__ void readUserAndInternalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t userId,
    uint64_t internalId,
    uint64_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = transport.read_signal(userId);
    out[1] = transport.read_internal_signal(internalId);
  }
}

} // namespace

void launchSetUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  setUserSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchSetInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  setInternalSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchAddUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  addUserSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchAddInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  addInternalSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchWaitAndReadUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out,
    cudaStream_t stream) {
  waitAndReadUserSignalKernel<<<1, 32, 0, stream>>>(
      transport, signalId, op, expected, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchWaitAndReadInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out,
    cudaStream_t stream) {
  waitAndReadInternalSignalKernel<<<1, 32, 0, stream>>>(
      transport, signalId, op, expected, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchReadUserAndInternal(
    MultimemNvlTransportDevice transport,
    uint64_t userId,
    uint64_t internalId,
    uint64_t* out,
    cudaStream_t stream) {
  readUserAndInternalKernel<<<1, 32, 0, stream>>>(
      transport, userId, internalId, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
