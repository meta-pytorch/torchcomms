// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pNvlTransportDeviceTest.cuh"

namespace comms::pipes::test {

// Helper to create the appropriate thread group based on type
__device__ inline ThreadGroup make_group(GroupType groupType) {
  switch (groupType) {
    case GroupType::WARP:
      return make_warp_group();
    case GroupType::BLOCK:
      return make_block_group();
    default:
      return make_warp_group();
  }
}

// =============================================================================
// P2pNvlTransportDevice signal API test kernels
// =============================================================================

__global__ void testDeviceSignalKernel(
    P2pNvlTransportDevice p2p,
    uint64_t signalId,
    SignalOp op,
    uint64_t value,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p.signal_threadgroup(group, signalId, op, value);
}

__global__ void testDeviceWaitSignalKernel(
    P2pNvlTransportDevice p2p,
    uint64_t signalId,
    CmpOp op,
    uint64_t value,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p.wait_signal_until_threadgroup(group, signalId, op, value);
}

__global__ void testDeviceSignalThenWaitKernel(
    P2pNvlTransportDevice p2p,
    uint64_t signalId,
    SignalOp signalOp,
    uint64_t signalValue,
    CmpOp waitOp,
    uint64_t waitValue,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p.signal_threadgroup(group, signalId, signalOp, signalValue);
  p2p.wait_signal_until_threadgroup(group, signalId, waitOp, waitValue);
}

void testDeviceSignal(
    P2pNvlTransportDevice p2p,
    uint64_t signalId,
    SignalOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testDeviceSignalKernel<<<numBlocks, blockSize>>>(
      p2p, signalId, op, value, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testDeviceWaitSignal(
    P2pNvlTransportDevice p2p,
    uint64_t signalId,
    CmpOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testDeviceWaitSignalKernel<<<numBlocks, blockSize>>>(
      p2p, signalId, op, value, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testDeviceSignalThenWait(
    P2pNvlTransportDevice p2p,
    uint64_t signalId,
    SignalOp signalOp,
    uint64_t signalValue,
    CmpOp waitOp,
    uint64_t waitValue,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testDeviceSignalThenWaitKernel<<<numBlocks, blockSize>>>(
      p2p, signalId, signalOp, signalValue, waitOp, waitValue, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Direct Signal struct test kernels
// =============================================================================

__global__ void testRawSignalKernel(
    SignalState* signal_d,
    SignalOp op,
    uint64_t value,
    GroupType groupType) {
  auto group = make_group(groupType);
  signal_d->signal(group, op, value);
}

__global__ void testRawWaitSignalKernel(
    SignalState* signal_d,
    CmpOp op,
    uint64_t value,
    GroupType groupType) {
  auto group = make_group(groupType);
  signal_d->wait_until(group, op, value);
}

__global__ void testReadSignalKernel(
    SignalState* signal_d,
    uint64_t* result_d) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *result_d = signal_d->load();
  }
}

void testRawSignal(
    SignalState* signal_d,
    SignalOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testRawSignalKernel<<<numBlocks, blockSize>>>(signal_d, op, value, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRawWaitSignal(
    SignalState* signal_d,
    CmpOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testRawWaitSignalKernel<<<numBlocks, blockSize>>>(
      signal_d, op, value, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testReadSignal(SignalState* signal_d, uint64_t* result_d) {
  testReadSignalKernel<<<1, 1>>>(signal_d, result_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// LL128 transport send/recv test kernels
// =============================================================================

__global__ void testLl128SendKernel(
    P2pNvlTransportDevice p2p,
    const char* src,
    size_t nbytes,
    int64_t flag_value) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  p2p.ll128_send(group, src, nbytes, flag_value, timeout);
}

__global__ void testLl128RecvKernel(
    P2pNvlTransportDevice p2p,
    char* dst,
    size_t nbytes,
    int64_t flag_value) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  p2p.ll128_recv(group, dst, nbytes, flag_value, timeout);
}

void testLl128SendRecv(
    P2pNvlTransportDevice sender,
    P2pNvlTransportDevice receiver,
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize) {
  constexpr int64_t flag_value = 1;

  cudaStream_t send_stream, recv_stream;

  // Sender on GPU 0
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  PIPES_CUDA_CHECK(cudaStreamCreate(&send_stream));
  testLl128SendKernel<<<numBlocks, blockSize, 0, send_stream>>>(
      sender, src_d, nbytes, flag_value);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Receiver on GPU 1
  PIPES_CUDA_CHECK(cudaSetDevice(1));
  PIPES_CUDA_CHECK(cudaStreamCreate(&recv_stream));
  testLl128RecvKernel<<<numBlocks, blockSize, 0, recv_stream>>>(
      receiver, dst_d, nbytes, flag_value);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Wait for both to complete and destroy streams
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(send_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(send_stream));
  PIPES_CUDA_CHECK(cudaSetDevice(1));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(recv_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(recv_stream));
}

} // namespace comms::pipes::test
