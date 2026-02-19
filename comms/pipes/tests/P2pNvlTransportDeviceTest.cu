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
// RecvStream/SendStream test kernels
// =============================================================================

__global__ void testRecvSendStreamSenderKernel(
    P2pNvlTransportDevice transport,
    char* srcBuffer,
    std::size_t nbytes) {
  auto group = make_warp_group();
  auto send = transport.send_stream(nbytes);

  send.for_each_slot(group, [&](auto slot) {
    // Copy from source buffer to staging
    memcpy_vectorized(slot.data, srcBuffer + slot.offset, slot.size, group);
  });
}

__global__ void testRecvSendStreamReceiverKernel(
    P2pNvlTransportDevice transport,
    char* dstBuffer,
    std::size_t nbytes) {
  auto group = make_warp_group();
  auto recv = transport.recv_stream(nbytes);

  recv.for_each_ready_chunk(group, [&](auto chunk) {
    // Copy from staging to destination buffer
    memcpy_vectorized(dstBuffer + chunk.offset, chunk.data, chunk.size, group);
  });
}

void testRecvSendStreamLoopback(
    P2pNvlTransportDevice transport0,
    P2pNvlTransportDevice transport1,
    char* srcBuffer0,
    char* dstBuffer1,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  // Launch sender and receiver kernels concurrently
  cudaStream_t stream0, stream1;
  PIPES_CUDA_CHECK(cudaStreamCreate(&stream0));
  PIPES_CUDA_CHECK(cudaStreamCreate(&stream1));

  // Sender on GPU 0
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  testRecvSendStreamSenderKernel<<<numBlocks, blockSize, 0, stream0>>>(
      transport0, srcBuffer0, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Receiver on GPU 1
  PIPES_CUDA_CHECK(cudaSetDevice(1));
  testRecvSendStreamReceiverKernel<<<numBlocks, blockSize, 0, stream1>>>(
      transport1, dstBuffer1, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Wait for both to complete
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream0));
  PIPES_CUDA_CHECK(cudaSetDevice(1));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream1));

  PIPES_CUDA_CHECK(cudaStreamDestroy(stream0));
  PIPES_CUDA_CHECK(cudaStreamDestroy(stream1));
}

// Intermediate rank kernel: receives from predecessor, forwards to successor
// using the positional API (slot_for + commit_slot).
__global__ void testRecvSendStreamIntermediateKernel(
    P2pNvlTransportDevice transport_recv,
    P2pNvlTransportDevice transport_send,
    std::size_t nbytes) {
  auto group = make_warp_group();
  auto recv = transport_recv.recv_stream(nbytes);
  auto send = transport_send.send_stream(nbytes);

  recv.for_each_ready_chunk(group, [&](auto chunk) {
    auto slot = send.slot_for(group, chunk);
    memcpy_vectorized(slot.data, chunk.data, chunk.size, group);
    send.commit_slot(group, slot);
  });
}

void testRecvSendStreamForwarding(
    P2pNvlTransportDevice transport_send_0to1,
    P2pNvlTransportDevice transport_recv_1from0,
    P2pNvlTransportDevice transport_send_1to0,
    P2pNvlTransportDevice transport_recv_0from1,
    char* srcBuffer0,
    char* dstBuffer0,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  // Launch 3 kernels concurrently:
  // GPU0 sender → GPU1 intermediate → GPU0 receiver
  cudaStream_t streamSender, streamIntermediate, streamReceiver;
  PIPES_CUDA_CHECK(cudaStreamCreate(&streamSender));
  PIPES_CUDA_CHECK(cudaStreamCreate(&streamIntermediate));
  PIPES_CUDA_CHECK(cudaStreamCreate(&streamReceiver));

  // Sender on GPU 0
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  testRecvSendStreamSenderKernel<<<numBlocks, blockSize, 0, streamSender>>>(
      transport_send_0to1, srcBuffer0, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Intermediate on GPU 1
  PIPES_CUDA_CHECK(cudaSetDevice(1));
  testRecvSendStreamIntermediateKernel<<<
      numBlocks,
      blockSize,
      0,
      streamIntermediate>>>(transport_recv_1from0, transport_send_1to0, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Receiver on GPU 0
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  testRecvSendStreamReceiverKernel<<<numBlocks, blockSize, 0, streamReceiver>>>(
      transport_recv_0from1, dstBuffer0, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Wait for all to complete
  PIPES_CUDA_CHECK(cudaStreamSynchronize(streamSender));
  PIPES_CUDA_CHECK(cudaSetDevice(1));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(streamIntermediate));
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(streamReceiver));

  PIPES_CUDA_CHECK(cudaStreamDestroy(streamSender));
  PIPES_CUDA_CHECK(cudaStreamDestroy(streamIntermediate));
  PIPES_CUDA_CHECK(cudaStreamDestroy(streamReceiver));
}

} // namespace comms::pipes::test
