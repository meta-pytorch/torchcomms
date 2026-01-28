// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pNvlSendRecvMultipleTest.cuh"

namespace comms::pipes::test {

// testSendMultipleKernel: Tests send_multiple (multiple chunks with varying
// sizes)
__global__ void testSendMultipleKernel(
    P2pNvlTransportDevice* p2p,
    const void* src_d,
    DeviceSpan<const size_t> chunk_sizes,
    DeviceSpan<const size_t> chunk_indices) {
  auto group = make_warp_group();
  p2p->send_multiple(group, src_d, chunk_sizes, chunk_indices);
}

// testRecvMultipleKernel: Tests recv_multiple (multiple chunks with varying
// sizes)
__global__ void testRecvMultipleKernel(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    DeviceSpan<size_t> chunk_sizes) {
  auto group = make_warp_group();
  p2p->recv_multiple(group, dst_d, chunk_sizes);
}

void testSendMultiple(
    P2pNvlTransportDevice* p2p,
    const void* src_d,
    const size_t* chunk_sizes_d,
    size_t chunk_sizes_count,
    const size_t* chunk_indices_d,
    size_t chunk_indices_count,
    int numBlocks,
    int blockSize) {
  DeviceSpan<const size_t> chunk_sizes(chunk_sizes_d, chunk_sizes_count);
  DeviceSpan<const size_t> chunk_indices(chunk_indices_d, chunk_indices_count);
  testSendMultipleKernel<<<numBlocks, blockSize>>>(
      p2p, src_d, chunk_sizes, chunk_indices);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRecvMultiple(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t* chunk_sizes_d,
    size_t chunk_sizes_count,
    int numBlocks,
    int blockSize) {
  DeviceSpan<size_t> chunk_sizes(chunk_sizes_d, chunk_sizes_count);
  testRecvMultipleKernel<<<numBlocks, blockSize>>>(p2p, dst_d, chunk_sizes);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
