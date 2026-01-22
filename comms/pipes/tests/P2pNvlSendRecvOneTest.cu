// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pNvlSendRecvOneTest.cuh"

namespace comms::pipes::test {

// testSendOneKernel: Tests send_one (single chunk with metadata)
__global__ void testSendOneKernel(
    P2pNvlTransportDevice p2p,
    const void* src_d,
    size_t nbytes,
    size_t offset_in_output,
    bool has_more) {
  auto group = make_warp_group();
  p2p.send_one(group, src_d, nbytes, 0, offset_in_output, has_more);
}

// testRecvOneKernel: Tests recv_one (single chunk with metadata)
__global__ void testRecvOneKernel(
    P2pNvlTransportDevice p2p,
    void* dst_base_d,
    size_t* nbytes_d,
    size_t* offset_d,
    bool* has_more_d) {
  auto group = make_warp_group();

  // Local variables to receive metadata
  size_t nbytes = 0;
  size_t offset = 0;
  bool has_more = false;

  p2p.recv_one(group, dst_base_d, &nbytes, 0, &offset, &has_more);

  // Write results to device memory for verification
  // Only one thread needs to write
  if (group.is_leader()) {
    *nbytes_d = nbytes;
    *offset_d = offset;
    *has_more_d = has_more;
  }
}

void testSendOne(
    P2pNvlTransportDevice p2p,
    const void* src_d,
    size_t nbytes,
    size_t offset_in_output,
    bool has_more,
    int numBlocks,
    int blockSize) {
  testSendOneKernel<<<numBlocks, blockSize>>>(
      p2p, src_d, nbytes, offset_in_output, has_more);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRecvOne(
    P2pNvlTransportDevice p2p,
    void* dst_base_d,
    size_t* nbytes_d,
    size_t* offset_d,
    bool* has_more_d,
    int numBlocks,
    int blockSize) {
  testRecvOneKernel<<<numBlocks, blockSize>>>(
      p2p, dst_base_d, nbytes_d, offset_d, has_more_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// testSendOneMultipleTimesKernel: Tests send_one called multiple times in one
// kernel
__global__ void testSendOneMultipleTimesKernel(
    P2pNvlTransportDevice p2p,
    const void* const* src_d_array,
    const size_t* nbytes_array,
    const size_t* offset_array,
    const bool* has_more_array,
    size_t num_calls) {
  auto group = make_warp_group();

  for (size_t i = 0; i < num_calls; i++) {
    p2p.send_one(
        group,
        src_d_array[i],
        nbytes_array[i],
        static_cast<uint32_t>(i), // call_index increments for each call
        offset_array[i],
        has_more_array[i]);
  }
}

// testRecvOneMultipleTimesKernel: Tests recv_one called multiple times in one
// kernel
__global__ void testRecvOneMultipleTimesKernel(
    P2pNvlTransportDevice p2p,
    void* dst_base_d,
    size_t* nbytes_array_d,
    size_t* offset_array_d,
    bool* has_more_array_d,
    size_t num_calls) {
  auto group = make_warp_group();

  for (size_t i = 0; i < num_calls; i++) {
    size_t nbytes = 0;
    size_t offset = 0;
    bool has_more = false;

    p2p.recv_one(
        group,
        dst_base_d,
        &nbytes,
        static_cast<uint32_t>(i), // call_index increments for each call
        &offset,
        &has_more);

    // Write results to device memory for verification
    if (group.is_leader()) {
      nbytes_array_d[i] = nbytes;
      offset_array_d[i] = offset;
      has_more_array_d[i] = has_more;
    }
  }
}

void testSendOneMultipleTimes(
    P2pNvlTransportDevice p2p,
    const void* const* src_d_array,
    const size_t* nbytes_array,
    const size_t* offset_array,
    const bool* has_more_array,
    size_t num_calls,
    int numBlocks,
    int blockSize) {
  testSendOneMultipleTimesKernel<<<numBlocks, blockSize>>>(
      p2p, src_d_array, nbytes_array, offset_array, has_more_array, num_calls);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRecvOneMultipleTimes(
    P2pNvlTransportDevice p2p,
    void* dst_base_d,
    size_t* nbytes_array_d,
    size_t* offset_array_d,
    bool* has_more_array_d,
    size_t num_calls,
    int numBlocks,
    int blockSize) {
  testRecvOneMultipleTimesKernel<<<numBlocks, blockSize>>>(
      p2p,
      dst_base_d,
      nbytes_array_d,
      offset_array_d,
      has_more_array_d,
      num_calls);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
