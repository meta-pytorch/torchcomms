// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pSelfTransportDeviceTest.cuh"

namespace comms::pipes::test {

__global__ void
testSelfPutKernel(char* dst_d, const char* src_d, size_t nbytes) {
  P2pSelfTransportDevice transport;
  auto warp = make_warp_group();
  transport.put(warp, dst_d, src_d, nbytes);
}

void testSelfPut(
    char* dst_d,
    const char* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize) {
  testSelfPutKernel<<<numBlocks, blockSize>>>(dst_d, src_d, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
