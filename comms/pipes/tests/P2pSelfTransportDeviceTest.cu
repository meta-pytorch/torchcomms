// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pSelfTransportDeviceTest.cuh"

namespace comms::pipes::test {

__global__ void
testSelfWriteKernel(char* dst_d, const char* src_d, size_t nbytes) {
  P2pSelfTransportDevice transport;
  auto warp = make_warp_group();
  transport.write(warp, dst_d, src_d, nbytes);
}

void testSelfWrite(
    char* dst_d,
    const char* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize) {
  testSelfWriteKernel<<<numBlocks, blockSize>>>(dst_d, src_d, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
