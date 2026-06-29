// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/torchcomms/transport/tests/cpp/busy_kernel/BusyKernel.h"

namespace meta::comms::stagedrdma_test {
namespace {

// Each thread spins on the SM clock for `ticks` cycles, doing no memory work,
// so the kernel keeps SMs busy without consuming memory bandwidth.
__global__ void busyKernel(long long ticks) {
  const long long start = clock64();
  while (clock64() - start < ticks) {
  }
}

} // namespace

void launchBusyKernel(cudaStream_t stream, long long spinTicks) {
  int dev = 0;
  cudaGetDevice(&dev);
  int sms = 1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  // Oversubscribe every SM (4 blocks/SM) so the whole GPU stays saturated.
  const int blocks = sms * 4;
  busyKernel<<<blocks, 256, 0, stream>>>(spinTicks);
}

} // namespace meta::comms::stagedrdma_test
