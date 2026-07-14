// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/MultiPeerTransportKernelTest.cuh"

#include "comms/prims/transport/Transport.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"
#include "comms/prims/transport/self/P2pSelfTransportDevice.cuh"

namespace comms::prims::test {

__global__ void test_device_handle_type_map_kernel(
    MultiPeerDeviceHandle handle,
    int* output_d) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < static_cast<uint32_t>(handle.nRanks)) {
    output_d[tid] = static_cast<int>(handle.get_type(tid));
  }
}

__global__ void test_multi_peer_self_put_kernel(
    MultiPeerDeviceHandle handle,
    void* dst_d,
    const void* src_d,
    size_t nbytes) {
  // Verify via device handle that myRank's transport is SELF
  if (handle.get_type(handle.myRank) != TransportType::SELF) {
    return;
  }
  // Use P2pSelfTransportDevice::put_group() through the device handle
  auto group = make_warp_group();
  P2pSelfTransportDevice selfTransport;
  selfTransport.put_group(
      group,
      reinterpret_cast<char*>(dst_d),
      reinterpret_cast<const char*>(src_d),
      nbytes);
}

void test_device_handle_type_map(
    MultiPeerDeviceHandle handle,
    int* output_d,
    int numBlocks,
    int blockSize) {
  test_device_handle_type_map_kernel<<<numBlocks, blockSize>>>(
      handle, output_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void test_multi_peer_self_put(
    MultiPeerDeviceHandle handle,
    void* dst_d,
    const void* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize) {
  test_multi_peer_self_put_kernel<<<numBlocks, blockSize>>>(
      handle, dst_d, src_d, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
