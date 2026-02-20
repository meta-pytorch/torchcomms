// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/MultiPeerTransportKernelTest.cuh"

#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes::test {

__global__ void test_device_handle_type_map_kernel(
    MultiPeerDeviceHandle handle,
    int* output_d) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < static_cast<uint32_t>(handle.nRanks)) {
    output_d[tid] = static_cast<int>(handle.get_type(tid));
  }
}

__global__ void test_multi_peer_nvl_send_kernel(
    MultiPeerDeviceHandle handle,
    int peerRank,
    void* src_d,
    size_t nbytes) {
  auto group = make_warp_group();
  auto& nvl = handle.get_nvl(peerRank);
  nvl.send(group, src_d, nbytes);
}

__global__ void test_multi_peer_nvl_recv_kernel(
    MultiPeerDeviceHandle handle,
    int peerRank,
    void* dst_d,
    size_t nbytes) {
  auto group = make_warp_group();
  auto& nvl = handle.get_nvl(peerRank);
  nvl.recv(group, dst_d, nbytes);
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
  // Use P2pSelfTransportDevice::put() through the device handle
  auto group = make_warp_group();
  P2pSelfTransportDevice selfTransport;
  selfTransport.put(
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

void test_multi_peer_nvl_send(
    MultiPeerDeviceHandle handle,
    int peerRank,
    void* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize) {
  test_multi_peer_nvl_send_kernel<<<numBlocks, blockSize>>>(
      handle, peerRank, src_d, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void test_multi_peer_nvl_recv(
    MultiPeerDeviceHandle handle,
    int peerRank,
    void* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize) {
  test_multi_peer_nvl_recv_kernel<<<numBlocks, blockSize>>>(
      handle, peerRank, dst_d, nbytes);
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

__global__ void test_ibgda_accessor_kernel(
    MultiPeerDeviceHandle handle,
    int* output_d) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < static_cast<uint32_t>(handle.nRanks)) {
    if (handle.get_type(tid) == TransportType::P2P_IBGDA) {
      auto& ibgda = handle.get_ibgda(tid);
      output_d[tid] = ibgda.getNumSignals();
    } else {
      output_d[tid] = -1;
    }
  }
}

void test_ibgda_accessor(
    MultiPeerDeviceHandle handle,
    int* output_d,
    int numBlocks,
    int blockSize) {
  test_ibgda_accessor_kernel<<<numBlocks, blockSize>>>(handle, output_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
