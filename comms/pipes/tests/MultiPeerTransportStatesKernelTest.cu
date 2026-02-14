// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/MultiPeerTransportStatesKernelTest.cuh"

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
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto* dst = reinterpret_cast<char*>(dst_d);
  auto* src = reinterpret_cast<const char*>(src_d);
  for (size_t i = tid; i < nbytes; i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
  }
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

} // namespace comms::pipes::test
