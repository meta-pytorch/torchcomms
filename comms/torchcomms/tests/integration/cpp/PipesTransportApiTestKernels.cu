// Copyright (c) Meta Platforms, Inc. and affiliates.
// CUDA kernels for PipesTransportApiTest - tests NVL send/recv through
// the pipes MultiPeerDeviceHandle obtained via torchcomms.

#include "PipesTransportApiTestKernels.cuh"

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"

#include <stdexcept>
#include <string>

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CUDA_LAUNCH_CHECK()                                                   \
  do {                                                                        \
    cudaError_t err__ = cudaGetLastError();                                   \
    if (err__ != cudaSuccess) {                                               \
      throw std::runtime_error(                                               \
          std::string("Kernel launch failed: ") + cudaGetErrorString(err__)); \
    }                                                                         \
  } while (0)

namespace torchcomms::device::test {

// =============================================================================
// NVL Send Kernel
// =============================================================================
// Sends nbytes from src_d to the given peer via NVLink transport.
// Uses a single warp (32 threads) for the send operation.

__global__ void nvlSendKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* src_d,
    size_t nbytes) {
  auto group = comms::pipes::make_warp_group();
  auto& nvl = handle.get_nvl(peerRank);
  nvl.send(group, src_d, nbytes);
}

// =============================================================================
// NVL Recv Kernel
// =============================================================================
// Receives nbytes into dst_d from the given peer via NVLink transport.
// Uses a single warp (32 threads) for the recv operation.

__global__ void nvlRecvKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* dst_d,
    size_t nbytes) {
  auto group = comms::pipes::make_warp_group();
  auto& nvl = handle.get_nvl(peerRank);
  nvl.recv(group, dst_d, nbytes);
}

// =============================================================================
// Host-callable wrapper functions
// =============================================================================

void launchNvlSendKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* src_d,
    size_t nbytes,
    cudaStream_t stream) {
  nvlSendKernel<<<1, 32, 0, stream>>>(handle, peerRank, src_d, nbytes);
  CUDA_LAUNCH_CHECK();
}

void launchNvlRecvKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* dst_d,
    size_t nbytes,
    cudaStream_t stream) {
  nvlRecvKernel<<<1, 32, 0, stream>>>(handle, peerRank, dst_d, nbytes);
  CUDA_LAUNCH_CHECK();
}

} // namespace torchcomms::device::test
