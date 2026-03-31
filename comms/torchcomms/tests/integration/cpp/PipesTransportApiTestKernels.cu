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
// NVL Signal Kernel
// =============================================================================
// Both ranks signal each other (ADD 1 on signal_id 0) and wait (GE 1).

__global__ void nvlSignalKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank) {
  auto group = comms::pipes::make_warp_group();
  auto& nvl = handle.get_nvl(peerRank);
  nvl.signal_threadgroup(group, 0, comms::pipes::SignalOp::SIGNAL_ADD, 1);
  nvl.wait_signal_until_threadgroup(group, 0, comms::pipes::CmpOp::CMP_GE, 1);
}

// =============================================================================
// NVL LL128 Send/Recv Kernels
// =============================================================================

__global__ void nvlLl128SendKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    char* src_d,
    size_t nbytes) {
  auto group = comms::pipes::make_warp_group();
  auto& nvl = handle.get_nvl(peerRank);
  if (nvl.get_ll128_buffer_num_packets() == 0) {
    return; // LL128 not configured — skip
  }
  nvl.ll128_send(group, src_d, nbytes);
}

__global__ void nvlLl128RecvKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    char* dst_d,
    size_t nbytes) {
  auto group = comms::pipes::make_warp_group();
  auto& nvl = handle.get_nvl(peerRank);
  if (nvl.get_ll128_buffer_num_packets() == 0) {
    return; // LL128 not configured — skip
  }
  nvl.ll128_recv(group, dst_d, nbytes);
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

void launchNvlSignalKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    cudaStream_t stream) {
  nvlSignalKernel<<<1, 32, 0, stream>>>(handle, peerRank);
  CUDA_LAUNCH_CHECK();
}

void launchNvlLl128SendKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* src_d,
    size_t nbytes,
    cudaStream_t stream) {
  nvlLl128SendKernel<<<1, 32, 0, stream>>>(
      handle, peerRank, static_cast<char*>(src_d), nbytes);
  CUDA_LAUNCH_CHECK();
}

void launchNvlLl128RecvKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* dst_d,
    size_t nbytes,
    cudaStream_t stream) {
  nvlLl128RecvKernel<<<1, 32, 0, stream>>>(
      handle, peerRank, static_cast<char*>(dst_d), nbytes);
  CUDA_LAUNCH_CHECK();
}

// =============================================================================
// LL128 Availability Check
// =============================================================================
// Single-thread kernel that writes get_ll128_buffer_num_packets() to *result_d.

__global__ void ll128AvailableKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    int* result_d) {
  auto& nvl = handle.get_nvl(peerRank);
  *result_d = static_cast<int>(nvl.get_ll128_buffer_num_packets());
}

int checkLl128Available(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    cudaStream_t stream) {
  int* d_result = nullptr;
  auto err = cudaMalloc(&d_result, sizeof(int));
  if (err != cudaSuccess) {
    return 0;
  }
  cudaMemset(d_result, 0, sizeof(int));
  ll128AvailableKernel<<<1, 1, 0, stream>>>(handle, peerRank, d_result);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_result);
    return 0;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(d_result);
    return 0;
  }
  int h_result = 0;
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_result);
  return h_result;
}

} // namespace torchcomms::device::test
