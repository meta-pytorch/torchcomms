// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel implementations for PipesTransportApiTest.
// Tests P2pNvlTransportDevice APIs under stress.

#include "PipesTransportApiTestKernels.cuh"
#include "StressTestKernelUtils.cuh"

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/Transport.cuh"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

// Transport layer uses SIGNAL_ADD / CMP_GE (not window layer's ADD / GE)
using comms::prims::CmpOp;
using comms::prims::SignalOp;

// Kernel launch error check for test code.
#define TRANSPORT_KERNEL_LAUNCH_CHECK()              \
  do {                                               \
    cudaError_t err__ = cudaGetLastError();          \
    assert(err__ == cudaSuccess && "kernel launch"); \
    (void)err__;                                     \
  } while (0)

namespace torchcomms::device::test {

// ---------------------------------------------------------------------------
// Helper: create thread group based on launch configuration
// ---------------------------------------------------------------------------
__device__ inline comms::prims::ThreadGroup make_group_for_launch() {
  if (blockDim.x >= 256) {
    return comms::prims::make_block_group();
  }
  return comms::prims::make_warp_group();
}

// ---------------------------------------------------------------------------
// Stress Signal Kernel
// ---------------------------------------------------------------------------
// Both ranks signal each other and wait in a ring pattern.
// Uses monotonic ADD signals with GE waits.
__global__ void transportStressSignalKernel(
    comms::prims::MultiPeerDeviceHandle handle,
    int peer,
    int iterations) {
  auto group = make_group_for_launch();
  auto& nvl = handle.get_nvl(peer);

  for (int iter = 0; iter < iterations; iter++) {
    // Signal peer: add 1 to signal_id 0
    nvl.signal(group, 0, SignalOp::SIGNAL_ADD, 1);
    // Wait for peer's signal: expect monotonically increasing value
    nvl.wait_signal_until(
        group, 0, CmpOp::CMP_GE, static_cast<uint64_t>(iter + 1));
  }
}

void launchTransportStressSignalKernel(
    comms::prims::MultiPeerDeviceHandle handle,
    int peer,
    int iterations,
    int num_threads,
    cudaStream_t stream) {
  transportStressSignalKernel<<<1, num_threads, 0, stream>>>(
      handle, peer, iterations);
  TRANSPORT_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// LL128 Send/Recv Kernel
// ---------------------------------------------------------------------------
// Warp-only LL128 protocol test. Rank 0 sends, rank 1 receives.
// Fills with byte pattern, verifies on receiver.
__global__ void transportStressLl128Kernel(
    comms::prims::MultiPeerDeviceHandle handle,
    char* buf,
    size_t nbytes,
    int peer,
    int iterations,
    int* results) {
  auto group = comms::prims::make_warp_group();
  auto& nvl = handle.get_nvl(peer);
  int rank = handle.myRank;

  // Check if LL128 is configured
  if (nvl.get_ll128_buffer_num_packets() == 0) {
    // LL128 not available — mark all iterations as passed (skip)
    if (threadIdx.x == 0) {
      for (int i = 0; i < iterations; i++) {
        results[i] = 1;
      }
    }
    return;
  }

  for (int iter = 0; iter < iterations; iter++) {
    char pattern = static_cast<char>((iter + 1) & 0xFF);

    if (rank % 2 == 0) {
      // Fill with byte pattern
      for (size_t i = threadIdx.x; i < nbytes; i += blockDim.x) {
        buf[i] = pattern;
      }
      __syncthreads();
      nvl.ll128_send_group(group, buf, nbytes);
      if (threadIdx.x == 0) {
        results[iter] = 1;
      }
    } else {
      // Clear buffer
      for (size_t i = threadIdx.x; i < nbytes; i += blockDim.x) {
        buf[i] = 0;
      }
      __syncthreads();
      nvl.ll128_recv_group(group, buf, nbytes);
      // Verify
      __shared__ int any_mismatch;
      if (threadIdx.x == 0) {
        any_mismatch = 0;
      }
      __syncthreads();
      for (size_t i = threadIdx.x; i < nbytes; i += blockDim.x) {
        if (buf[i] != pattern) {
          atomicExch(&any_mismatch, 1);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        results[iter] = (any_mismatch == 0) ? 1 : 0;
      }
    }

    // Signal-based barrier before next iteration (barrier buffers not
    // available via get_device_transport())
    nvl.signal(group, 0, SignalOp::SIGNAL_ADD, 1);
    nvl.wait_signal_until(
        group, 0, CmpOp::CMP_GE, static_cast<uint64_t>(iter + 1));
  }
}

void launchTransportStressLl128Kernel(
    comms::prims::MultiPeerDeviceHandle handle,
    char* buf,
    size_t nbytes,
    int peer,
    int iterations,
    int* results,
    cudaStream_t stream) {
  transportStressLl128Kernel<<<1, 32, 0, stream>>>(
      handle, buf, nbytes, peer, iterations, results);
  TRANSPORT_KERNEL_LAUNCH_CHECK();
}

} // namespace torchcomms::device::test
