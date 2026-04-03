// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel implementations for PipesTransportApiTest.
// Tests P2pNvlTransportDevice APIs under stress.

#include "PipesTransportApiTestKernels.cuh"
#include "StressTestKernelUtils.cuh"

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"

// Transport layer uses SIGNAL_ADD / CMP_GE (not window layer's ADD / GE)
using comms::pipes::CmpOp;
using comms::pipes::SignalOp;

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
__device__ inline comms::pipes::ThreadGroup make_group_for_launch() {
  if (blockDim.x >= 256) {
    return comms::pipes::make_block_group();
  }
  return comms::pipes::make_warp_group();
}

// ---------------------------------------------------------------------------
// Stress Send/Recv Kernel
// ---------------------------------------------------------------------------
// Rank 0 fills buffer with pattern, sends to rank 1 via NVLink transport.
// Rank 1 receives, verifies data. Barrier sync between iterations.
__global__ void transportStressSendRecvKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    float* buf,
    size_t count,
    int peer,
    int iterations,
    int* results) {
  auto group = make_group_for_launch();
  auto& nvl = handle.get_nvl(peer);
  int rank = handle.myRank;
  size_t nbytes = count * sizeof(float);

  for (int iter = 0; iter < iterations; iter++) {
    if (rank % 2 == 0) {
      // Fill src with identifiable pattern
      fillPattern(buf, count, rank, iter);
      group.sync();
      nvl.send(group, buf, nbytes);
      // Sender always passes
      if (group.thread_id_in_group == 0) {
        results[iter] = 1;
      }
    } else {
      // Clear dst before recv
      for (size_t i = group.thread_id_in_group; i < count;
           i += group.group_size) {
        buf[i] = 0.0f;
      }
      group.sync();
      nvl.recv(group, buf, nbytes);
      // Verify received data matches sender's pattern
      verifyPattern(buf, count, peer, iter, &results[iter]);
    }
    group.sync();

    // Signal-based barrier: both ranks signal and wait before next iteration.
    // Barrier buffers are not available via get_device_transport(), so we use
    // monotonic signal ADD/GE on signal_id 0 as a barrier replacement.
    nvl.signal_threadgroup(group, 0, SignalOp::SIGNAL_ADD, 1);
    nvl.wait_signal_until_threadgroup(
        group, 0, CmpOp::CMP_GE, static_cast<uint64_t>(iter + 1));
  }
}

void launchTransportStressSendRecvKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    float* buf,
    size_t count,
    int peer,
    int iterations,
    int num_threads,
    int* results,
    cudaStream_t stream) {
  transportStressSendRecvKernel<<<1, num_threads, 0, stream>>>(
      handle, buf, count, peer, iterations, results);
  TRANSPORT_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Stress Signal Kernel
// ---------------------------------------------------------------------------
// Both ranks signal each other and wait in a ring pattern.
// Uses monotonic ADD signals with GE waits.
__global__ void transportStressSignalKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peer,
    int iterations) {
  auto group = make_group_for_launch();
  auto& nvl = handle.get_nvl(peer);

  for (int iter = 0; iter < iterations; iter++) {
    // Signal peer: add 1 to signal_id 0
    nvl.signal_threadgroup(group, 0, SignalOp::SIGNAL_ADD, 1);
    // Wait for peer's signal: expect monotonically increasing value
    nvl.wait_signal_until_threadgroup(
        group, 0, CmpOp::CMP_GE, static_cast<uint64_t>(iter + 1));
  }
}

void launchTransportStressSignalKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peer,
    int iterations,
    int num_threads,
    cudaStream_t stream) {
  transportStressSignalKernel<<<1, num_threads, 0, stream>>>(
      handle, peer, iterations);
  TRANSPORT_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Combined Ops Kernel
// ---------------------------------------------------------------------------
// Exercises barrier + send/recv + signal/wait per iteration.
__global__ void transportStressCombinedKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    float* buf,
    size_t count,
    int peer,
    int iterations,
    int* results) {
  auto group = make_group_for_launch();
  auto& nvl = handle.get_nvl(peer);
  int rank = handle.myRank;
  size_t nbytes = count * sizeof(float);

  for (int iter = 0; iter < iterations; iter++) {
    // Phase 1: Signal-based barrier (barrier buffers not available via
    // get_device_transport(), so use signal ADD/GE on signal_id 0).
    // Two signals per iteration: one here, one after send/recv.
    nvl.signal_threadgroup(group, 0, SignalOp::SIGNAL_ADD, 1);
    nvl.wait_signal_until_threadgroup(
        group, 0, CmpOp::CMP_GE, static_cast<uint64_t>(2 * iter + 1));

    // Phase 2: Send/recv
    if (rank % 2 == 0) {
      fillPattern(buf, count, rank, iter);
      group.sync();
      nvl.send(group, buf, nbytes);
    } else {
      for (size_t i = group.thread_id_in_group; i < count;
           i += group.group_size) {
        buf[i] = 0.0f;
      }
      group.sync();
      nvl.recv(group, buf, nbytes);
    }

    // Phase 3: Signal/wait (confirms both ranks finished send/recv)
    nvl.signal_threadgroup(group, 0, SignalOp::SIGNAL_ADD, 1);
    nvl.wait_signal_until_threadgroup(
        group, 0, CmpOp::CMP_GE, static_cast<uint64_t>(2 * iter + 2));

    // Phase 4: Verify on receiver
    if (rank % 2 == 1) {
      verifyPattern(buf, count, peer, iter, &results[iter]);
    } else {
      if (group.thread_id_in_group == 0) {
        results[iter] = 1;
      }
    }
    group.sync();
  }
}

void launchTransportStressCombinedKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    float* buf,
    size_t count,
    int peer,
    int iterations,
    int num_threads,
    int* results,
    cudaStream_t stream) {
  transportStressCombinedKernel<<<1, num_threads, 0, stream>>>(
      handle, buf, count, peer, iterations, results);
  TRANSPORT_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// LL128 Send/Recv Kernel
// ---------------------------------------------------------------------------
// Warp-only LL128 protocol test. Rank 0 sends, rank 1 receives.
// Fills with byte pattern, verifies on receiver.
__global__ void transportStressLl128Kernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    char* buf,
    size_t nbytes,
    int peer,
    int iterations,
    int* results) {
  auto group = comms::pipes::make_warp_group();
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
      nvl.ll128_send(group, buf, nbytes);
      if (threadIdx.x == 0) {
        results[iter] = 1;
      }
    } else {
      // Clear buffer
      for (size_t i = threadIdx.x; i < nbytes; i += blockDim.x) {
        buf[i] = 0;
      }
      __syncthreads();
      nvl.ll128_recv(group, buf, nbytes);
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
    nvl.signal_threadgroup(group, 0, SignalOp::SIGNAL_ADD, 1);
    nvl.wait_signal_until_threadgroup(
        group, 0, CmpOp::CMP_GE, static_cast<uint64_t>(iter + 1));
  }
}

void launchTransportStressLl128Kernel(
    comms::pipes::MultiPeerDeviceHandle handle,
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
