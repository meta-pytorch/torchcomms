// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/TimeoutTrapTest.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include "comms/pipes/ChunkState.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/TimeoutUtils.h"

namespace comms::pipes::test {

// CUDA error checking macro for test setup code
// Throws on failure so tests fail clearly
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      throw std::runtime_error(                                            \
          std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + \
          __FILE__ + ":" + std::to_string(__LINE__));                      \
    }                                                                      \
  } while (0)

// Kernel that waits on ChunkState that will never become ready
// This should trigger a timeout and call __trap()
__global__ void chunkStateTimeoutKernel(ChunkState* state, Timeout timeout) {
  // Start the timeout timer - captures clock64() once at kernel entry
  timeout.start();
  auto group = make_thread_group(SyncScope::WARP);

  // State is initialized to READY_TO_SEND (-1), so waiting for stepId=0
  // will spin forever unless timeout triggers
  state->wait_ready_to_recv(group, 0, 0, timeout);
}

// Kernel that waits on SignalState that will never be signaled
// This should trigger a timeout and call __trap()
// Uses the single-threaded API which creates a ThreadGroup internally
__global__ void signalStateTimeoutKernel(SignalState* state, Timeout timeout) {
  // Start the timeout timer - captures clock64() once at kernel entry
  timeout.start();

  // State is initialized to 0, so waiting for value 1 will spin forever
  // unless timeout triggers
  // Uses simple API - ThreadGroup is created internally for timeout checking
  state->wait_until(CmpOp::CMP_EQ, 1, timeout);
}

// Kernel that starts timeout, checks once, and completes successfully
// This is a positive test case that exercises the full timeout path
// without actually timing out
__global__ void noTimeoutKernel(Timeout timeout) {
  // Start the timeout timer
  timeout.start();

  // Check timeout once - should not be expired since we're well within timeout
  if (timeout.checkExpired()) {
    printf("CUDA TIMEOUT ERROR: Unexpected timeout in noTimeoutKernel\n");
    __trap();
  }
}

void launchChunkStateTimeoutKernel(int device, uint32_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device));

  // Allocate ChunkState on device
  ChunkState* d_state;
  CUDA_CHECK(cudaMalloc(&d_state, sizeof(ChunkState)));

  // Initialize to READY_TO_SEND (default constructor state)
  ChunkState h_state;
  CUDA_CHECK(cudaMemcpy(
      d_state, &h_state, sizeof(ChunkState), cudaMemcpyHostToDevice));

  // Create timeout configuration
  Timeout timeout = makeTimeout(timeout_ms, device);

  // Launch kernel with a full warp - should trap due to timeout
  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  chunkStateTimeoutKernel<<<1, 32>>>(d_state, timeout);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaDeviceSynchronize();

  // Don't free - device will be reset by test
}

void launchSignalStateTimeoutKernel(int device, uint32_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device));

  // Allocate SignalState on device
  SignalState* d_state;
  CUDA_CHECK(cudaMalloc(&d_state, sizeof(SignalState)));

  // Initialize to 0 (default constructor state)
  SignalState h_state;
  CUDA_CHECK(cudaMemcpy(
      d_state, &h_state, sizeof(SignalState), cudaMemcpyHostToDevice));

  // Create timeout configuration
  Timeout timeout = makeTimeout(timeout_ms, device);

  // Launch kernel with a full warp - should trap due to timeout
  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  signalStateTimeoutKernel<<<1, 32>>>(d_state, timeout);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaDeviceSynchronize();

  // Don't free - device will be reset by test
}

void launchNoTimeoutKernel(int device, uint32_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device));

  // Create timeout configuration
  Timeout timeout = makeTimeout(timeout_ms, device);

  // Launch kernel - should complete normally
  noTimeoutKernel<<<1, 1>>>(timeout);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Kernel that uses ThreadGroup-based timeout checking for ChunkState
// This tests the new check(ThreadGroup&) method with leader-only checking
__global__ void chunkStateThreadGroupTimeoutKernel(
    ChunkState* state,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(SyncScope::WARP);

  // State is initialized to READY_TO_SEND (-1), so waiting for stepId=0
  // will spin forever unless timeout triggers
  // Uses ThreadGroup-based wait which calls timeout.check(group)
  state->wait_ready_to_recv(group, 0, 0, timeout);
}

// Kernel that uses ThreadGroup-based timeout checking for SignalState
__global__ void signalStateThreadGroupTimeoutKernel(
    SignalState* state,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(SyncScope::WARP);

  // State is initialized to 0, so waiting for value 1 will spin forever
  // Uses ThreadGroup-based wait which calls timeout.check(group)
  state->wait_until(group, CmpOp::CMP_EQ, 1, timeout);
}

// Kernel that calls start() twice - should trap on the second call
__global__ void doubleStartKernel(Timeout timeout) {
  timeout.start(); // First start - OK
  timeout.start(); // Second start - should trap!
}

// Simple kernel that sets a flag to indicate it ran
__global__ void setFlagKernel(int* flag) {
  *flag = 1;
}

// Kernel that will timeout and trap, used to test stream behavior
__global__ void timeoutTrapKernel(Timeout timeout) {
  timeout.start();
  // Spin until timeout expires, then trap
  while (true) {
    if (timeout.checkExpired()) {
      printf("CUDA TIMEOUT ERROR: timeoutTrapKernel timed out\n");
      __trap();
    }
  }
}

void launchChunkStateThreadGroupTimeoutKernel(int device, uint32_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device));

  // Allocate ChunkState on device
  ChunkState* d_state;
  CUDA_CHECK(cudaMalloc(&d_state, sizeof(ChunkState)));

  // Initialize to READY_TO_SEND (default constructor state)
  ChunkState h_state;
  CUDA_CHECK(cudaMemcpy(
      d_state, &h_state, sizeof(ChunkState), cudaMemcpyHostToDevice));

  // Create timeout configuration
  Timeout timeout = makeTimeout(timeout_ms, device);

  // Launch kernel with a full warp - should trap due to timeout
  // Leader-only checking means only thread 0 calls clock64()
  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  chunkStateThreadGroupTimeoutKernel<<<1, 32>>>(d_state, timeout);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaDeviceSynchronize();

  // Don't free - device will be reset by test
}

void launchSignalStateThreadGroupTimeoutKernel(
    int device,
    uint32_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device));

  // Allocate SignalState on device
  SignalState* d_state;
  CUDA_CHECK(cudaMalloc(&d_state, sizeof(SignalState)));

  // Initialize to 0 (default constructor state)
  SignalState h_state;
  CUDA_CHECK(cudaMemcpy(
      d_state, &h_state, sizeof(SignalState), cudaMemcpyHostToDevice));

  // Create timeout configuration
  Timeout timeout = makeTimeout(timeout_ms, device);

  // Launch kernel with a full warp - should trap due to timeout
  // Leader-only checking means only thread 0 calls clock64()
  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  signalStateThreadGroupTimeoutKernel<<<1, 32>>>(d_state, timeout);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaDeviceSynchronize();

  // Don't free - device will be reset by test
}

void launchDoubleStartKernel(int device, uint32_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device));

  // Create timeout configuration
  Timeout timeout = makeTimeout(timeout_ms, device);

  // Launch kernel - should trap on second start() call
  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  doubleStartKernel<<<1, 1>>>(timeout);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaDeviceSynchronize();
}

bool launchMultipleKernelsOnStreamTest(int device, uint32_t timeout_ms) {
  CUDA_CHECK(cudaSetDevice(device));

  // Create a stream for ordered execution
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Allocate a flag on device to track if second kernel ran
  int* d_flag;
  CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
  int zero = 0;
  CUDA_CHECK(cudaMemcpy(d_flag, &zero, sizeof(int), cudaMemcpyHostToDevice));

  // Create timeout configuration
  Timeout timeout = makeTimeout(timeout_ms, device);

  // Launch first kernel that will trap due to timeout
  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  timeoutTrapKernel<<<1, 1, 0, stream>>>(timeout);

  // Launch second kernel on the same stream - should NOT run if first traps
  // Intentionally unchecked - previous kernel will trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  setFlagKernel<<<1, 1, 0, stream>>>(d_flag);

  // Synchronize - this will return an error due to the trap
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaDeviceSynchronize();

  // Copy flag back to check if second kernel ran
  // Note: After a trap, the context is corrupted, so this may fail
  // but we try anyway to verify the second kernel didn't run
  int h_flag = -1; // Use -1 as sentinel
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaError_t copyErr =
      cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

  // If copy failed or flag is still 0, second kernel did not run
  // Return true if second kernel did NOT run (expected behavior)
  return (copyErr != cudaSuccess || h_flag == 0);
}

} // namespace comms::pipes::test
