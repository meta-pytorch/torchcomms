// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel implementations for DeviceApiIteratedTest (NCCLx backend).

#include "DeviceApiIteratedTestKernels.cuh"
#include "IteratedTestKernelUtils.cuh"

#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

// Kernel launch error check for test code.
#define KERNEL_LAUNCH_CHECK()                        \
  do {                                               \
    cudaError_t err__ = cudaGetLastError();          \
    assert(err__ == cudaSuccess && "kernel launch"); \
    (void)err__;                                     \
  } while (0)

namespace torchcomms::device::test {

// ---------------------------------------------------------------------------
// Iterated Put Kernel
// ---------------------------------------------------------------------------
// Each iteration: fill src -> put to dst_rank -> wait signal from src_rank ->
// verify received data. Uses monotonic signal values (signal = iter+1).
__global__ void iteratedPutKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    float* src_ptr,
    float* win_base,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    size_t count,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope,
    int* results) {
  int rank = win->rank();

  for (int iter = 0; iter < iterations; iter++) {
    // Fill source buffer with pattern for this iteration
    fillPattern(src_ptr, count, rank, iter);
    __syncthreads();

    // Put to destination with signal notification
    // Use monotonic signal values: each put increments signal by 1
    win->put(
        dst_offset, src_buf, src_offset, dst_rank, bytes, signal_id, -1, scope);
    win->flush(scope);

    // Wait for signal from src_rank (monotonic: iter+1)
    win->wait_signal(
        signal_id, CmpOp::GE, static_cast<uint64_t>(iter + 1), scope);

    // Verify received data from src_rank
    float* recv_slot = win_base + src_rank * count;
    verifyPattern(recv_slot, count, src_rank, iter, &results[iter]);

    // Barrier ensures both ranks finish reading before either starts the next
    // iteration's put (which overwrites the receive slot)
    win->barrier(iter % 2, scope);
  }
}

void launchIteratedPutKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    float* src_ptr,
    float* win_base,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    size_t count,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope,
    int num_threads,
    int* results,
    cudaStream_t stream) {
  iteratedPutKernel<<<1, num_threads, 0, stream>>>(
      win,
      src_buf,
      src_ptr,
      win_base,
      src_offset,
      dst_offset,
      bytes,
      count,
      dst_rank,
      src_rank,
      signal_id,
      iterations,
      scope,
      results);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Iterated Signal Kernel
// ---------------------------------------------------------------------------
// Ring signal pattern: rank i signals rank (i+1), waits for signal from (i-1).
// Uses monotonic signal values.

__global__ void iteratedSignalKernel(
    DeviceWindowNCCL* win,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope) {
  for (int iter = 0; iter < iterations; iter++) {
    // All threads must call signal() — GIN cooperative ops (ncclCoopWarp,
    // ncclCoopCta) require all threads in the group to participate.
    win->signal(dst_rank, signal_id, SignalOp::ADD, 1, scope);
    __syncthreads();

    // Wait for signal from previous rank (monotonic)
    win->wait_signal_from(
        src_rank, signal_id, CmpOp::GE, static_cast<uint64_t>(iter + 1), scope);
  }
}

void launchIteratedSignalKernel(
    DeviceWindowNCCL* win,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream) {
  iteratedSignalKernel<<<1, num_threads, 0, stream>>>(
      win, dst_rank, src_rank, signal_id, iterations, scope);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Iterated Barrier Kernel
// ---------------------------------------------------------------------------
// Calls barrier repeatedly, alternating between two barrier IDs.

__global__ void
iteratedBarrierKernel(DeviceWindowNCCL* win, int iterations, CoopScope scope) {
  for (int iter = 0; iter < iterations; iter++) {
    int barrier_id = iter % 2;
    win->barrier(barrier_id, scope);
  }
}

void launchIteratedBarrierKernel(
    DeviceWindowNCCL* win,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream) {
  iteratedBarrierKernel<<<1, num_threads, 0, stream>>>(win, iterations, scope);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Combined Ops Kernel
// ---------------------------------------------------------------------------
// Each iteration: barrier -> fill -> put -> wait_signal -> verify -> barrier

__global__ void iteratedCombinedKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    float* src_ptr,
    float* win_base,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    size_t count,
    int dst_rank,
    int src_rank,
    int signal_id,
    int barrier_id_base,
    int iterations,
    int* results) {
  int rank = win->rank();

  for (int iter = 0; iter < iterations; iter++) {
    // Pre-barrier: synchronize all ranks before this iteration
    win->barrier(barrier_id_base + (iter % 2));

    // Fill source with iteration-specific pattern
    fillPattern(src_ptr, count, rank, iter);

    // Put with signal (thread scope for combined test)
    if (threadIdx.x == 0) {
      win->put(dst_offset, src_buf, src_offset, dst_rank, bytes, signal_id, -1);
      win->flush();
    }
    __syncthreads();

    // Wait for data from src_rank
    if (threadIdx.x == 0) {
      win->wait_signal(signal_id, CmpOp::GE, static_cast<uint64_t>(iter + 1));
    }
    __syncthreads();

    // Verify received data
    float* recv_slot = win_base + src_rank * count;
    verifyPattern(recv_slot, count, src_rank, iter, &results[iter]);
    __syncthreads();
  }
}

void launchIteratedCombinedKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    float* src_ptr,
    float* win_base,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    size_t count,
    int dst_rank,
    int src_rank,
    int signal_id,
    int barrier_id_base,
    int iterations,
    int* results,
    cudaStream_t stream) {
  iteratedCombinedKernel<<<1, 1, 0, stream>>>(
      win,
      src_buf,
      src_ptr,
      win_base,
      src_offset,
      dst_offset,
      bytes,
      count,
      dst_rank,
      src_rank,
      signal_id,
      barrier_id_base,
      iterations,
      results);
  KERNEL_LAUNCH_CHECK();
}

} // namespace torchcomms::device::test
