// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel implementations for PipesDeviceApiIteratedTest (Pipes backend).
// Key difference from NCCLx: Pipes does NOT support reset_signal (traps!).
// All signal patterns use monotonic values.

#include "IteratedTestKernelUtils.cuh"
#include "PipesDeviceApiIteratedTestKernels.cuh"

#include "comms/torchcomms/device/pipes/TorchCommDevicePipes.cuh"

// Kernel launch error check for test code.
#define KERNEL_LAUNCH_CHECK()                        \
  do {                                               \
    cudaError_t err__ = cudaGetLastError();          \
    assert(err__ == cudaSuccess && "kernel launch"); \
    (void)err__;                                     \
  } while (0)

namespace torchcomms::device::test {

// ---------------------------------------------------------------------------
// Iterated Put Kernel (Pipes)
// ---------------------------------------------------------------------------
__global__ void pipesIteratedPutKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
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
    fillPattern(src_ptr, count, rank, iter);
    __syncthreads();

    // Monotonic signals: each put adds 1 to signal_id on destination
    win->put(
        dst_offset, src_buf, src_offset, dst_rank, bytes, signal_id, -1, scope);
    win->flush(scope);

    // Wait for monotonic signal value (no reset — Pipes doesn't support it)
    win->wait_signal(
        signal_id, CmpOp::GE, static_cast<uint64_t>(iter + 1), scope);

    float* recv_slot = win_base + src_rank * count;
    verifyPattern(recv_slot, count, src_rank, iter, &results[iter]);
    __syncthreads();

    // Reverse signal: tell the sender we're done reading, so it can safely
    // overwrite our receive slot in the next iteration.
    // All threads must call signal() — Pipes signal_peer(group, ...) has
    // group.sync() internally, so all threads in the group must participate.
    win->signal(src_rank, signal_id + 1, SignalOp::ADD, 1, scope);
    __syncthreads();

    // Wait for receiver of our data to finish reading before next put
    win->wait_signal(
        signal_id + 1, CmpOp::GE, static_cast<uint64_t>(iter + 1), scope);
  }
}

void launchPipesIteratedPutKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
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
  pipesIteratedPutKernel<<<1, num_threads, 0, stream>>>(
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
// Iterated Signal Kernel (Pipes)
// ---------------------------------------------------------------------------
__global__ void pipesIteratedSignalKernel(
    DeviceWindowPipes* win,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope) {
  for (int iter = 0; iter < iterations; iter++) {
    // All threads must call signal() — Pipes signal_peer(group, ...) has
    // group.sync() internally, so all threads in the group must participate.
    win->signal(dst_rank, signal_id, SignalOp::ADD, 1, scope);
    __syncthreads();

    win->wait_signal_from(
        src_rank, signal_id, CmpOp::GE, static_cast<uint64_t>(iter + 1), scope);
  }
}

void launchPipesIteratedSignalKernel(
    DeviceWindowPipes* win,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream) {
  pipesIteratedSignalKernel<<<1, num_threads, 0, stream>>>(
      win, dst_rank, src_rank, signal_id, iterations, scope);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Iterated Barrier Kernel (Pipes)
// ---------------------------------------------------------------------------
__global__ void pipesIteratedBarrierKernel(
    DeviceWindowPipes* win,
    int iterations,
    CoopScope scope) {
  for (int iter = 0; iter < iterations; iter++) {
    // Pipes uses a shared barrierExpected_ counter across all barrier IDs,
    // so we must reuse the same barrier_id (not alternate like NCCLx).
    win->barrier(0, scope);
  }
}

void launchPipesIteratedBarrierKernel(
    DeviceWindowPipes* win,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream) {
  pipesIteratedBarrierKernel<<<1, num_threads, 0, stream>>>(
      win, iterations, scope);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Combined Ops Kernel (Pipes)
// ---------------------------------------------------------------------------
__global__ void pipesIteratedCombinedKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
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
    // Pipes uses a shared barrierExpected_ counter — must reuse same ID
    win->barrier(barrier_id_base);

    fillPattern(src_ptr, count, rank, iter);

    if (threadIdx.x == 0) {
      win->put(dst_offset, src_buf, src_offset, dst_rank, bytes, signal_id, -1);
      win->flush();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      win->wait_signal(signal_id, CmpOp::GE, static_cast<uint64_t>(iter + 1));
    }
    __syncthreads();

    float* recv_slot = win_base + src_rank * count;
    verifyPattern(recv_slot, count, src_rank, iter, &results[iter]);
    __syncthreads();
  }
}

void launchPipesIteratedCombinedKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
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
  pipesIteratedCombinedKernel<<<1, 1, 0, stream>>>(
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
