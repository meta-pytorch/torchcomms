// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel implementations for PipesDeviceApiTest (Pipes backend).
// Key difference from NCCLx: Pipes does NOT support reset_signal (traps!).
// All signal patterns use monotonic values.

#include "PipesDeviceApiTestKernels.cuh"
#include "StressTestKernelUtils.cuh"

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
// Stress Put Kernel (Pipes)
// ---------------------------------------------------------------------------
__global__ void pipesStressPutKernel(
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

void launchPipesStressPutKernel(
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
  pipesStressPutKernel<<<1, num_threads, 0, stream>>>(
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
// Stress Signal Kernel (Pipes)
// ---------------------------------------------------------------------------
__global__ void pipesStressSignalKernel(
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

void launchPipesStressSignalKernel(
    DeviceWindowPipes* win,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream) {
  pipesStressSignalKernel<<<1, num_threads, 0, stream>>>(
      win, dst_rank, src_rank, signal_id, iterations, scope);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Stress Barrier Kernel (Pipes)
// ---------------------------------------------------------------------------
__global__ void pipesStressBarrierKernel(
    DeviceWindowPipes* win,
    int iterations,
    CoopScope scope) {
  for (int iter = 0; iter < iterations; iter++) {
    // Pipes uses a shared barrierExpected_ counter across all barrier IDs,
    // so we must reuse the same barrier_id (not alternate like NCCLx).
    win->barrier(0, scope);
  }
}

void launchPipesStressBarrierKernel(
    DeviceWindowPipes* win,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream) {
  pipesStressBarrierKernel<<<1, num_threads, 0, stream>>>(
      win, iterations, scope);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Combined Ops Kernel (Pipes)
// ---------------------------------------------------------------------------
__global__ void pipesStressCombinedKernel(
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

void launchPipesStressCombinedKernel(
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
  pipesStressCombinedKernel<<<1, 1, 0, stream>>>(
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

// ---------------------------------------------------------------------------
// Stress Put with Counter Kernel (Pipes)
// ---------------------------------------------------------------------------
// Each iteration: fill src, put with signal+counter, wait_counter (if IBGDA
// counters are active), wait_signal on receiver, verify data, write
// counter value to output array. Uses monotonic signals (no reset).
__global__ void pipesStressPutCounterKernel(
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
    int counter_id,
    int barrier_id,
    int iterations,
    int* results,
    uint64_t* counter_values) {
  int rank = win->rank();

  for (int iter = 0; iter < iterations; iter++) {
    // Barrier ensures all ranks finished verifying previous iteration before
    // any rank starts the next put (prevents data race on receive slots).
    win->barrier(barrier_id);

    fillPattern(src_ptr, count, rank, iter);
    __syncthreads();

    if (threadIdx.x == 0) {
      // Put with signal + counter
      win->put(
          dst_offset,
          src_buf,
          src_offset,
          dst_rank,
          bytes,
          signal_id,
          counter_id);
      win->flush();
    }
    __syncthreads();

    // Read counter value — IBGDA peers increment it, NVLink peers leave at 0
    if (threadIdx.x == 0) {
      counter_values[iter] = win->read_counter(counter_id);
    }
    __syncthreads();

    // Wait for signal from sender (monotonic)
    if (threadIdx.x == 0) {
      win->wait_signal(signal_id, CmpOp::GE, static_cast<uint64_t>(iter + 1));
    }
    __syncthreads();

    float* recv_slot = win_base + src_rank * count;
    verifyPattern(recv_slot, count, src_rank, iter, &results[iter]);
    __syncthreads();

    // Reset counter for next iteration
    if (threadIdx.x == 0) {
      win->reset_counter(counter_id);
    }
    __syncthreads();
  }
}

void launchPipesStressPutCounterKernel(
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
    int counter_id,
    int barrier_id,
    int iterations,
    int* results,
    uint64_t* counter_values,
    cudaStream_t stream) {
  pipesStressPutCounterKernel<<<1, 1, 0, stream>>>(
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
      counter_id,
      barrier_id,
      iterations,
      results,
      counter_values);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Read Signal Kernel (for host-side verification)
// ---------------------------------------------------------------------------
__global__ void
pipesReadSignalKernel_(DeviceWindowPipes* win, int signal_id, uint64_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = win->read_signal(signal_id);
  }
}

void launchPipesReadSignalKernel(
    DeviceWindowPipes* win,
    int signal_id,
    uint64_t* out,
    cudaStream_t stream) {
  pipesReadSignalKernel_<<<1, 1, 0, stream>>>(win, signal_id, out);
  KERNEL_LAUNCH_CHECK();
}

} // namespace torchcomms::device::test
