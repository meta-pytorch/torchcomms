// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel implementations for DeviceApiStressTest (NCCLx backend).

#include "DeviceApiTestKernels.cuh"
#include "StressTestKernelUtils.cuh"

#include <cuda_fp16.h>
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
// Stress Put Kernel
// ---------------------------------------------------------------------------
// Each iteration: fill src -> put to dst_rank -> wait signal from src_rank ->
// verify received data. Uses monotonic signal values (signal = iter+1).
__global__ void stressPutKernel(
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

void launchStressPutKernel(
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
  stressPutKernel<<<1, num_threads, 0, stream>>>(
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
// Stress Signal Kernel
// ---------------------------------------------------------------------------
// Ring signal pattern: rank i signals rank (i+1), waits for signal from (i-1).
// Uses monotonic signal values.

__global__ void stressSignalKernel(
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

void launchStressSignalKernel(
    DeviceWindowNCCL* win,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream) {
  stressSignalKernel<<<1, num_threads, 0, stream>>>(
      win, dst_rank, src_rank, signal_id, iterations, scope);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Stress Barrier Kernel
// ---------------------------------------------------------------------------
// Calls barrier repeatedly, alternating between two barrier IDs.

__global__ void
stressBarrierKernel(DeviceWindowNCCL* win, int iterations, CoopScope scope) {
  for (int iter = 0; iter < iterations; iter++) {
    int barrier_id = iter % 2;
    win->barrier(barrier_id, scope);
  }
}

void launchStressBarrierKernel(
    DeviceWindowNCCL* win,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream) {
  stressBarrierKernel<<<1, num_threads, 0, stream>>>(win, iterations, scope);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Combined Ops Kernel
// ---------------------------------------------------------------------------
// Each iteration: barrier -> fill -> put -> wait_signal -> verify -> barrier

__global__ void stressCombinedKernel(
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

void launchStressCombinedKernel(
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
  stressCombinedKernel<<<1, 1, 0, stream>>>(
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
// Stress Aggregated Signal Kernel
// ---------------------------------------------------------------------------
// Each iteration: signal dst_rank -> aggregated wait_signal (not per-peer)
// -> read_signal -> reset_signal -> verify read_signal returns 0 after reset.

__global__ void stressAggregatedSignalKernel(
    DeviceWindowNCCL* win,
    int dst_rank,
    int signal_id,
    int iterations,
    int* results) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  for (int iter = 0; iter < iterations; iter++) {
    // Signal next rank
    win->signal(dst_rank, signal_id, SignalOp::ADD, 1);

    // Aggregated wait (sums all per-peer slots)
    win->wait_signal(signal_id, CmpOp::GE, static_cast<uint64_t>(1));

    // Read the aggregated signal value
    uint64_t sig_val = win->read_signal(signal_id);
    // It should be >= 1
    int ok = (sig_val >= 1) ? 1 : 0;

    // Reset signal back to 0
    win->reset_signal(signal_id);

    // Read again to verify reset
    uint64_t after_reset = win->read_signal(signal_id);
    if (after_reset != 0) {
      ok = 0;
    }

    results[iter] = ok;

    // Barrier to ensure both ranks have completed reset before next iteration
    win->barrier(iter % 2);
  }
}

void launchStressAggregatedSignalKernel(
    DeviceWindowNCCL* win,
    int dst_rank,
    int signal_id,
    int iterations,
    int* results,
    cudaStream_t stream) {
  stressAggregatedSignalKernel<<<1, 1, 0, stream>>>(
      win, dst_rank, signal_id, iterations, results);
  KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Stress Half-Precision Put Kernel
// ---------------------------------------------------------------------------
// Same as stressPutKernel but operates on __half data. Each iteration:
// fill src with half-precision pattern -> put -> wait_signal -> verify.

__global__ void stressPutHalfKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    __half* src_ptr,
    __half* win_base,
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
    // Fill source buffer with half-precision pattern
    // Use small values to avoid half overflow: (rank+1)*10 + iter%100
    __half val =
        __float2half(static_cast<float>((rank + 1) * 10 + (iter % 100)));
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
      src_ptr[i] = val;
    }
    __syncthreads();

    // Put to destination with signal notification
    win->put(
        dst_offset, src_buf, src_offset, dst_rank, bytes, signal_id, -1, scope);
    win->flush(scope);

    // Wait for signal from src_rank (monotonic: iter+1)
    win->wait_signal(
        signal_id, CmpOp::GE, static_cast<uint64_t>(iter + 1), scope);

    // Verify received data
    __half* recv_slot = win_base + src_rank * count;
    __half expected =
        __float2half(static_cast<float>((src_rank + 1) * 10 + (iter % 100)));

    // Thread-cooperative verification
    __shared__ int any_mismatch;
    if (threadIdx.x == 0) {
      any_mismatch = 0;
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
      float diff = __half2float(recv_slot[i]) - __half2float(expected);
      if (diff > 0.1f || diff < -0.1f) {
        atomicExch(&any_mismatch, 1);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      results[iter] = (any_mismatch == 0) ? 1 : 0;
    }

    // Barrier before next iteration
    win->barrier(iter % 2, scope);
  }
}

void launchStressPutHalfKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    void* src_ptr,
    void* win_base,
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
  stressPutHalfKernel<<<1, num_threads, 0, stream>>>(
      win,
      src_buf,
      static_cast<__half*>(src_ptr),
      static_cast<__half*>(win_base),
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
// Get Multimem Address Kernel
// ---------------------------------------------------------------------------
// Calls win->get_multimem_address(offset) on device and stores the pointer
// value for host-side comparison against the host API result.

__global__ void getMultimemAddressKernel(
    DeviceWindowNCCL* win,
    size_t offset,
    int64_t* result) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void* ptr = win->get_multimem_address(offset);
    *result = reinterpret_cast<int64_t>(ptr);
  }
}

void launchGetMultimemAddressKernel(
    DeviceWindowNCCL* win,
    size_t offset,
    int64_t* result,
    cudaStream_t stream) {
  getMultimemAddressKernel<<<1, 1, 0, stream>>>(win, offset, result);
  KERNEL_LAUNCH_CHECK();
}

} // namespace torchcomms::device::test
