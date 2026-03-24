// Copyright (c) Meta Platforms, Inc. and affiliates.
// CUDA kernels for PipesDeviceApiTest - tests device-side communication
// primitives using the Pipes backend (IBGDA + NVLink)

#include "PipesDeviceApiTestKernels.cuh"

// Include the Pipes device API implementation (header-only)
#include "comms/torchcomms/device/pipes/TorchCommDevicePipes.cuh"

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
// Standalone Signal Test Kernel
// =============================================================================
// Sends a signal to a peer without any associated data transfer.
// Used to test the per-peer signal model via Pipes transport.
//
// Signal semantics for Pipes:
//   - signal_id is ignored (slots are indexed by sender rank, not signal_id)
//   - NVL path: atomicAdd/store to remote signal slot at nvl_remote_signal_ptr
//   - IBGDA path: signal_remote_with_fence to peer's remote signal buffer

__global__ void pipesSignalKernel(
    DeviceWindowPipes* win,
    int peer,
    int signal_id,
    SignalOp op,
    uint64_t value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    win->signal(peer, signal_id, op, value);
  }
}

// =============================================================================
// Wait Signal Kernel
// =============================================================================
// Waits for aggregated signal from all peers to reach expected_value.
// Pipes aggregates by summing all per-sender slots in local signal buffer.

__global__ void pipesWaitSignalKernel(
    DeviceWindowPipes* win,
    int signal_id,
    uint64_t expected_value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    win->wait_signal(signal_id, CmpOp::GE, expected_value);
  }
}

// =============================================================================
// Reset Signal Kernel
// =============================================================================
// Resets all per-sender signal slots to 0.

__global__ void pipesResetSignalKernel(DeviceWindowPipes* win, int signal_id) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    win->reset_signal(signal_id);
  }
}

// =============================================================================
// Read Signal Kernel
// =============================================================================
// Reads the aggregated signal value (sum of all per-sender slots).

__global__ void
pipesReadSignalKernel(DeviceWindowPipes* win, int signal_id, uint64_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = win->read_signal(signal_id);
  }
}

// =============================================================================
// Wait Signal From Specific Peer Kernel
// =============================================================================
// Waits for a signal from a specific peer rank (point-to-point, not
// aggregated).

__global__ void pipesWaitSignalFromKernel(
    DeviceWindowPipes* win,
    int peer,
    int signal_id,
    CmpOp cmp,
    uint64_t value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    win->wait_signal_from(peer, signal_id, cmp, value);
  }
}

// =============================================================================
// Barrier Kernel
// =============================================================================
// Synchronizes all ranks via DeviceWindow::barrier().

__global__ void pipesBarrierKernel(DeviceWindowPipes* win, int barrier_id) {
  // WARP scope: all 32 threads participate in the barrier collectively.
  // NOTE: Pipes barriers use monotonic counters (barrierExpected_ accumulates).
  // When DeviceWindow is accessed via pointer (not by value), barrierExpected_
  // persists across kernel launches. Multiple barriers must reuse the same
  // barrier_id (counters accumulate in lockstep).
  win->barrier(barrier_id, CoopScope::WARP);
}

// =============================================================================
// Put with Signal Kernel
// =============================================================================
// Performs a put operation to dst_rank with signal notification.
// Single thread performs the put + flush (CoopScope::THREAD).

__global__ void pipesPutKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    int dst_rank,
    int signal_id) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    win->put(dst_offset, src_buf, src_offset, dst_rank, bytes, signal_id, -1);
    win->flush();
  }
}

// =============================================================================
// Put with Signal + Counter Kernel
// =============================================================================
// Performs a put with both signal and counter.
// Counter is only incremented for IBGDA peers (companion QP loopback atomic).
// For NVLink-only peers, counter stays 0 (silently ignored, same as GIN LSA).
// Caller should NOT wait_local for NVLink-only configs — it would spin forever.

__global__ void pipesPutCounterKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    int dst_rank,
    int signal_id,
    int counter_id) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
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
}

// =============================================================================
// Read Counter Kernel
// =============================================================================
// Reads the aggregated counter value (summed across all peers).

__global__ void
pipesReadCounterKernel(DeviceWindowPipes* win, int counter_id, uint64_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = win->read_counter(counter_id);
  }
}

// =============================================================================
// Reset Counter Kernel
// =============================================================================
// Resets counter for all peers.

__global__ void pipesResetCounterKernel(
    DeviceWindowPipes* win,
    int counter_id) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    win->reset_counter(counter_id);
  }
}

// =============================================================================
// Host-callable wrapper functions
// =============================================================================

void launchPipesSignalKernel(
    DeviceWindowPipes* win,
    int peer,
    int signal_id,
    SignalOp op,
    uint64_t value,
    cudaStream_t stream) {
  pipesSignalKernel<<<1, 1, 0, stream>>>(win, peer, signal_id, op, value);
  CUDA_LAUNCH_CHECK();
}

void launchPipesWaitSignalKernel(
    DeviceWindowPipes* win,
    int signal_id,
    uint64_t expected_value,
    cudaStream_t stream) {
  pipesWaitSignalKernel<<<1, 1, 0, stream>>>(win, signal_id, expected_value);
  CUDA_LAUNCH_CHECK();
}

void launchPipesResetSignalKernel(
    DeviceWindowPipes* win,
    int signal_id,
    cudaStream_t stream) {
  pipesResetSignalKernel<<<1, 1, 0, stream>>>(win, signal_id);
  CUDA_LAUNCH_CHECK();
}

void launchPipesReadSignalKernel(
    DeviceWindowPipes* win,
    int signal_id,
    uint64_t* out,
    cudaStream_t stream) {
  pipesReadSignalKernel<<<1, 1, 0, stream>>>(win, signal_id, out);
  CUDA_LAUNCH_CHECK();
}

void launchPipesWaitSignalFromKernel(
    DeviceWindowPipes* win,
    int peer,
    int signal_id,
    CmpOp cmp,
    uint64_t value,
    cudaStream_t stream) {
  pipesWaitSignalFromKernel<<<1, 1, 0, stream>>>(
      win, peer, signal_id, cmp, value);
  CUDA_LAUNCH_CHECK();
}

void launchPipesBarrierKernel(
    DeviceWindowPipes* win,
    int barrier_id,
    cudaStream_t stream) {
  // Launch with 32 threads (1 warp) to match CoopScope::WARP in the kernel.
  pipesBarrierKernel<<<1, 32, 0, stream>>>(win, barrier_id);
  CUDA_LAUNCH_CHECK();
}

void launchPipesPutKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    int dst_rank,
    int signal_id,
    cudaStream_t stream) {
  pipesPutKernel<<<1, 1, 0, stream>>>(
      win, src_buf, src_offset, dst_offset, bytes, dst_rank, signal_id);
  CUDA_LAUNCH_CHECK();
}

void launchPipesPutCounterKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    int dst_rank,
    int signal_id,
    int counter_id,
    cudaStream_t stream) {
  pipesPutCounterKernel<<<1, 1, 0, stream>>>(
      win,
      src_buf,
      src_offset,
      dst_offset,
      bytes,
      dst_rank,
      signal_id,
      counter_id);
  CUDA_LAUNCH_CHECK();
}

void launchPipesReadCounterKernel(
    DeviceWindowPipes* win,
    int counter_id,
    uint64_t* out,
    cudaStream_t stream) {
  pipesReadCounterKernel<<<1, 1, 0, stream>>>(win, counter_id, out);
  CUDA_LAUNCH_CHECK();
}

void launchPipesResetCounterKernel(
    DeviceWindowPipes* win,
    int counter_id,
    cudaStream_t stream) {
  pipesResetCounterKernel<<<1, 1, 0, stream>>>(win, counter_id);
  CUDA_LAUNCH_CHECK();
}

} // namespace torchcomms::device::test
