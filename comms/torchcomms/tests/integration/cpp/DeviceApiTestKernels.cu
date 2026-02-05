// Copyright (c) Meta Platforms, Inc. and affiliates.
// CUDA kernels for DeviceApiTest - tests device-side communication primitives

#include "DeviceApiTestKernels.cuh"

// Include the NCCLX device API implementation (header-only)
#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

namespace torchcomms::device::test {

// =============================================================================
// Device Put Test Kernel
// =============================================================================
// This kernel performs a ring-based put operation where each rank puts data
// to the next rank: rank 0 -> rank 1, rank 1 -> rank 2, ..., rank N-1 -> rank 0
//
// Pattern:
//   1. Each rank puts its data to dst_rank = (rank + 1) % size
//   2. After put, signal the destination rank
//   3. Wait for signal from src_rank = (rank - 1 + size) % size
//   4. Verify data arrived correctly (done on host side)
//
// Note: win is a DEVICE pointer (allocated via cudaMalloc on host side)

__global__ void devicePutKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t bytes,
    int dst_rank,
    int signal_id) {
  // Only thread 0 performs the operation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int rank = win->rank();

    // Put data from local src_buf to destination window at offset = rank *
    // bytes This means each rank writes to its own "slot" in the destination
    // window
    size_t dst_offset = rank * bytes;
    size_t src_offset = 0;

    // Put with signal (no counter needed for this simple test)
    win->put(dst_offset, src_buf, src_offset, dst_rank, bytes, signal_id, -1);

    // Flush to ensure put completes
    win->flush();
  }
}

// Kernel with explicit source and destination offsets
// This is used when source and destination are different sections of the same
// window buffer.
__global__ void devicePutKernelWithOffsets(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    int dst_rank,
    int signal_id) {
  // Only thread 0 performs the operation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Put with signal (no counter needed for this simple test)
    win->put(dst_offset, src_buf, src_offset, dst_rank, bytes, signal_id, -1);

    // Flush to ensure put completes
    win->flush();
  }
}

__global__ void deviceWaitSignalKernel(
    DeviceWindowNCCL* win,
    int signal_id,
    uint64_t expected_value) {
  // Only thread 0 performs the operation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Wait for signal from sender
    win->wait_signal(signal_id, CmpOp::GE, expected_value);
  }
}

__global__ void deviceResetSignalKernel(DeviceWindowNCCL* win, int signal_id) {
  // Only thread 0 performs the operation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    win->reset_signal(signal_id);
  }
}

// =============================================================================
// Host-callable wrapper functions
// =============================================================================

void launchDevicePutKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t bytes,
    int dst_rank,
    int signal_id,
    cudaStream_t stream) {
  devicePutKernel<<<1, 1, 0, stream>>>(
      win, src_buf, bytes, dst_rank, signal_id);
}

void launchDevicePutKernelWithOffsets(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    int dst_rank,
    int signal_id,
    cudaStream_t stream) {
  devicePutKernelWithOffsets<<<1, 1, 0, stream>>>(
      win, src_buf, src_offset, dst_offset, bytes, dst_rank, signal_id);
}

void launchDeviceWaitSignalKernel(
    DeviceWindowNCCL* win,
    int signal_id,
    uint64_t expected_value,
    cudaStream_t stream) {
  deviceWaitSignalKernel<<<1, 1, 0, stream>>>(win, signal_id, expected_value);
}

void launchDeviceResetSignalKernel(
    DeviceWindowNCCL* win,
    int signal_id,
    cudaStream_t stream) {
  deviceResetSignalKernel<<<1, 1, 0, stream>>>(win, signal_id);
}

} // namespace torchcomms::device::test
