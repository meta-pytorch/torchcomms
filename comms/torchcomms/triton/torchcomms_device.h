// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Triton Device API - C-style declarations for LLVM bitcode
//
// This header declares C-style wrapper functions for TorchComms Device API.
// Compiled to LLVM bitcode with clang for linking with Triton kernels.
//
// All functions use opaque void* handles to avoid C++ type dependencies.
// The actual implementations (in torchcomms_device.cu) cast these to the
// appropriate TorchComms types.
//
// Usage:
//   1. Compile this header with clang to produce bitcode (.bc)
//   2. Pass the .bc file to Triton via extern_libs parameter
//   3. Use @core.extern to declare these functions in Triton kernels
//   4. Triton statically links the bitcode into generated kernel PTX at JIT
//   time

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
// These are void* to avoid any NCCLX header dependencies.
// The actual types are resolved in the nvcc-compiled implementation.
typedef void* TorchCommsWindowHandle;
typedef void* TorchCommsBufferHandle;

// =============================================================================
// RMA Operations
// =============================================================================

// Put data from local registered buffer to remote window.
// The source buffer is specified by its components (base_ptr, size, nccl_win)
// rather than a pointer to a RegisteredBuffer struct. This avoids GPU memory
// allocation conflicts with NCCLX's cuMemMap-based memory management.
// Returns: 0 on success, negative on error
__device__ int torchcomms_put(
    TorchCommsWindowHandle win,
    unsigned long long dst_offset,
    void* src_base_ptr,
    unsigned long long src_size,
    void* src_nccl_win,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long bytes,
    int signal_id,
    int counter_id);

// =============================================================================
// Signal Operations (Remote Notification)
// =============================================================================

// Send signal to remote peer (atomic add)
// Returns: 0 on success, negative on error
__device__ int torchcomms_signal(
    TorchCommsWindowHandle win,
    int peer,
    int signal_id,
    unsigned long long value);

// Wait for signal to reach expected value (>=)
// Returns: 0 on success, negative on error
__device__ int torchcomms_wait_signal(
    TorchCommsWindowHandle win,
    int signal_id,
    unsigned long long expected_value);

// Read current signal value
__device__ unsigned long long torchcomms_read_signal(
    TorchCommsWindowHandle win,
    int signal_id);

// Reset signal to zero
__device__ void torchcomms_reset_signal(
    TorchCommsWindowHandle win,
    int signal_id);

// =============================================================================
// Counter Operations (Local Completion)
// =============================================================================

// Wait for local counter to reach expected value (>=)
// Returns: 0 on success, negative on error
__device__ int torchcomms_wait_local(
    TorchCommsWindowHandle win,
    int counter_id,
    unsigned long long expected_value);

// Read current counter value
__device__ unsigned long long torchcomms_read_counter(
    TorchCommsWindowHandle win,
    int counter_id);

// Reset counter to zero
__device__ void torchcomms_reset_counter(
    TorchCommsWindowHandle win,
    int counter_id);

// =============================================================================
// Synchronization & Completion
// =============================================================================

// Memory fence (ordering guarantee)
// Returns: 0 on success
__device__ int torchcomms_fence(TorchCommsWindowHandle win);

// Flush all pending operations
// Returns: 0 on success
__device__ int torchcomms_flush(TorchCommsWindowHandle win);

// =============================================================================
// Window Properties
// =============================================================================

// Get rank of this process
__device__ int torchcomms_rank(TorchCommsWindowHandle win);

// Get total number of ranks
__device__ int torchcomms_num_ranks(TorchCommsWindowHandle win);

// Get window base pointer
__device__ void* torchcomms_base(TorchCommsWindowHandle win);

// Get window size in bytes
__device__ unsigned long long torchcomms_size(TorchCommsWindowHandle win);

#ifdef __cplusplus
}
#endif
