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
//
// Design:
//   - Block-scope ops (put_block, signal_block, flush_block, barrier_block):
//     All 128 block threads call these simultaneously (Triton
//     extern_elementwise). The caller must invoke convergently (no divergent
//     control flow before the call site).
//       - put_block/signal_block: delegate to win->put()/win->signal() with
//         CoopScope::BLOCK, which handles LSA and GIN internally. LSA: all
//         threads cooperate on memcpy_vectorized; signal() uses
//         atom.release.sys to order prior stores. GIN: ncclCoopCta{} emits
//         __syncthreads__ before/after posting the WQE.
//       - flush_block/barrier_block: threadIdx.x == 0 guard + CoopScope::THREAD
//         to avoid __syncthreads__ in gin.flush()/ncclBarrierSession, which
//         would deadlock if threads entered the internal barrier at different
//         times. Non-zero threads return immediately.
//   - Thread-scope ops (wait_signal, fence, read/reset, rank, etc.):
//     Idempotent w.r.t. thread count — spin-polls, PTX fences, and atomic
//     reads produce the same result whether called from 1 or 128 threads.
//     All 128 threads do call these (extern_elementwise), which is harmless.

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
// Block-scope RMA Operations
//
// ALL threads in the calling block must call these together (convergently).
// No threadIdx guard — cooperative execution across all block threads.
//
// win->put()/win->signal() with CoopScope::BLOCK handle both paths internally:
//   NVLink: all threads cooperate on vectorized memcpy; signal uses
//   atom.release.sys. GIN: __syncthreads() + thread 0 posts WQE +
//   __syncthreads().
//
// Contract:
//   - Must be called convergently — no divergent control flow before this call
//   - All threads in the block participate; do NOT gate on threadIdx inside
// =============================================================================

// Block-scope put: all block threads cooperate on the transfer.
// src buffer specified by its components (base_ptr, size, nccl_win) rather
// than a pointer to a RegisteredBuffer struct. This avoids GPU memory
// allocation conflicts with NCCLX's cuMemMap-based memory management.
// Returns: 0 on success, negative on error
__device__ int torchcomms_put_block(
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

// Block-scope signal: all block threads cooperate.
// Returns: 0 on success, negative on error
__device__ int torchcomms_signal_block(
    TorchCommsWindowHandle win,
    int peer,
    int signal_id,
    unsigned long long value);

// Block-scope flush: thread 0 polls per-peer completion queues
// (CoopScope::THREAD); other threads return immediately. Call convergently from
// all block threads. Returns: 0 on success
__device__ int torchcomms_flush_block(TorchCommsWindowHandle win);

// Block-scope barrier: thread 0 runs the protocol (CoopScope::THREAD);
// other threads return immediately. Call convergently from all block threads.
// Returns: 0 on success, negative on error
__device__ int torchcomms_barrier_block(
    TorchCommsWindowHandle win,
    int barrier_id);

// =============================================================================
// Signal Operations (Remote Notification)
// Thread-scope (idempotent) — all 128 threads call these; same result from any
// count.
// =============================================================================

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
// Thread-scope (idempotent) — all 128 threads call these; same result from any
// count.
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
// Thread-scope (idempotent) — all 128 threads call these; same result from any
// count.
// =============================================================================

// Memory fence (ordering guarantee)
// Returns: 0 on success
__device__ int torchcomms_fence(TorchCommsWindowHandle win);

// =============================================================================
// Window Properties
// Thread-scope (idempotent) — all 128 threads call these; same result from any
// count.
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
