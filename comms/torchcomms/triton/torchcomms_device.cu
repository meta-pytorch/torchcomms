// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Triton Device API - nvcc-compiled implementations
//
// This file provides extern "C" wrappers around TorchCommDeviceWindow methods.
// Compiled with nvcc to support NCCLX GIN templates.
// Linked at compile time into Triton kernels via extern_libs bitcode.
//
// Design:
//   - Block-scope ops (put_block, signal_block, flush_block, barrier_block):
//     All 128 block threads call these functions simultaneously (Triton
//     extern_elementwise invokes each extern once per thread). The caller must
//     invoke these convergently (no divergent control flow before the call
//     site).
//
//     put_block / signal_block:
//       Delegate to win->put() / win->signal() with CoopScope::BLOCK, which
//       handles both paths internally:
//       - LSA (NVLink): all threads cooperate on memcpy_vectorized; signal()
//         uses atom.release.sys to order prior stores before the signal write.
//       - GIN (RDMA): CoopScope::BLOCK → ncclCoopCta{} → __syncthreads__
//         before/after posting WQE. Safe because all threads enter
//         convergently.
//
//     flush_block / barrier_block:
//       Cannot use CoopScope::BLOCK — two problems:
//       (a) __syncthreads__: ncclGin::flush and ncclBarrierSession both emit
//           __syncthreads__ (via ncclCoopCta::sync()), risking deadlock if any
//           thread diverges before the call.
//       (b) barrier semantic: each thread independently signals peers in the
//           GIN barrier protocol, so 128 threads would each send 128x the
//           intended signal increments, corrupting epoch tracking.
//       Fix: threadIdx.x == 0 guard + CoopScope::THREAD (ncclCoopTile<1>
//       whose sync() is a compile-time no-op). Other threads return
//       immediately.
//
//   - Thread-scope ops (wait_signal, fence, read/reset, rank, etc.):
//     Idempotent w.r.t. thread count — spin-polls, PTX fences, and atomic
//     reads produce the same result whether called from 1 or 128 threads.
//     All 128 threads do call these (extern_elementwise), which is harmless.

#include <cuda_runtime.h>

#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

using namespace torchcomms::device;

using DeviceWindow = TorchCommDeviceWindow<NCCLGinBackend>;

extern "C" {

// =============================================================================
// Block-scope RMA Operations
//
// All 128 block threads call these functions simultaneously (Triton
// extern_elementwise invokes each extern once per thread). The caller must
// invoke convergently (no divergent control flow before the call site).
// See the file-level design comment for details per function.
// =============================================================================

// torchcomms_put_block: block-cooperative data transfer.
//
// win->put(CoopScope::BLOCK) handles LSA and GIN internally:
//   - LSA (NVLink): all threads cooperate on memcpy_vectorized; signal() uses
//     atom.release.sys which orders all prior stores before the signal write.
//   - GIN (RDMA): CoopScope::BLOCK → ncclCoopCta{} → __syncthreads__ before/
//     after posting WQE. Safe because all threads enter convergently.
//
// src buffer specified by its components (base_ptr, size, nccl_win) rather
// than a pointer to a RegisteredBuffer struct to avoid GPU memory allocation
// conflicts with NCCLX's cuMemMap-based memory management.
__device__ int torchcomms_put_block(
    void* win_ptr,
    unsigned long long dst_offset,
    void* src_base_ptr,
    unsigned long long src_size,
    void* src_nccl_win,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long bytes,
    int signal_id,
    int counter_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);

  RegisteredBuffer src_buf;
  src_buf.base_ptr = src_base_ptr;
  src_buf.size = static_cast<size_t>(src_size);
  src_buf.backend_window = src_nccl_win;

  return win->put(
      static_cast<size_t>(dst_offset),
      src_buf,
      static_cast<size_t>(src_offset),
      dst_rank,
      static_cast<size_t>(bytes),
      signal_id,
      counter_id,
      CoopScope::BLOCK);
}

__device__ int torchcomms_signal_block(
    void* win_ptr,
    int peer,
    int signal_id,
    unsigned long long value) {
  // LSA: signal() guards with thread_id_in_group==0 internally
  // (CoopScope::BLOCK). GIN: CoopScope::BLOCK → ncclCoopCta{} → __syncthreads__
  // before/after posting the atomic WQE. Safe when all block threads enter this
  // function convergently.
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->signal(peer, signal_id, SignalOp::ADD, value, CoopScope::BLOCK);
}

__device__ int torchcomms_flush_block(void* win_ptr) {
  // Cannot call flush(CoopScope::BLOCK) from all threads.
  //
  // gin.flush(ncclCoopCta{}) emits __syncthreads__ unconditionally (twice:
  // before and after the peer-poll loop in ncclGin_BackendMask::flush). If
  // any thread reaches that barrier without the others (e.g. due to divergent
  // control flow in the caller), the block hangs forever.
  //
  // Fix: only thread 0 runs flush(CoopScope::THREAD), which uses
  // ncclCoopTile<1> whose sync() is a compile-time no-op. Other threads return
  // immediately.
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->flush(CoopScope::THREAD);
}

__device__ int torchcomms_barrier_block(void* win_ptr, int barrier_id) {
  // Cannot call barrier(CoopScope::BLOCK) from all threads — two problems:
  //
  // 1. __syncthreads__: ncclBarrierSession emits __syncthreads__ multiple
  //    times (LSA arrive/wait, GIN sync×2, destructors). Same deadlock risk
  //    as flush_block if any thread diverges before the call.
  //
  // 2. Semantic correctness: the GIN barrier's signal loop runs independently
  //    per thread — each thread signals (nRanks-1) peers. With 128 threads,
  //    each peer would receive 128× the intended signal increments, corrupting
  //    epoch tracking and firing the barrier protocol 128 times.
  //
  // Fix: only thread 0 runs barrier(CoopScope::THREAD). Others return
  // immediately.
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->barrier(barrier_id, CoopScope::THREAD);
}

// =============================================================================
// Signal Operations (Remote Notification)
// Thread-scope (idempotent) — all 128 threads call these; result is the same
// as if only one thread called. Spin-polls and atomic reads are thread-safe.
// =============================================================================

__device__ int torchcomms_wait_signal(
    void* win_ptr,
    int signal_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_signal(signal_id, CmpOp::GE, expected_value);
}

__device__ unsigned long long torchcomms_read_signal(
    void* win_ptr,
    int signal_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->read_signal(signal_id);
}

__device__ void torchcomms_reset_signal(void* win_ptr, int signal_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  win->reset_signal(signal_id);
}

// =============================================================================
// Counter Operations (Local Completion)
// Thread-scope (idempotent) — all 128 threads call these; result is the same
// as if only one thread called. Spin-polls and atomic reads are thread-safe.
// (wait_local/read_counter/reset_counter use ncclCoopThread{} internally.)
// =============================================================================

__device__ int torchcomms_wait_local(
    void* win_ptr,
    int counter_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_local(counter_id, CmpOp::GE, expected_value);
}

__device__ unsigned long long torchcomms_read_counter(
    void* win_ptr,
    int counter_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->read_counter(counter_id);
}

__device__ void torchcomms_reset_counter(void* win_ptr, int counter_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  win->reset_counter(counter_id);
}

// =============================================================================
// Synchronization & Completion
// Thread-scope (idempotent) — all 128 threads call these; result is the same
// as if only one thread called. Spin-polls and atomic reads are thread-safe.
// =============================================================================

__device__ int torchcomms_fence(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->fence();
}

// =============================================================================
// Window Properties
// Thread-scope (idempotent) — all 128 threads call these; result is the same
// as if only one thread called. Spin-polls and atomic reads are thread-safe.
// =============================================================================

__device__ int torchcomms_rank(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->rank();
}

__device__ int torchcomms_num_ranks(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->num_ranks();
}

__device__ void* torchcomms_base(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->base();
}

__device__ unsigned long long torchcomms_size(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return static_cast<unsigned long long>(win->size());
}

} // extern "C"
