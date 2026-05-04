// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Triton Device Window - generic (backend-agnostic) implementations
//
// This file provides extern "C" wrappers around TorchCommDeviceWindow methods.
// It is compiled TWICE to produce two bitcode files:
//   - Without USE_PIPES_BACKEND → libdevice_window.bc (GIN/NCCLGinBackend)
//   - With USE_PIPES_BACKEND    → libdevice_window_pipes.bc
//   (PipesDeviceBackend)
//
// All functions in this file use the generic TorchCommDeviceWindow API
// (win->put(), win->signal(), win->flush(), etc.) and work with both backends.
//
// For GIN-specific NVLink-optimized put operations (put_block_direct,
// put_warp_chunked_direct), see device_window_nvl_opt.cu.
//
// Design:
//   - Block-scope ops (put_block, signal_block, flush_block, barrier_block):
//     All 128 block threads call these functions simultaneously (Triton
//     extern_elementwise invokes each extern once per thread). The caller must
//     invoke these convergently (no divergent control flow before the call
//     site).
//   - Thread-scope ops (wait_signal, fence, read/reset, rank, etc.):
//     Idempotent w.r.t. thread count — spin-polls, PTX fences, and atomic
//     reads produce the same result whether called from 1 or 128 threads.
//     All 128 threads do call these (extern_elementwise), which is harmless.

#include <cuda_runtime.h>

#ifdef USE_PIPES_BACKEND
#include "comms/torchcomms/device/pipes/TorchCommDevicePipes.cuh"
#else
#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"
#endif

using namespace torchcomms::device;
using torch::comms::RegisteredBuffer;

#ifdef USE_PIPES_BACKEND
using DeviceWindow = TorchCommDeviceWindow<PipesDeviceBackend>;
#else
using DeviceWindow = TorchCommDeviceWindow<NCCLGinBackend>;
#endif

extern "C" {

// =============================================================================
// Block-scope RMA Operations
// =============================================================================

// torchcomms_self_copy_block: block-cooperative local memory copy for
// self-send (peer == my_rank) in alltoallv.
__device__ int torchcomms_self_copy_block(
    void* dst_ptr,
    unsigned long long dst_offset,
    void* src_ptr,
    unsigned long long src_offset,
    unsigned long long bytes) {
  auto* dst = reinterpret_cast<char*>(dst_ptr) + dst_offset;
  auto* src = reinterpret_cast<const char*>(src_ptr) + src_offset;
#ifdef USE_PIPES_BACKEND
  auto group = detail::make_pipes_thread_group(CoopScope::BLOCK);
#else
  auto group = detail::make_thread_group(CoopScope::BLOCK);
#endif
  comms::pipes::memcpy_vectorized(dst, src, static_cast<size_t>(bytes), group);
  return 0;
}

// torchcomms_put_block: block-cooperative data transfer.
//
// win->put(CoopScope::BLOCK) handles both paths internally:
//   - GIN/LSA (NVLink): all threads cooperate on memcpy_vectorized.
//   - GIN (RDMA): CoopScope::BLOCK → ncclCoopCta{} → __syncthreads__.
//   - Pipes (NVLink): all threads cooperate via ThreadGroup memcpy.
//   - Pipes (IBGDA): leader thread posts RDMA write via DOCA GPUNetIO.
__device__ int torchcomms_put_block(
    void* win_ptr,
    unsigned long long dst_offset,
    void* src_buf_ptr,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long bytes,
    int signal_id,
    int counter_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  const auto& src_buf = *reinterpret_cast<const RegisteredBuffer*>(src_buf_ptr);

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
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->signal(peer, signal_id, SignalOp::ADD, value, CoopScope::BLOCK);
}

__device__ int torchcomms_flush_block(void* win_ptr) {
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->flush(CoopScope::THREAD);
}

__device__ int torchcomms_barrier_block(void* win_ptr, int barrier_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->barrier(barrier_id, CoopScope::BLOCK);
}

// =============================================================================
// Block-scope Wait Operations
//
// Thread 0 polls the signal/counter; remaining threads synchronize via
// __syncthreads__ (CoopScope::BLOCK → make_thread_group → group.sync()).
// Reduces spin-poll traffic from N acquire loads per poll cycle
// (thread-scope, N = blockDim.x) to 1 acquire load + 1 __syncthreads__.
// =============================================================================

__device__ int torchcomms_wait_signal_from_block(
    void* win_ptr,
    int peer,
    int signal_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_signal_from(
      peer, signal_id, CmpOp::GE, expected_value, CoopScope::BLOCK);
}

__device__ int torchcomms_wait_counter_block(
    void* win_ptr,
    int counter_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_counter(
      counter_id, CmpOp::GE, expected_value, CoopScope::BLOCK);
}

// =============================================================================
// Signal Operations (Remote Notification)
// Thread-scope (idempotent)
// =============================================================================

__device__ int torchcomms_wait_signal(
    void* win_ptr,
    int signal_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_signal(signal_id, CmpOp::GE, expected_value);
}

__device__ int torchcomms_wait_signal_from(
    void* win_ptr,
    int peer,
    int signal_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_signal_from(peer, signal_id, CmpOp::GE, expected_value);
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
// Thread-scope (idempotent)
// =============================================================================

__device__ int torchcomms_wait_counter(
    void* win_ptr,
    int counter_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_counter(counter_id, CmpOp::GE, expected_value);
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
// Thread-scope (idempotent)
// =============================================================================

__device__ int torchcomms_fence(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->fence();
}

// =============================================================================
// Window Properties
// Thread-scope (idempotent)
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

// =============================================================================
// NVLink Address Query
// Thread-scope (idempotent)
// =============================================================================

__device__ void* torchcomms_get_nvlink_address(void* win_ptr, int peer) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->get_nvlink_address(peer);
}

// =============================================================================
// Multimem Address Query
// Thread-scope (idempotent)
// =============================================================================

__device__ void* torchcomms_get_multimem_address(
    void* win_ptr,
    unsigned long long offset) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->get_multimem_address(static_cast<size_t>(offset));
}

} // extern "C"
