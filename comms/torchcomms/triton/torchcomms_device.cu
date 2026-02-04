// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Triton Device API - nvcc-compiled implementations
//
// This file provides extern "C" wrappers around TorchCommDeviceWindow methods.
// Compiled with nvcc to support NCCLX GIN templates.
// Linked at runtime with Triton-compiled kernels via cuLink* APIs.
//
// Design:
// - All functions take void* handles (TorchCommsWindowHandle,
// TorchCommsBufferHandle)
// - Internally cast to TorchCommDeviceWindow<NCCLGinBackend>* and
// RegisteredBuffer*
// - 1:1 mapping with TorchCommDeviceWindow methods

#include <cuda_runtime.h>

#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

using namespace torchcomms::device;

using DeviceWindow = TorchCommDeviceWindow<NCCLGinBackend>;

extern "C" {

// =============================================================================
// RMA Operations
// =============================================================================

__device__ int torchcomms_put(
    void* win_ptr,
    unsigned long long dst_offset,
    void* src_buf_ptr,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long bytes,
    int signal_id,
    int counter_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  auto* src_buf = reinterpret_cast<RegisteredBuffer*>(src_buf_ptr);

  return win->put(
      static_cast<size_t>(dst_offset),
      *src_buf,
      static_cast<size_t>(src_offset),
      dst_rank,
      static_cast<size_t>(bytes),
      signal_id,
      counter_id);
}

// =============================================================================
// Signal Operations (Remote Notification)
// =============================================================================

__device__ int torchcomms_signal(
    void* win_ptr,
    int peer,
    int signal_id,
    unsigned long long value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->signal(peer, signal_id, SignalOp::ADD, value);
}

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
// =============================================================================

__device__ int torchcomms_fence(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->fence();
}

__device__ int torchcomms_flush(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->flush();
}

// =============================================================================
// Window Properties
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
