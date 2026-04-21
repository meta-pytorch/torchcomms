// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API - Pipes Backend Implementation (IBGDA + NVLink)
//
// Device-side implementations for TorchComms using Pipes:
//   - NVLink peers: direct vectorized memcpy via NVLink-mapped pointers
//   - IBGDA peers: RDMA Write via DOCA GPUNetIO (P2pIbgdaTransportDevice)
//   - SELF: NVLink local copy
//
// This file delegates to the comms::pipes::DeviceWindow public API for all
// signal, barrier, and put operations. The DeviceWindow handles transport
// dispatch (NVL vs IBGDA) internally.
//
// Header-only library — implementations are inline for template instantiation.
//
// IMPORTANT: Must only be included from .cu files compiled with nvcc.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#if defined(ENABLE_PIPES)

#ifndef __CUDACC__
#error "TorchCommDevicePipes.cuh must be compiled with nvcc."
#endif

#include <cuda_runtime.h>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/window/DeviceWindow.cuh"
#include "comms/torchcomms/device/pipes/PipesDeviceBackend.hpp"
#include "comms/torchcomms/device/pipes/TorchCommDevicePipesTypes.hpp"

namespace torchcomms::device {

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Map TorchComms SignalOp to Pipes SignalOp.
__device__ inline comms::pipes::SignalOp to_pipes_signal_op(SignalOp op) {
  return (op == SignalOp::ADD) ? comms::pipes::SignalOp::SIGNAL_ADD
                               : comms::pipes::SignalOp::SIGNAL_SET;
}

// Map TorchComms CmpOp to Pipes CmpOp.
__device__ inline comms::pipes::CmpOp to_pipes_cmp_op(CmpOp cmp) {
  switch (cmp) {
    case CmpOp::EQ:
      return comms::pipes::CmpOp::CMP_EQ;
    case CmpOp::NE:
      return comms::pipes::CmpOp::CMP_NE;
    case CmpOp::LT:
      return comms::pipes::CmpOp::CMP_LT;
    case CmpOp::LE:
      return comms::pipes::CmpOp::CMP_LE;
    case CmpOp::GT:
      return comms::pipes::CmpOp::CMP_GT;
    case CmpOp::GE:
      return comms::pipes::CmpOp::CMP_GE;
  }
  return comms::pipes::CmpOp::CMP_GE;
}

// Build a pipes::ThreadGroup for the given CoopScope.
__device__ inline comms::pipes::ThreadGroup make_pipes_thread_group(
    CoopScope scope) {
  switch (scope) {
    case CoopScope::WARP:
      return comms::pipes::make_warp_group();
    case CoopScope::BLOCK:
      return comms::pipes::make_block_group();
    case CoopScope::THREAD:
      return comms::pipes::make_thread_solo();
  }
  __builtin_unreachable();
}

} // namespace detail

// =============================================================================
// TorchCommDeviceWindow<PipesDeviceBackend> Signal Operations
// =============================================================================
//
// Delegates to comms::pipes::DeviceWindow::signal_peer() which handles
// NVL vs IBGDA transport dispatch internally. Signal slots are indexed
// by (peer, signal_id) pairs.

template <>
__device__ inline int TorchCommDeviceWindow<PipesDeviceBackend>::signal(
    int peer,
    int signal_id,
    SignalOp op,
    uint64_t value,
    CoopScope scope) {
  auto& win = *window_;
  auto pipes_op = detail::to_pipes_signal_op(op);

  if (scope == CoopScope::THREAD) {
    win.signal_peer(peer, signal_id, pipes_op, value);
  } else {
    auto group = detail::make_pipes_thread_group(scope);
    win.signal_peer(group, peer, signal_id, pipes_op, value);
  }
  return 0;
}

// =============================================================================
// TorchCommDeviceWindow<PipesDeviceBackend> RMA Operations
// =============================================================================

template <>
__device__ inline int TorchCommDeviceWindow<PipesDeviceBackend>::put(
    size_t dst_offset,
    const torch::comms::RegisteredBuffer& src_buf,
    size_t src_offset,
    int dst_rank,
    size_t bytes,
    int signal_id,
    int counter_id,
    CoopScope scope) {
  auto& win = *window_;
  auto group = detail::make_pipes_thread_group(scope);

  // Build Pipes LocalBufferRegistration from RegisteredBuffer.
  // Pipes uses lkey (IBGDA local key); GIN uses backend_window.
  ::comms::pipes::LocalBufferRegistration pipes_src{
      src_buf.base_ptr,
      src_buf.size,
      ::comms::pipes::NetworkLKey{src_buf.lkey}};

  bool has_signal = signal_id >= 0;
  bool has_counter = counter_id >= 0;

  if (has_signal && has_counter) {
    win.put_signal_counter(
        group,
        dst_rank,
        dst_offset,
        pipes_src,
        src_offset,
        bytes,
        signal_id,
        /*signalVal=*/1,
        counter_id,
        /*counterVal=*/1);
  } else if (has_signal) {
    win.put_signal(
        group,
        dst_rank,
        dst_offset,
        pipes_src,
        src_offset,
        bytes,
        signal_id,
        /*signalVal=*/1);
  } else if (has_counter) {
    win.put_counter(
        group,
        dst_rank,
        dst_offset,
        pipes_src,
        src_offset,
        bytes,
        counter_id,
        /*counterVal=*/1);
  } else {
    win.put(group, dst_rank, dst_offset, pipes_src, src_offset, bytes);
  }

  return 0;
}

// =============================================================================
// TorchCommDeviceWindow<PipesDeviceBackend> Wait Signal Operations
// =============================================================================

template <>
__device__ inline int TorchCommDeviceWindow<PipesDeviceBackend>::wait_signal(
    int signal_id,
    CmpOp cmp,
    uint64_t value,
    CoopScope scope) {
  auto& win = *window_;
  auto pipes_cmp = detail::to_pipes_cmp_op(cmp);
  auto group = detail::make_pipes_thread_group(scope);
  win.wait_signal(group, signal_id, pipes_cmp, value);
  return 0;
}

template <>
__device__ inline int
TorchCommDeviceWindow<PipesDeviceBackend>::wait_signal_from(
    int peer,
    int signal_id,
    CmpOp cmp,
    uint64_t value,
    CoopScope scope) {
  auto& win = *window_;
  auto pipes_cmp = detail::to_pipes_cmp_op(cmp);
  auto group = detail::make_pipes_thread_group(scope);
  win.wait_signal_from(group, peer, signal_id, pipes_cmp, value);
  return 0;
}

template <>
__device__ inline uint64_t
TorchCommDeviceWindow<PipesDeviceBackend>::read_signal(int signal_id) const {
  // window_ is DeviceWindow* (not const), so dereference works in const
  // methods. DeviceWindow::read_signal() only reads inbox values despite being
  // non-const.
  auto& win = *window_;
  return win.read_signal(signal_id);
}

template <>
__device__ inline void TorchCommDeviceWindow<PipesDeviceBackend>::reset_signal(
    int signal_id,
    CoopScope scope) {
  // Pipes DeviceWindow does not support device-side signal reset.
  // Use monotonically increasing signal values, or reset from host side.
  (void)signal_id;
  (void)scope;
  __trap();
}

// =============================================================================
// TorchCommDeviceWindow<PipesDeviceBackend> Counter Operations
// =============================================================================
// Counters use companion RDMA QP loopback atomic signaling for NIC completion
// tracking. read_counter/reset_counter must precede wait_counter (which calls
// read_counter).

template <>
__device__ inline uint64_t
TorchCommDeviceWindow<PipesDeviceBackend>::read_counter(int counter_id) const {
  // Sum the counter across all peers (aggregate model).
  auto& win = *window_;
  int nPeers = win.num_peers();
  uint64_t total = 0;
  for (int peer_index = 0; peer_index < nPeers; ++peer_index) {
    int r = win.peer_index_to_rank(peer_index);
    total += win.read_counter(r, counter_id);
  }
  return total;
}

template <>
__device__ inline void TorchCommDeviceWindow<PipesDeviceBackend>::reset_counter(
    int counter_id,
    CoopScope scope) {
  auto group = detail::make_pipes_thread_group(scope);
  group.sync();

  if (group.is_leader()) {
    auto& win = *window_;
    int nPeers = win.num_peers();
    for (int peer_index = 0; peer_index < nPeers; ++peer_index) {
      int r = win.peer_index_to_rank(peer_index);
      win.reset_counter(r, counter_id);
    }
  }
}

template <>
__device__ inline int TorchCommDeviceWindow<PipesDeviceBackend>::wait_counter(
    int counter_id,
    CmpOp cmp,
    uint64_t value,
    CoopScope scope) {
  auto group = detail::make_pipes_thread_group(scope);

  if (group.is_leader()) {
    while (true) {
      uint64_t total = read_counter(counter_id);
      bool satisfied = false;
      switch (cmp) {
        case CmpOp::EQ:
          satisfied = (total == value);
          break;
        case CmpOp::NE:
          satisfied = (total != value);
          break;
        case CmpOp::GE:
          satisfied = (total >= value);
          break;
        case CmpOp::GT:
          satisfied = (total > value);
          break;
        case CmpOp::LE:
          satisfied = (total <= value);
          break;
        case CmpOp::LT:
          satisfied = (total < value);
          break;
      }
      if (satisfied) {
        break;
      }
    }
  }

  group.sync();
  return 0;
}

// =============================================================================
// TorchCommDeviceWindow<PipesDeviceBackend> Synchronization Operations
// =============================================================================

template <>
__device__ inline int TorchCommDeviceWindow<PipesDeviceBackend>::fence() {
  // Compiler barrier: prevents reordering of put() calls across this point.
  asm volatile("" ::: "memory");
  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<PipesDeviceBackend>::flush(
    CoopScope scope) {
  // flush() = local completion: source buffers are safe to reuse.
  //
  // NVLink: puts are inline stores. group.sync() ensures all threads have
  // completed their memcpy operations. No async NIC DMA to wait for.
  //
  // IBGDA: fence() drains each QP by posting a NOP WQE and waiting for
  // completion, ensuring all prior puts have been handed off to the NIC.
  auto group = detail::make_pipes_thread_group(scope);
  group.sync();

  auto& win = *window_;
  int nPeers = win.num_peers();
  for (int peer_index = 0; peer_index < nPeers; ++peer_index) {
    int r = win.peer_index_to_rank(peer_index);
    if (win.get_type(r) == comms::pipes::TransportType::P2P_IBGDA) {
      win.get_ibgda(r).fence(group);
    }
  }

  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<PipesDeviceBackend>::barrier(
    int barrier_id,
    CoopScope scope) {
  // Delegate to DeviceWindow::barrier() which handles the full
  // arrive + wait protocol across NVL and IBGDA peers.
  auto& win = *window_;
  auto group = detail::make_pipes_thread_group(scope);
  win.barrier(group, barrier_id);
  return 0;
}

// =============================================================================
// TorchCommDeviceWindow<PipesDeviceBackend> NVLink Address Query
// =============================================================================

template <>
__device__ inline void*
TorchCommDeviceWindow<PipesDeviceBackend>::get_nvlink_address(int peer) {
  return window_->get_nvlink_address(peer);
}

// =============================================================================
// TorchCommDeviceWindow<PipesDeviceBackend> Multimem Address Query
// =============================================================================

template <>
__device__ inline void*
TorchCommDeviceWindow<PipesDeviceBackend>::get_multimem_address(
    size_t /*offset*/) {
  // Multimem (NVLS multicast) is an NCCL/GIN feature, not available via Pipes.
  return nullptr;
}

} // namespace torchcomms::device

#endif // ENABLE_PIPES
