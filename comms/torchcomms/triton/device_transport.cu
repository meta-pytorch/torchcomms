// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Transport API - implementations for LLVM bitcode
//
// Extern C wrappers around comms::prims transport methods.
// All operations construct a block-scope ThreadGroup internally.

#include <cuda_runtime.h>

#include <cstddef>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"

using namespace comms::prims;

namespace {

__device__ __noinline__ void
unsupported_transport(const char* op, TransportType type, int peer) {
  printf(
      "torchcomms_transport_%s: unsupported transport type %d for peer %d\n",
      op,
      static_cast<int>(type),
      peer);
  __trap();
}

} // namespace

extern "C" {

// --- Signaling ---

__device__ int torchcomms_transport_signal(
    void* handle_ptr,
    int peer,
    int signal_id,
    int op,
    unsigned long long value) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  handle->get_nvl(peer).signal(
      group,
      static_cast<uint64_t>(signal_id),
      static_cast<SignalOp>(op),
      static_cast<uint64_t>(value));
  return 0;
}

__device__ int torchcomms_transport_wait_signal(
    void* handle_ptr,
    int peer,
    int signal_id,
    int op,
    unsigned long long value) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  handle->get_nvl(peer).wait_signal_until(
      group,
      static_cast<uint64_t>(signal_id),
      static_cast<CmpOp>(op),
      static_cast<uint64_t>(value));
  return 0;
}

// --- Send/Recv ---

__device__ __noinline__ int torchcomms_transport_send(
    void* handle_ptr,
    int peer,
    void* src_ptr,
    unsigned long long nbytes,
    unsigned long long max_signal_bytes) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  const TransportType type = handle->get_type(peer);
  switch (type) {
    case TransportType::P2P_NVL:
      handle->get_nvl(peer).send(
          group,
          src_ptr,
          static_cast<std::size_t>(nbytes),
          static_cast<std::size_t>(max_signal_bytes));
      break;
    case TransportType::P2P_IBGDA:
      handle->get_ibgda(peer).send(
          group,
          src_ptr,
          static_cast<std::size_t>(nbytes),
          static_cast<std::size_t>(max_signal_bytes));
      break;
    default:
      unsupported_transport("send", type, peer);
      break;
  }
  return 0;
}

__device__ __noinline__ int torchcomms_transport_recv(
    void* handle_ptr,
    int peer,
    void* dst_ptr,
    unsigned long long nbytes,
    unsigned long long max_signal_bytes) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  const TransportType type = handle->get_type(peer);
  switch (type) {
    case TransportType::P2P_NVL:
      handle->get_nvl(peer).recv(
          group,
          dst_ptr,
          static_cast<std::size_t>(nbytes),
          static_cast<std::size_t>(max_signal_bytes));
      break;
    case TransportType::P2P_IBGDA:
      handle->get_ibgda(peer).recv(
          group,
          dst_ptr,
          static_cast<std::size_t>(nbytes),
          static_cast<std::size_t>(max_signal_bytes));
      break;
    default:
      unsupported_transport("recv", type, peer);
      break;
  }
  return 0;
}

// --- Barrier ---

__device__ int
torchcomms_transport_barrier(void* handle_ptr, int peer, int barrier_id) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  handle->get_nvl(peer).barrier_sync(group, static_cast<uint64_t>(barrier_id));
  return 0;
}

// --- Properties ---

__device__ int torchcomms_transport_my_rank(void* handle_ptr) {
  return reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr)->myRank;
}

__device__ int torchcomms_transport_n_ranks(void* handle_ptr) {
  return reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr)->nRanks;
}

__device__ int torchcomms_transport_num_nvl_peers(void* handle_ptr) {
  return reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr)->numNvlPeers;
}

__device__ int torchcomms_transport_get_type(void* handle_ptr, int rank) {
  return static_cast<int>(
      reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr)->get_type(rank));
}

} // extern "C"
