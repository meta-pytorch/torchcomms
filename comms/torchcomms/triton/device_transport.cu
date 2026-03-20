// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Transport API - implementations for LLVM bitcode
//
// Extern C wrappers around comms::pipes P2pNvlTransportDevice methods.
// All operations construct a block-scope ThreadGroup internally.

#include <cuda_runtime.h>

#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/ThreadGroup.cuh"

using namespace comms::pipes;

extern "C" {

// --- Data Transfer ---

__device__ __noinline__ int torchcomms_transport_send(
    void* handle_ptr,
    int peer,
    void* src_ptr,
    unsigned long long nbytes) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  handle->get_nvl(peer).send(group, src_ptr, static_cast<size_t>(nbytes));
  return 0;
}

__device__ __noinline__ int torchcomms_transport_recv(
    void* handle_ptr,
    int peer,
    void* dst_ptr,
    unsigned long long nbytes) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  handle->get_nvl(peer).recv(group, dst_ptr, static_cast<size_t>(nbytes));
  return 0;
}

// --- Signaling ---

__device__ int torchcomms_transport_signal(
    void* handle_ptr,
    int peer,
    int signal_id,
    int op,
    unsigned long long value) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  handle->get_nvl(peer).signal_threadgroup(
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
  handle->get_nvl(peer).wait_signal_until_threadgroup(
      group,
      static_cast<uint64_t>(signal_id),
      static_cast<CmpOp>(op),
      static_cast<uint64_t>(value));
  return 0;
}

// --- Barrier ---

__device__ int
torchcomms_transport_barrier(void* handle_ptr, int peer, int barrier_id) {
  auto* handle = reinterpret_cast<MultiPeerDeviceHandle*>(handle_ptr);
  auto group = make_block_group();
  handle->get_nvl(peer).barrier_sync_threadgroup(
      group, static_cast<uint64_t>(barrier_id));
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
