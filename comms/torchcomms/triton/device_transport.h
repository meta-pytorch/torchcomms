// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Transport API - C-style declarations for LLVM bitcode
//
// Declares extern C wrappers for pipes transport operations
// (P2pNvlTransportDevice send/recv/signal/barrier).
// Compiled to LLVM bitcode with clang for linking with Triton kernels.
//
// Handle: void* device pointer to comms::pipes::MultiPeerDeviceHandle,
// allocated by PipesDeviceBackend::create_device_transport().
//
// All operations use comms::pipes::make_block_group() internally —
// all 128 Triton threads cooperate as one block group.
// Send/recv are __noinline__ to prevent memcpy_vectorized alloca inlining.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* TorchCommsTransportHandle;

// --- Data Transfer (block-cooperative, pipelined NVLink) ---

__device__ int torchcomms_transport_send(
    TorchCommsTransportHandle handle,
    int peer,
    void* src_ptr,
    unsigned long long nbytes);

__device__ int torchcomms_transport_recv(
    TorchCommsTransportHandle handle,
    int peer,
    void* dst_ptr,
    unsigned long long nbytes);

// --- Signaling ---

// op: 0=SIGNAL_SET, 1=SIGNAL_ADD
__device__ int torchcomms_transport_signal(
    TorchCommsTransportHandle handle,
    int peer,
    int signal_id,
    int op,
    unsigned long long value);

// op: 0=CMP_EQ, 1=CMP_GT, 2=CMP_LT, 3=CMP_GE, 4=CMP_LE, 5=CMP_NE
__device__ int torchcomms_transport_wait_signal(
    TorchCommsTransportHandle handle,
    int peer,
    int signal_id,
    int op,
    unsigned long long value);

// --- Barrier ---

__device__ int torchcomms_transport_barrier(
    TorchCommsTransportHandle handle,
    int peer,
    int barrier_id);

// --- Properties ---

__device__ int torchcomms_transport_my_rank(TorchCommsTransportHandle handle);
__device__ int torchcomms_transport_n_ranks(TorchCommsTransportHandle handle);
__device__ int torchcomms_transport_num_nvl_peers(
    TorchCommsTransportHandle handle);

// Returns TransportType as int: 0=SELF, 1=P2P_NVL, 2=P2P_IBGDA
__device__ int torchcomms_transport_get_type(
    TorchCommsTransportHandle handle,
    int rank);

#ifdef __cplusplus
}
#endif
