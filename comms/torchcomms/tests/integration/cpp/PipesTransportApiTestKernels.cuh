// Copyright (c) Meta Platforms, Inc. and affiliates.
// CUDA kernel declarations for PipesTransportApiTest
//
// This header provides function declarations for launching warp-level
// NVL transport kernels via the pipes MultiPeerDeviceHandle.
//
// The full device implementations are only in the .cu file which is
// compiled by nvcc.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_runtime.h>
#include "comms/pipes/MultiPeerDeviceHandle.cuh"

namespace torchcomms::device::test {

// Launch a warp-level NVL send kernel.
void launchNvlSendKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* src_d,
    size_t nbytes,
    cudaStream_t stream = nullptr);

// Launch a warp-level NVL recv kernel.
void launchNvlRecvKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* dst_d,
    size_t nbytes,
    cudaStream_t stream = nullptr);

// Launch signal kernel: signals peer (ADD 1 on signal_id 0) and waits for
// peer's signal (GE 1 on signal_id 0). Both ranks must call.
void launchNvlSignalKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    cudaStream_t stream = nullptr);

// Launch LL128 send kernel (warp-only, 16-byte aligned).
void launchNvlLl128SendKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* src_d,
    size_t nbytes,
    cudaStream_t stream = nullptr);

// Launch LL128 recv kernel (warp-only, 16-byte aligned).
void launchNvlLl128RecvKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    void* dst_d,
    size_t nbytes,
    cudaStream_t stream = nullptr);

// Host-side check: returns non-zero if LL128 is available for the given peer.
// Launches a tiny kernel to query get_ll128_buffer_num_packets() on device.
int checkLl128Available(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peerRank,
    cudaStream_t stream = nullptr);

} // namespace torchcomms::device::test
