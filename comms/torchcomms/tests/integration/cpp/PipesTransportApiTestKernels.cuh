// Copyright (c) Meta Platforms, Inc. and affiliates.
// CUDA kernel declarations for PipesTransportApiTest
//
// This header provides function declarations for launching warp-level
// NVL send/recv kernels via the pipes MultiPeerDeviceHandle.
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

} // namespace torchcomms::device::test
