// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/algos/common/GpeKernelSync.h"

namespace ctran::transport::ib {
using ctran::algos::GpeKernelSync;

constexpr int kDeviceMaxPipelineDepth = 8;

// Per-chunk device-visible descriptor.
// Each staging buffer slot has its own sync + staging pointer.
struct DeviceChunkDesc {
  GpeKernelSync* sync{nullptr};
  char* stagingSlot{nullptr};
  size_t chunkSize{0};
};

// Device-side IB transport struct.
// Constructed on CPU by HostCbTransport, then cudaMemcpy'd to device.
// Kernel receives HostTransportDev* via kernel args.
struct HostTransportDev {
  DeviceChunkDesc sendChunks[kDeviceMaxPipelineDepth];
  DeviceChunkDesc recvChunks[kDeviceMaxPipelineDepth];
  int pipelineDepth{0};
  size_t chunkSize{0};
};

// Device-side template ops and processChunk<> are defined in the
// kernel .cu files that include GpeKernelSyncDev.cuh (CUDA-only context).
// This header only defines the host-visible structs.

} // namespace ctran::transport::ib
