// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComm Device Backend Traits
//
// Defines backend trait types for compile-time polymorphism in device API.
// Each backend defines its communicator and window types.
//
// Current Backends:
//   - NCCLGinBackend: NCCL GIN for GPU-initiated networking
//
// Future Backends:
//   - NVSHMEMBackend: NVSHMEM for symmetric memory operations

#pragma once

#include <nccl.h> // @manual=//comms/ncclx:nccl
#include <nccl_device/impl/comm__types.h> // @manual=//comms/ncclx:nccl_device_api

namespace torchcomms::device {

// =============================================================================
// NCCLGinBackend - Backend traits for NCCL GIN
// =============================================================================
//
// Defines types for NCCL's GPU-Initiated Networking backend:
//   - Comm: ncclDevComm - Device communicator passed by value to kernels
//   - Window: ncclWindow_t - Window handle for RMA operations

struct NCCLGinBackend {
  using Comm = ncclDevComm;
  using Window = ncclWindow_t;
};

// =============================================================================
// Future Backends (placeholder)
// =============================================================================

// struct NVSHMEMBackend {
//   using Comm = nvshmem_team_t;
//   using Window = void*;  // NVSHMEM uses symmetric heap, no explicit window
// };

} // namespace torchcomms::device
