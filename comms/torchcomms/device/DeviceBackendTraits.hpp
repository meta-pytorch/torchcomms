// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComm Device Backend Traits
//
// Defines backend trait types for compile-time polymorphism in device API.
// Each backend defines its communicator and window types, plus static
// create_device_window/destroy_device_window methods for device state
// management.
//
// Current Backends:
//   - NCCLGinBackend: NCCL GIN for GPU-initiated networking
//
// Future Backends:
//   - NVSHMEMBackend: NVSHMEM for symmetric memory operations

#pragma once

#include <nccl.h> // @manual=//comms/ncclx:nccl
#include <nccl_device/impl/comm__types.h> // @manual=//comms/ncclx:nccl_device_api

// Forward declarations
namespace torch::comms {
class NcclxApi;
} // namespace torch::comms

namespace torchcomms::device {

// Forward declarations
struct DeviceBackendConfig;
template <typename Backend>
class TorchCommDeviceWindow;

// =============================================================================
// NCCLGinBackend - Backend traits for NCCL GIN
// =============================================================================
//
// Defines types and static methods for NCCL's GPU-Initiated Networking backend:
//   - Comm: ncclDevComm - Device communicator passed by value to kernels
//   - Window: ncclWindow_t - Window handle for RMA operations
//   - create_device_window(): Creates fully initialized TorchCommDeviceWindow
//   - destroy_device_window(): Destroys device window including ncclDevComm
//
// Stateless design - all state is passed as parameters, no instance needed.

struct NCCLGinBackend {
  using Comm = ncclDevComm;
  using Window = ncclWindow_t;

  // Create fully initialized device window struct.
  // Creates ncclDevComm internally and populates all window fields.
  // Returns TorchCommDeviceWindow by value (no heap allocation).
  static TorchCommDeviceWindow<NCCLGinBackend> create_device_window(
      ncclComm_t nccl_comm,
      torch::comms::NcclxApi* nccl_api,
      const DeviceBackendConfig& config,
      Window orig_window,
      void* base,
      size_t size);

  // Destroy device window, including the ncclDevComm.
  // Logs errors but doesn't throw (safe for cleanup/destructor use).
  static void destroy_device_window(
      ncclComm_t nccl_comm,
      torch::comms::NcclxApi* nccl_api,
      TorchCommDeviceWindow<NCCLGinBackend>& device_window);
};

// =============================================================================
// DeviceBackendConfig - Configuration for device state creation
// =============================================================================

struct DeviceBackendConfig {
  int signal_count{0};
  int counter_count{0};
  int barrier_count{1};
  int comm_rank{0};
  int comm_size{1};
};

// =============================================================================
// Future Backends (placeholder)
// =============================================================================

// struct NVSHMEMBackend {
//   using Comm = nvshmem_team_t;
//   using Window = void*;  // NVSHMEM uses symmetric heap, no explicit window
//
//   static TorchCommDeviceWindow<NVSHMEMBackend> create_device_window(
//       nvshmem_team_t team,
//       const DeviceBackendConfig& config,
//       void* base,
//       size_t size);
//   static void destroy_device_window(
//       TorchCommDeviceWindow<NVSHMEMBackend>& device_window);
// };

} // namespace torchcomms::device
