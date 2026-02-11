// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComm Device Backend Traits
//
// Defines backend trait types for compile-time polymorphism in device API.
// Each backend defines its communicator and window types, plus static
// create_device_window methods for device state management.
//
// Current Backends:
//   - NCCLDeviceBackend: Unified NCCL backend (GIN + LSA)
//
// Future Backends:
//   - NVSHMEMBackend: NVSHMEM for symmetric memory operations

#pragma once

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <memory>

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
// NCCLDeviceBackend - Unified NCCL backend (GIN + LSA)
// =============================================================================
//
// Defines types and static methods for the unified NCCL device backend:
//   - Comm: ncclDevComm - Device communicator passed by value to kernels
//   - Window: ncclWindow_t - Window handle for RMA operations
//   - Ptr: unique_ptr with custom deleter for device window ownership
//   - create_device_window(): Creates fully initialized TorchCommDeviceWindow
//
// The device window is allocated in DEVICE memory (via cudaMalloc) so that
// the pointer can be passed to GPU kernels (both C++ CUDA and Triton).
//
// Ownership is managed via Ptr (unique_ptr with custom deleter).
// The custom deleter stores dev_comm for cleanup and calls cudaFree.
// The caller must call ncclDevCommDestroy before destroying the Ptr.

struct NCCLDeviceBackend {
  using Comm = ncclDevComm;
  using Window = ncclWindow_t;

  // =========================================================================
  // DeviceWindowDeleter - Custom deleter for device window cleanup
  // =========================================================================
  //
  // This deleter is used with std::unique_ptr for TorchCommDeviceWindow.
  // It stores the dev_comm on the host side to avoid needing cudaMemcpy
  // from device memory during destruction.
  //
  // The deleter only calls cudaFree. The caller is responsible for calling
  // ncclDevCommDestroy before the unique_ptr is destroyed. Access the
  // dev_comm via unique_ptr::get_deleter().dev_comm.
  struct DeviceWindowDeleter {
    ncclComm_t nccl_comm{nullptr};
    torch::comms::NcclxApi* nccl_api{nullptr};
    Comm dev_comm{};

    DeviceWindowDeleter() = default;
    DeviceWindowDeleter(
        ncclComm_t comm,
        torch::comms::NcclxApi* api,
        Comm dev_comm_val)
        : nccl_comm(comm), nccl_api(api), dev_comm(dev_comm_val) {}

    void operator()(TorchCommDeviceWindow<NCCLDeviceBackend>* ptr) const;
  };

  // Type alias for device window unique_ptr with custom deleter
  using Ptr = std::
      unique_ptr<TorchCommDeviceWindow<NCCLDeviceBackend>, DeviceWindowDeleter>;

  // Create fully initialized device window struct in DEVICE memory.
  // Creates ncclDevComm internally and populates all window fields.
  // Returns Ptr (unique_ptr with custom deleter) for ownership.
  //
  // The returned pointer is a DEVICE pointer allocated via cudaMalloc.
  // It can be passed directly to CUDA kernels or Triton via void* cast.
  //
  // The custom deleter stores the dev_comm for access during cleanup.
  // The caller must call ncclDevCommDestroy using get_deleter().dev_comm
  // before destroying the Ptr (the deleter only calls cudaFree).
  //
  // Parameters:
  //   - nccl_comm: Host NCCL communicator (must not be null)
  //   - nccl_api: NCCL API abstraction (must not be null)
  //   - config: Device backend configuration
  //   - host_window: Host-side NCCL window handle
  //   - base: Window base pointer (can be null only if size is 0)
  //   - size: Window size in bytes
  static Ptr create_device_window(
      ncclComm_t nccl_comm,
      torch::comms::NcclxApi* nccl_api,
      const DeviceBackendConfig& config,
      Window host_window,
      void* base,
      size_t size);
};

// Type alias for backward compatibility
template <typename Backend>
using DeviceWindowPtr = typename Backend::Ptr;

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
//   struct DeviceWindowDeleter {
//     // NVSHMEM-specific cleanup state
//     bool symmetric_heap_registered{false};
//
//     void operator()(TorchCommDeviceWindow<NVSHMEMBackend>* ptr) const;
//   };
//
//   static std::unique_ptr<TorchCommDeviceWindow<NVSHMEMBackend>,
//   DeviceWindowDeleter> create_device_window(
//       nvshmem_team_t team,
//       const DeviceBackendConfig& config,
//       void* base,
//       size_t size);
// };

// Backward compatibility alias — will be removed in the triton update commit.
using NCCLGinBackend = NCCLDeviceBackend;

} // namespace torchcomms::device
