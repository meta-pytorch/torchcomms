// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/transport/TransportType.h"

namespace uniflow {

/// Memory sharing mode for NVLink transport serialization.
enum class MemSharingMode : uint8_t {
  Fabric = 0, // CUDA fabric handles (CU_MEM_HANDLE_TYPE_FABRIC), for GB200
              // multi-node NVLink (MNNVL)
  PosixFd = 1, // POSIX file descriptor handles
               // (CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) for H100
               // single-node cross-device transfers via pidfd IPC
};

// ---------------------------------------------------------------------------
// NVLinkFabricRegistrationHandle
// ---------------------------------------------------------------------------

/// Registration handle for NVLink fabric mode (GB200 MNNVL). Wraps a
/// CUmemGenericAllocationHandle and CUmemFabricHandle for cross-node memory
/// sharing via CUDA fabric handles.
class NVLinkFabricRegistrationHandle : public RegistrationHandle {
 public:
  /// Packed wire format for serialization (memcpy-based, like NVLinkTopology).
  struct __attribute__((packed)) Payload {
    MemSharingMode mode{MemSharingMode::Fabric};
    uint64_t allocationSize{0};
    CUmemFabricHandle fabricHandle{};
  };

  static constexpr size_t kSerializedSize =
      73; // 1 byte for mode + 8 bytes for size + 64 bytes for fabric handle
  static_assert(sizeof(Payload) == kSerializedSize);

  NVLinkFabricRegistrationHandle(
      uint64_t allocationSize,
      CUmemGenericAllocationHandle allocHandle,
      CUmemFabricHandle fabricHandle,
      std::shared_ptr<CudaDriverApi> cuda_driver_api);

  ~NVLinkFabricRegistrationHandle() override;

  TransportType transportType() const noexcept override {
    return TransportType::NVLink;
  }

  std::vector<uint8_t> serialize() const override;

  const CUmemFabricHandle& fabricHandle() const noexcept {
    return fabricHandle_;
  }

  CUmemGenericAllocationHandle allocHandle() const noexcept {
    return allocHandle_;
  }

 private:
  uint64_t allocationSize_;
  CUmemGenericAllocationHandle allocHandle_;
  CUmemFabricHandle fabricHandle_;
  std::shared_ptr<CudaDriverApi> cuda_driver_api_;
};

// ---------------------------------------------------------------------------
// NVLinkFdRegistrationHandle
// ---------------------------------------------------------------------------

/// Registration handle for NVLink FD mode (H100 single-node). Wraps a
/// CUmemGenericAllocationHandle and a POSIX file descriptor for cross-process
/// memory sharing via pidfd IPC.
class NVLinkFdRegistrationHandle : public RegistrationHandle {
 public:
  /// Packed wire format for serialization (memcpy-based, like NVLinkTopology).
  struct __attribute__((packed)) Payload {
    MemSharingMode mode{MemSharingMode::PosixFd};
    int32_t fd{-1};
    int32_t pid{0};
    uint64_t allocationSize{0};
  };

  static constexpr size_t kSerializedSize =
      17; // 1 byte for mode + 4 bytes for fd + 4 bytes for pid + 8 bytes for
          // allocation size
  static_assert(sizeof(Payload) == kSerializedSize);

  NVLinkFdRegistrationHandle(
      CUmemGenericAllocationHandle allocHandle,
      int exportedFd,
      pid_t ownerPid,
      size_t allocationSize,
      std::shared_ptr<CudaDriverApi> cuda_driver_api);

  ~NVLinkFdRegistrationHandle() override;

  TransportType transportType() const noexcept override {
    return TransportType::NVLink;
  }

  std::vector<uint8_t> serialize() const override;

  int exportedFd() const noexcept {
    return exportedFd_;
  }

  pid_t ownerPid() const noexcept {
    return ownerPid_;
  }

  size_t allocationSize() const noexcept {
    return allocationSize_;
  }

  CUmemGenericAllocationHandle allocHandle() const noexcept {
    return allocHandle_;
  }

 private:
  CUmemGenericAllocationHandle allocHandle_;
  int exportedFd_;
  pid_t ownerPid_;
  size_t allocationSize_;
  std::shared_ptr<CudaDriverApi> cuda_driver_api_;
};

// ---------------------------------------------------------------------------
// NVLinkRemoteRegistrationHandle
// ---------------------------------------------------------------------------

/// Remote registration handle for NVLink transport. Wraps a
/// CUmemGenericAllocationHandle imported from a remote peer via
/// cuMemImportFromShareableHandle, along with the VA mapping (reserve, map,
/// set access) needed for NVLink data transfers. Releasing the handle undoes
/// the import and VA mapping.
class NVLinkRemoteRegistrationHandle : public RemoteRegistrationHandle {
 public:
  NVLinkRemoteRegistrationHandle(
      CUmemGenericAllocationHandle allocHandle,
      CUdeviceptr mappedPtr,
      size_t mappedSize,
      std::shared_ptr<CudaDriverApi> cuda_driver_api);

  ~NVLinkRemoteRegistrationHandle() override;

  TransportType transportType() const noexcept override {
    return TransportType::NVLink;
  }

  void* mappedPtr() const {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(mappedPtr_);
  }

  size_t mappedSize() const {
    return mappedSize_;
  }

 private:
  CUmemGenericAllocationHandle allocHandle_;
  CUdeviceptr mappedPtr_;
  size_t mappedSize_;
  std::shared_ptr<CudaDriverApi> cuda_driver_api_;
};

} // namespace uniflow
