// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/transport/TransportType.h"

namespace uniflow {

/// Local registration handle for the AMD intra-node XGMI P2P transport.
///
/// Wraps a HIP IPC handle (CudaApi::IpcMemHandle) plus the metadata a peer
/// needs to map and address the exporter's device allocation. The exported
/// handle is pure bytes and the backing allocation is owned by the Segment, so
/// this handle holds no GPU resource and its destructor is trivial.
///
/// All vendor contact is confined to the CudaApi seam (neutral types only), so
/// this lives in a plain, non-hipified library.
class P2pRegistrationHandle : public RegistrationHandle {
 public:
  /// Packed wire format (byte-copy serialized, like the NVLink handles).
  struct __attribute__((packed)) Payload {
    int32_t ownerPid{0};
    uint64_t base{0}; // exporter allocation base address (same-pid fast path)
    uint64_t offset{0}; // segment offset within the allocation
    uint64_t size{0}; // segment length in bytes
    CudaApi::IpcMemHandle ipcHandle{}; // 64-byte opaque HIP IPC handle
  };
  static constexpr size_t kSerializedSize =
      sizeof(int32_t) + 3 * sizeof(uint64_t) + CudaApi::kIpcMemHandleSize;
  static_assert(sizeof(Payload) == kSerializedSize);

  P2pRegistrationHandle(
      const CudaApi::IpcMemHandle& ipcHandle,
      int32_t ownerPid,
      uint64_t base,
      uint64_t offset,
      uint64_t size);

  ~P2pRegistrationHandle() override = default;

  TransportType transportType() const noexcept override {
    // Shared intra-node GPU-interconnect tier (XGMI on AMD, NVLink on NVIDIA).
    return TransportType::NVLink;
  }

  std::vector<uint8_t> serialize() const override;

  /// Parse a serialized payload. Errors if @p bytes has the wrong length.
  static Result<Payload> deserialize(std::span<const uint8_t> bytes);

 private:
  Payload payload_{};
};

/// Remote registration handle: a peer's allocation mapped into this process.
///
/// Cross-process imports come from hipIpcOpenMemHandle and are closed on
/// destruction; same-process imports reuse the exporter's pointer directly (no
/// IPC open, nothing to close).
class P2pRemoteRegistrationHandle : public RemoteRegistrationHandle {
 public:
  /// @param mappedBase device pointer to the allocation base in this process.
  /// @param offset     byte offset added to @p mappedBase for the segment ptr.
  /// @param size       segment length in bytes.
  /// @param ownedByIpc true when @p mappedBase came from ipcOpenMemHandle and
  ///                   must be closed via ipcCloseMemHandle on destruction.
  P2pRemoteRegistrationHandle(
      void* mappedBase,
      uint64_t offset,
      size_t size,
      bool ownedByIpc,
      std::shared_ptr<CudaApi> cudaApi);

  ~P2pRemoteRegistrationHandle() override;

  // Owns the IPC mapping when ownedByIpc_; non-copyable. Moves transfer the
  // mapping and null out the source so only one instance ever closes it.
  P2pRemoteRegistrationHandle(const P2pRemoteRegistrationHandle&) = delete;
  P2pRemoteRegistrationHandle& operator=(const P2pRemoteRegistrationHandle&) =
      delete;
  P2pRemoteRegistrationHandle(P2pRemoteRegistrationHandle&& other) noexcept;
  P2pRemoteRegistrationHandle& operator=(
      P2pRemoteRegistrationHandle&& other) noexcept;

  TransportType transportType() const noexcept override {
    return TransportType::NVLink;
  }

  /// Usable device pointer for the segment (mappedBase + offset). Returns
  /// nullptr for a moved-from / invalid handle.
  void* mappedPtr() const noexcept;

  size_t mappedSize() const noexcept {
    return size_;
  }

 private:
  // Closes the IPC mapping if this handle owns one, logging (not throwing) on
  // failure, then nulls the owned state. Safe to call from the destructor and
  // move-assignment (both noexcept).
  void closeMapping() noexcept;

  void* mappedBase_{nullptr};
  uint64_t offset_{0};
  size_t size_{0};
  bool ownedByIpc_{false};
  std::shared_ptr<CudaApi> cudaApi_;
};

} // namespace uniflow
