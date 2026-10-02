// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/transport/TransportType.h"

/// Nullable-return annotation, equivalent to folly's FOLLY_NULLABLE. Defined
/// locally because comms/uniflow is a strict-OSS library and cannot depend on
/// folly (see OSS_DEPS_ALLOWLIST in comms/strict_oss_defs.bzl).
#ifndef UNIFLOW_NULLABLE
#if defined(__clang__)
#define UNIFLOW_NULLABLE _Nullable
#else
#define UNIFLOW_NULLABLE
#endif
#endif

namespace uniflow {

/// How a P2P segment is shared with peers. IPC payloads keep the legacy
/// layout without a mode byte so older peers still import them; every other
/// payload leads with its mode byte.
enum class P2pSharingMode : uint8_t {
  Ipc = 0, // HIP IPC handle (hipMalloc allocations)
  PosixFd = 1, // one POSIX fd per chunk of a VMM (hipMemCreate) allocation
};

/// Upper bound on VMM chunks per segment. At 28 bytes per chunk the payload
/// stays near 224 KiB, which covers 160 GiB of 20 MiB allocator chunks.
inline constexpr uint32_t kP2pMaxVmmChunks = 8192;

/// Where a pid is meaningful: the kernel boot and the PID namespace. A VMM
/// importer resolves the exporter's pid only when both match its own, so a
/// payload from another host or namespace never reaches pidfd_open.
struct P2pProcessScope {
  std::array<uint8_t, 16> bootId{}; // /proc/sys/kernel/random/boot_id
  uint64_t pidNamespace{0}; // st_ino of /proc/self/ns/pid

  bool operator==(const P2pProcessScope&) const = default;
};

/// Local registration handle for the AMD intra-node XGMI P2P transport.
///
/// Carries either a HIP IPC handle or, for VMM allocations, one exported POSIX
/// fd per physical chunk. The handle owns those fds and closes them on
/// destruction; importers duplicate them with pidfd_getfd while the segment is
/// registered. The backing allocation is owned by the Segment.
///
/// All vendor contact is confined to the CudaApi seam (neutral types only), so
/// this lives in a plain, non-hipified library.
class P2pRegistrationHandle : public RegistrationHandle {
 public:
  /// IPC wire format (byte-copy serialized, no mode byte).
  struct __attribute__((packed)) IpcPayload {
    int32_t ownerPid{0};
    uint64_t base{0}; // exporter allocation base address (same-pid fast path)
    uint64_t offset{0}; // segment offset within the allocation
    uint64_t size{0}; // segment length in bytes
    CudaApi::IpcMemHandle ipcHandle{}; // 64-byte opaque HIP IPC handle
  };
  static constexpr size_t kIpcSerializedSize =
      sizeof(int32_t) + 3 * sizeof(uint64_t) + CudaApi::kIpcMemHandleSize;
  static_assert(sizeof(IpcPayload) == kIpcSerializedSize);

  /// One VMM chunk as it appears on the wire. st_dev and st_ino identify the
  /// exported file host-wide, so an importer detects a reused fd number or pid.
  struct __attribute__((packed)) VmmChunk {
    int32_t fd{-1}; // exporter-side fd number, pulled with pidfd_getfd
    uint64_t dev{0}; // st_dev of the exported fd
    uint64_t inode{0}; // st_ino of the exported fd
    uint64_t size{0}; // chunk length in bytes
  };
  static_assert(sizeof(VmmChunk) == 28);

  /// VMM payload. The chunks are VA-contiguous in the exporter, in order.
  struct VmmPayload {
    P2pProcessScope scope; // where ownerPid is valid
    int32_t ownerPid{0};
    int32_t exporterDevice{-1};
    uint64_t offset{0}; // segment offset from the first chunk's base
    uint64_t size{0}; // segment length in bytes
    std::vector<VmmChunk> chunks;
  };

  using Payload = std::variant<IpcPayload, VmmPayload>;

  P2pRegistrationHandle(
      const CudaApi::IpcMemHandle& ipcHandle,
      int32_t ownerPid,
      uint64_t base,
      uint64_t offset,
      uint64_t size);

  /// Takes ownership of every chunk fd in @p payload.
  explicit P2pRegistrationHandle(VmmPayload payload);

  ~P2pRegistrationHandle() override;

  // Owns exported fds; always held behind a unique_ptr.
  P2pRegistrationHandle(const P2pRegistrationHandle&) = delete;
  P2pRegistrationHandle& operator=(const P2pRegistrationHandle&) = delete;
  P2pRegistrationHandle(P2pRegistrationHandle&&) = delete;
  P2pRegistrationHandle& operator=(P2pRegistrationHandle&&) = delete;

  TransportType transportType() const noexcept override {
    // Shared intra-node GPU-interconnect tier (XGMI on AMD, NVLink on NVIDIA).
    return TransportType::NVLink;
  }

  P2pSharingMode sharingMode() const noexcept;

  std::vector<uint8_t> serialize() const override;

  /// Parse a serialized payload: IPC by its exact length, anything else by its
  /// mode byte, validating the length, the chunk count bounds, and that the
  /// segment lies inside the chunks.
  static Result<Payload> deserialize(std::span<const uint8_t> bytes);

 private:
  Payload payload_;
};

/// Owner of a peer allocation mapped into this process; destruction tears the
/// mapping down, logging rather than throwing on failure.
class P2pMapping {
 public:
  virtual ~P2pMapping() = default;
};

/// Cross-process HIP IPC mapping, closed with ipcCloseMemHandle.
class P2pIpcMapping final : public P2pMapping {
 public:
  P2pIpcMapping(void* base, std::shared_ptr<CudaApi> cudaApi);
  ~P2pIpcMapping() override;

  P2pIpcMapping(const P2pIpcMapping&) = delete;
  P2pIpcMapping& operator=(const P2pIpcMapping&) = delete;
  P2pIpcMapping(P2pIpcMapping&&) = delete;
  P2pIpcMapping& operator=(P2pIpcMapping&&) = delete;

 private:
  void* base_{nullptr};
  std::shared_ptr<CudaApi> cudaApi_;
};

/// Remote registration handle: a peer's allocation mapped into this process.
class P2pRemoteRegistrationHandle : public RemoteRegistrationHandle {
 public:
  /// @param mappedBase device pointer to the mapping base in this process.
  /// @param offset     byte offset added to @p mappedBase for the segment ptr.
  /// @param size       segment length in bytes.
  /// @param mapping    owns the mapping behind @p mappedBase; nullptr when the
  ///                   pointer is borrowed (same-process IPC import).
  P2pRemoteRegistrationHandle(
      void* mappedBase,
      uint64_t offset,
      size_t size,
      std::unique_ptr<P2pMapping> mapping);

  // Always held behind a unique_ptr.
  P2pRemoteRegistrationHandle(const P2pRemoteRegistrationHandle&) = delete;
  P2pRemoteRegistrationHandle& operator=(const P2pRemoteRegistrationHandle&) =
      delete;
  P2pRemoteRegistrationHandle(P2pRemoteRegistrationHandle&&) = delete;
  P2pRemoteRegistrationHandle& operator=(P2pRemoteRegistrationHandle&&) =
      delete;

  TransportType transportType() const noexcept override {
    return TransportType::NVLink;
  }

  /// Usable device pointer for the segment (mappedBase + offset), or nullptr
  /// for a null base, so callers must null-check before doing pointer
  /// arithmetic on the result.
  void* UNIFLOW_NULLABLE mappedPtr() const noexcept;

  size_t mappedSize() const noexcept {
    return size_;
  }

 private:
  void* mappedBase_{nullptr};
  uint64_t offset_{0};
  size_t size_{0};
  std::unique_ptr<P2pMapping> mapping_;
};

} // namespace uniflow
