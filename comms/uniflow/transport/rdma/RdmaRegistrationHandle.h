// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/transport/TransportType.h"

namespace uniflow {

// ---------------------------------------------------------------------------
// RdmaRegistrationHandle
// ---------------------------------------------------------------------------

/// Registration handle for RDMA transport. Wraps one ibv_mr per NIC,
/// obtained from ibv_reg_mr. Each MR pins the same memory region but
/// is associated with a different protection domain.
///
/// The serialized payload contains a domain id, a registrationBase, and
/// per-MR rkeys, which the peer needs to perform one-sided RDMA operations
/// on this memory.
class RdmaRegistrationHandle : public RegistrationHandle {
 public:
  /// Packed wire format for serialization.
  struct __attribute__((packed)) Header {
    uint64_t domainId{0}; // Factory instance key
    uint64_t registrationBase{0}; // VA → wire-address offset
    uint8_t numMrs{0}; // Number of MRs (one per NIC)
    uint8_t isDeviceMemory{0}; // 1 if the region is device (VRAM) memory
    // Followed by numMrs uint32_t rkeys.
  };

  static constexpr size_t kPayloadHeaderSize = sizeof(Header);

  RdmaRegistrationHandle(
      std::vector<ibv_mr*> mrs,
      std::shared_ptr<IbvApi> ibvApi,
      uint64_t domainId,
      uint64_t registrationBase = 0,
      std::shared_ptr<DeviceAdapter> deviceAdapter = nullptr,
      int deviceId = -1,
      int bufferNumaNode = -1);

  ~RdmaRegistrationHandle() override;

  RdmaRegistrationHandle(const RdmaRegistrationHandle&) = delete;
  RdmaRegistrationHandle& operator=(const RdmaRegistrationHandle&) = delete;
  RdmaRegistrationHandle(RdmaRegistrationHandle&&) = delete;
  RdmaRegistrationHandle& operator=(RdmaRegistrationHandle&&) = delete;

  TransportType transportType() const noexcept override {
    return TransportType::RDMA;
  }

  std::vector<uint8_t> serialize() const override;

  /// Local key for the given MR index (one MR per NIC).
  uint32_t lkey(size_t idx) const noexcept {
    assert(idx < mrs_.size());
    return mrs_[idx]->lkey;
  }

  /// Remote key for the given MR index (one MR per NIC).
  uint32_t rkey(size_t idx) const noexcept {
    assert(idx < mrs_.size());
    return mrs_[idx]->rkey;
  }

  /// Number of MRs (one per NIC).
  size_t numMrs() const noexcept {
    return mrs_.size();
  }

  /// Factory key identifying which factory created this handle.
  uint64_t domainId() const noexcept {
    return domainId_;
  }

  /// NUMA node of the registered host buffer (-1 = unknown, device memory,
  /// or mixed locality across NUMA nodes). Detected at registration time by
  /// sampling pages after ibv_reg_mr has pinned (non-ODP) memory.
  int hostBufferNumaNode() const noexcept {
    return hostBufferNumaNode_;
  }

  /// Value to subtract from a VA to produce the address the NIC expects
  /// on the wire (SGE.addr / wr.rdma.remote_addr). Zero for ordinary
  /// single-buffer registrations; non-zero when the underlying MR covers
  /// a larger containing region.
  uint64_t registrationBase() const noexcept {
    return registrationBase_;
  }

  /// Translate a client-supplied pointer to the wire address the NIC
  /// expects in an SGE (`sge.addr`). Folds two platform-specific steps.
  uint64_t toWireAddr(const void* ptr) const;

 private:
  std::vector<ibv_mr*> mrs_;
  std::shared_ptr<IbvApi> ibvApi_;
  uint64_t domainId_;
  int hostBufferNumaNode_{-1};
  uint64_t registrationBase_{0};
  std::shared_ptr<DeviceAdapter> deviceAdapter_;
  int deviceId_{-1};
};

// ---------------------------------------------------------------------------
// RdmaRemoteRegistrationHandle
// ---------------------------------------------------------------------------

/// Remote registration handle for RDMA transport. Stores the remote peer's
/// per-MR rkeys (One MR per NIC) and domain id. The rkey for MR index i is
/// used in RDMA work requests posted on QPs belonging to that NIC.
class RdmaRemoteRegistrationHandle : public RemoteRegistrationHandle {
 public:
  RdmaRemoteRegistrationHandle(
      std::vector<uint32_t> rkeys,
      uint64_t domainId,
      uint64_t registrationBase = 0,
      bool isDeviceMemory = false,
      std::shared_ptr<DeviceAdapter> deviceAdapter = nullptr);

  ~RdmaRemoteRegistrationHandle() override = default;

  TransportType transportType() const noexcept override {
    return TransportType::RDMA;
  }

  /// Remote key for the given MR index (one MR per NIC).
  uint32_t rkey(size_t idx) const noexcept {
    assert(idx < rkeys_.size());
    return rkeys_[idx];
  }

  /// Number of remote NIC rkeys.
  size_t numMrs() const noexcept {
    return rkeys_.size();
  }

  /// Factory key identifying which factory created this handle.
  uint64_t domainId() const noexcept {
    return domainId_;
  }

  /// Mirror of the sender's `RdmaRegistrationHandle::registrationBase()` —
  /// subtract from remote VA to form `wr.rdma.remote_addr`.
  uint64_t registrationBase() const noexcept {
    return registrationBase_;
  }

  /// Whether the remote region is device (VRAM) memory, as advertised by the
  /// exporting peer. Gates device-pointer resolution in toWireAddr().
  bool isDeviceMemory() const noexcept {
    return isDeviceMemory_;
  }

  /// Translate a remote pointer to the wire address the NIC expects in
  /// `wr.rdma.remote_addr`. When the remote region is device memory, the
  /// pointer is resolved to a bare device address via the adapter; base is
  /// then subtracted.
  uint64_t toWireAddr(const void* ptr) const;

 private:
  std::vector<uint32_t> rkeys_;
  uint64_t domainId_;
  uint64_t registrationBase_{0};
  bool isDeviceMemory_{false};
  std::shared_ptr<DeviceAdapter> deviceAdapter_;
};

} // namespace uniflow
