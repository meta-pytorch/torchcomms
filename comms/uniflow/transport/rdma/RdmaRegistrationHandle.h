// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "comms/uniflow/Segment.h"
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
/// The serialized payload contains a domain id and per-MR rkeys, which
/// the peer needs to perform one-sided RDMA operations on this memory.
class RdmaRegistrationHandle : public RegistrationHandle {
 public:
  /// Packed wire format for serialization.
  struct __attribute__((packed)) Header {
    uint64_t domainId{0}; // Factory instance key
    uint8_t numMrs{0}; // Number of MRs (one per NIC)
    // Followed by numMrs uint32_t rkeys.
  };

  static constexpr size_t kPayloadHeaderSize = sizeof(Header);

  RdmaRegistrationHandle(
      std::vector<ibv_mr*> mrs,
      std::shared_ptr<IbvApi> ibvApi,
      uint64_t domainId);

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

 private:
  std::vector<ibv_mr*> mrs_;
  std::shared_ptr<IbvApi> ibvApi_;
  uint64_t domainId_;
};

// ---------------------------------------------------------------------------
// RdmaRemoteRegistrationHandle
// ---------------------------------------------------------------------------

/// Remote registration handle for RDMA transport. Stores the remote peer's
/// per-MR rkeys (One MR per NIC) and domain id. The rkey for MR index i is
/// used in RDMA work requests posted on QPs belonging to that NIC.
class RdmaRemoteRegistrationHandle : public RemoteRegistrationHandle {
 public:
  RdmaRemoteRegistrationHandle(std::vector<uint32_t> rkeys, uint64_t domainId);

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

 private:
  std::vector<uint32_t> rkeys_;
  uint64_t domainId_;
};

} // namespace uniflow
