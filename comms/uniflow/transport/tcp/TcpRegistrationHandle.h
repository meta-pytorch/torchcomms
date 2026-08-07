// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/transport/TransportType.h"

namespace uniflow {

class TcpRegistrationHandle : public RegistrationHandle {
 public:
  struct __attribute__((packed)) Header {
    uint64_t segId{0};
    uint64_t base{0};
    uint64_t len{0};
  };

  TcpRegistrationHandle(
      uint64_t segId,
      uint64_t base,
      uint64_t len,
      std::function<void()> onDestroy = {});

  // Deregisters segId_ from the owning registry (RegistrationHandle contract).
  ~TcpRegistrationHandle() override;

  TcpRegistrationHandle(const TcpRegistrationHandle&) = delete;
  TcpRegistrationHandle& operator=(const TcpRegistrationHandle&) = delete;
  TcpRegistrationHandle(TcpRegistrationHandle&&) = delete;
  TcpRegistrationHandle& operator=(TcpRegistrationHandle&&) = delete;

  TransportType transportType() const noexcept override {
    return TransportType::TCP;
  }

  std::vector<uint8_t> serialize() const override;

  uint64_t segId() const noexcept {
    return segId_;
  }

  uint64_t base() const noexcept {
    return base_;
  }

  uint64_t len() const noexcept {
    return len_;
  }

 private:
  uint64_t segId_{0};
  uint64_t base_{0};
  uint64_t len_{0};
  std::function<void()> onDestroy_;
};

class TcpRemoteRegistrationHandle : public RemoteRegistrationHandle {
 public:
  TcpRemoteRegistrationHandle(uint64_t segId, uint64_t base, uint64_t len);

  static Result<std::unique_ptr<TcpRemoteRegistrationHandle>> deserialize(
      size_t segmentLength,
      std::span<const uint8_t> payload);

  TransportType transportType() const noexcept override {
    return TransportType::TCP;
  }

  uint64_t segId() const noexcept {
    return segId_;
  }

  uint64_t base() const noexcept {
    return base_;
  }

  uint64_t len() const noexcept {
    return len_;
  }

 private:
  uint64_t segId_{0};
  uint64_t base_{0};
  uint64_t len_{0};
};

} // namespace uniflow
