// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/tcpstore/Backoff.h"

namespace ncclx::tcpstore::detail {

class SocketOptions {
 public:
  SocketOptions& prefer_ipv6(bool value) noexcept {
    prefer_ipv6_ = value;

    return *this;
  }

  bool prefer_ipv6() const noexcept {
    return prefer_ipv6_;
  }

  SocketOptions& connect_timeout(std::chrono::milliseconds value) noexcept {
    connect_timeout_ = value;

    return *this;
  }

  std::chrono::milliseconds connect_timeout() const noexcept {
    return connect_timeout_;
  }

  // Sets the backoff policy to use for socket connect ops.
  SocketOptions& connect_backoff(std::shared_ptr<Backoff> value) noexcept {
    connect_backoff_ = std::move(value);

    return *this;
  }

  const std::shared_ptr<Backoff>& connect_backoff() const noexcept {
    return connect_backoff_;
  }

 private:
  bool prefer_ipv6_ = true;
  std::chrono::milliseconds connect_timeout_{
      std::chrono::seconds{NCCL_TCPSTORE_CONNECT_TIMEOUT}};
  std::shared_ptr<Backoff> connect_backoff_{
      std::make_shared<ncclx::tcpstore::FixedBackoff>(
          std::chrono::milliseconds(100))};
};

class SocketImpl;

class Socket {
 public:
  // This function initializes the underlying socket library and must be called
  // before any other socket function.
  static void initialize();

  static Socket connect(
      const std::string& host,
      std::uint16_t port,
      const SocketOptions& opts = {});

  Socket() noexcept = default;

  Socket(const Socket& other) = delete;

  Socket& operator=(const Socket& other) = delete;

  Socket(Socket&& other) noexcept;

  Socket& operator=(Socket&& other) noexcept;

  ~Socket();

  int handle() const noexcept;

  bool waitForInput(std::chrono::milliseconds timeout);

 private:
  explicit Socket(std::unique_ptr<SocketImpl>&& impl) noexcept;

  std::unique_ptr<SocketImpl> impl_;
};

} // namespace ncclx::tcpstore::detail
