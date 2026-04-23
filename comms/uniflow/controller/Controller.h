// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <future>
#include <memory>
#include <span>
#include <vector>
#include "comms/uniflow/Result.h"

namespace uniflow::controller {

class Conn {
 public:
  virtual ~Conn() = default;

  /// The data is owned by the caller, not the connection. The caller
  /// must ensure the buffer outlives the returned future.
  [[nodiscard]] virtual std::future<Result<size_t>> send(
      std::span<const uint8_t> data) = 0;

  /// Allocating recv: reads length prefix, allocates buffer, fills it.
  [[nodiscard]] virtual std::future<Result<size_t>> recv(
      std::vector<uint8_t>& data) = 0;

  /// Zero-copy recv: reads length prefix, fills caller's pre-allocated buffer.
  /// Error if payload exceeds buf.size(). The buffer is owned by the
  /// caller, not the connection. The caller must ensure the buffer
  /// outlives the returned future.
  [[nodiscard]] virtual std::future<Result<size_t>> recv(
      std::span<uint8_t> buf) = 0;

  /// Interrupt any blocked recv(). After close(), recv() must return an error.
  /// Used by Connection to stop its reader thread during shutdown.
  virtual void close() {}
};

class Server {
 public:
  virtual ~Server() = default;

  virtual Status init() = 0;

  virtual const std::string& getId() const = 0;

  [[nodiscard]] virtual std::future<std::unique_ptr<Conn>> accept() = 0;
};

class Client {
 public:
  Client() = default;
  virtual ~Client() = default;

  [[nodiscard]] virtual std::future<std::unique_ptr<Conn>> connect(
      std::string id) = 0;
};

} // namespace uniflow::controller
