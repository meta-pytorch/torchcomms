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
  virtual Result<size_t> send(std::span<const uint8_t> data) = 0;
  virtual Result<size_t> recv(std::vector<uint8_t>& data) = 0;

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
