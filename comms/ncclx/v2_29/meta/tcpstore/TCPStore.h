// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <folly/Synchronized.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ncclx::tcpstore {

static constexpr std::chrono::milliseconds kDefaultTimeout =
    std::chrono::seconds(300);
static constexpr std::chrono::milliseconds kNoTimeout =
    std::chrono::milliseconds::zero();

namespace detail {

class TCPServer;
class TCPClient;

struct SocketAddress {
  std::string host{};
  std::uint16_t port{};
};

// Magic number for client validation.
static const uint32_t validationMagicNumber = 0x3C85F7CE;

enum class QueryType : uint8_t {
  VALIDATE,
  SET,
  COMPARE_SET,
  GET,
  ADD,
  CHECK,
  WAIT,
  GETNUMKEYS,
  DELETE_KEY,
  APPEND,
  MULTI_GET,
  MULTI_SET,
  CANCEL_WAIT,
  PING
};

enum class CheckResponseType : uint8_t { READY, NOT_READY };

enum class WaitResponseType : uint8_t { STOP_WAITING, WAIT_CANCELED };

} // namespace detail

struct TCPStoreOptions {
  static constexpr std::uint16_t kDefaultPort = 29500;

  std::uint16_t port = kDefaultPort;
  bool isServer = false;
  std::optional<std::size_t> numWorkers = std::nullopt;
  bool waitWorkers = true;
  std::chrono::milliseconds timeout = std::chrono::seconds(300);

  // A boolean value indicating whether multiple store instances can be
  // initialized with the same host:port pair.
  bool multiTenant = false;

  // If specified, and if isServer is true, the underlying TCPServer will take
  // over the bound socket associated to this fd. This option is useful to avoid
  // port assignment races in certain scenarios.
  std::optional<int> masterListenFd = std::nullopt;

  // A boolean value indicating whether to use the experimental libUV backend.
  bool useLibUV = false;
};

class TCPStore {
 public:
  explicit TCPStore(
      const std::string& masterAddr,
      std::uint16_t masterPort,
      std::optional<int> numWorkers = std::nullopt,
      bool isServer = false,
      const std::chrono::milliseconds& timeout = std::chrono::seconds(300),
      bool waitWorkers = true);

  explicit TCPStore(std::string host, const TCPStoreOptions& opts = {});

  ~TCPStore();

  void set(const std::string& key, const std::vector<uint8_t>& value);

  std::vector<uint8_t> get(const std::string& key);

  // Waits for all workers to join.
  void waitForWorkers();

  static constexpr std::chrono::milliseconds kConnectRetryDelay{1000};

 private:
  int64_t incrementValueBy(const std::string& key, int64_t delta);

  void ping();
  void validate();

  std::vector<uint8_t> doGet(detail::TCPClient* client, const std::string& key);

  void doWait(
      detail::TCPClient* client,
      const std::vector<std::string>& keys,
      std::chrono::milliseconds timeout);

  std::chrono::milliseconds timeout_;
  detail::SocketAddress addr_;
  std::shared_ptr<detail::TCPServer> server_;
  folly::Synchronized<std::unique_ptr<detail::TCPClient>> syncClient_;
  std::optional<std::size_t> numWorkers_;

  const std::string initKey_ = "init/";
  const std::string keyPrefix_ = "/";
  bool usingLibUv_ = false;
};

} // namespace ncclx::tcpstore
