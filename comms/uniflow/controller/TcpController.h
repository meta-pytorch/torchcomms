// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>

#include "comms/uniflow/controller/Controller.h"

namespace uniflow {
class EventBase;
} // namespace uniflow

namespace uniflow::controller {

/// Socket configuration for TcpServer and TcpClient.
/// Optional fields: valued → setsockopt, nullopt → OS kernel default.
/// Default-constructed with production-tuned values.
struct TcpSocketConfig {
  std::optional<std::chrono::seconds> connTimeout{std::chrono::seconds{30}};
  std::optional<int> socketBufSize{1 << 20}; // 1MB
  std::optional<bool> tcpNoDelay{true};
  std::optional<bool> enableKeepalive{true};
  std::optional<std::chrono::seconds> keepaliveIdle{std::chrono::seconds{60}};
  std::optional<std::chrono::seconds> keepaliveInterval{
      std::chrono::seconds{5}};
  std::optional<int> keepaliveCount{3};
  std::optional<std::chrono::milliseconds> userTimeout{
      std::chrono::milliseconds{60000}};

  int acceptRetryCnt{5};
  size_t connectRetries{10};
  std::chrono::milliseconds retryTimeout{std::chrono::milliseconds{1000}};

  static TcpSocketConfig osDefaults();
  Status validate() const;
};

class TcpConn : public Conn {
 public:
  static std::unique_ptr<TcpConn> create(int sock);

  TcpConn(const TcpConn&) = delete;
  TcpConn& operator=(const TcpConn&) = delete;
  TcpConn(TcpConn&&) = delete;
  TcpConn& operator=(TcpConn&&) = delete;

  ~TcpConn() override;

  Result<size_t> send(std::span<const uint8_t> data) override;
  Result<size_t> recv(std::vector<uint8_t>& data) override;
  void close() override;

  int getFd() const {
    return sock_;
  }

 private:
  explicit TcpConn(int sock) : sock_(sock) {}

  bool sendAll(const void* buf, size_t len);
  bool recvAll(void* buf, size_t len);
  bool exchangeMagic();

  int sock_;
};

// ---------------------------------------------------------------------------
// Accept policies — compile-time dispatch for BasicTcpServer<AcceptPolicy>.
// ---------------------------------------------------------------------------

/// Blocks the calling thread until a connection arrives, then returns a
/// ready future. No EventBase dependency. Zero async state.
struct SyncAccept {
  std::future<std::unique_ptr<Conn>> accept(
      std::atomic<int>& listenSock,
      int acceptRetryCnt,
      const std::string& id);

  /// Blocks until any in-flight accept() has returned, then closes the
  /// listen socket. Safe to destroy the server after shutdown() returns.
  void shutdown(std::atomic<int>& listenSock, const std::string& id);

 private:
  std::mutex mutex_;
};

/// Non-blocking accept via EventBase fd-watching. The first call lazily
/// sets up non-blocking mode and registers the listen socket with the
/// EventBase. Subsequent calls queue promises that are resolved as
/// connections arrive. All async state is accessed exclusively on the
/// loop thread via dispatch() — no atomics needed.
///
/// Lifetime: The EventBase must outlive all calls to shutdown(). The
/// BasicTcpServer destructor calls shutdown(), so either:
///   (a) the EventBase outlives the BasicTcpServer, OR
///   (b) the caller calls shutdown() before the EventBase is destroyed.
struct AsyncAccept {
  explicit AsyncAccept(EventBase& evb) : evb_(evb) {}

  std::future<std::unique_ptr<Conn>> accept(
      std::atomic<int>& listenSock,
      int acceptRetryCnt,
      const std::string& id);

  void shutdown(std::atomic<int>& listenSock, const std::string& id);

 private:
  // All private methods run on the loop thread only.
  void acceptPendingConnections(std::atomic<int>& listenSock);
  void teardown(int fd);

  EventBase& evb_;
  bool accepting_{false}; // loop-thread-only
  std::queue<std::unique_ptr<Conn>> readyConns_;
  std::queue<std::promise<std::unique_ptr<Conn>>> pendingPromises_;
};

// ---------------------------------------------------------------------------
// BasicTcpServer<AcceptPolicy> — policy-based TCP server template.
// ---------------------------------------------------------------------------

template <typename AcceptPolicy>
class BasicTcpServer : public Server {
 public:
  // Sync:  TcpServer("host:0")
  // Async: AsyncTcpServer("host:0", {}, evb)
  template <typename... PolicyArgs>
  explicit BasicTcpServer(
      std::string id,
      TcpSocketConfig config = {},
      PolicyArgs&&... args)
      : id_(std::move(id)),
        config_(std::move(config)),
        policy_(std::forward<PolicyArgs>(args)...) {
    parseId();
  }

  ~BasicTcpServer() override {
    shutdown();
  }
  BasicTcpServer(const BasicTcpServer&) = delete;
  BasicTcpServer& operator=(const BasicTcpServer&) = delete;
  BasicTcpServer(BasicTcpServer&&) = delete;
  BasicTcpServer& operator=(BasicTcpServer&&) = delete;

  Status init() override;

  const std::string& getId() const override {
    return id_;
  }

  std::future<std::unique_ptr<Conn>> accept() override {
    return policy_.accept(listenSock_, config_.acceptRetryCnt, id_);
  }

  int getPort() const {
    return port_;
  }

  // Thread-safe: can be called from a different thread than accept().
  void shutdown() {
    policy_.shutdown(listenSock_, id_);
  }

 private:
  void parseId();

  std::string id_;
  std::string host_;
  int port_{-1};
  std::atomic<int> listenSock_{-1};
  TcpSocketConfig config_;
  AcceptPolicy policy_;
};

extern template class BasicTcpServer<SyncAccept>;
extern template class BasicTcpServer<AsyncAccept>;

using TcpServer = BasicTcpServer<SyncAccept>;
using AsyncTcpServer = BasicTcpServer<AsyncAccept>;

// ---------------------------------------------------------------------------
// Connect policies — compile-time dispatch for BasicTcpClient<ConnectPolicy>.
// ---------------------------------------------------------------------------

/// Blocking connect with linear backoff retries. Returns a ready future.
struct SyncConnect {
  std::future<std::unique_ptr<Conn>> connect(
      const std::string& id,
      const TcpSocketConfig& config);
};

/// Non-blocking connect via EventBase EPOLLOUT watching. The EventBase must
/// outlive the TcpClient.
///
/// Note: This policy makes a single connect attempt. If the connect fails
/// immediately (e.g., ECONNREFUSED) no retry is performed.
/// TODO: Retry-with-backoff for async connects requires an EventBase timer.
struct AsyncConnect {
  explicit AsyncConnect(EventBase& evb) : evb_(evb) {}

  std::future<std::unique_ptr<Conn>> connect(
      const std::string& id,
      const TcpSocketConfig& config);

 private:
  EventBase& evb_;
};

// ---------------------------------------------------------------------------
// BasicTcpClient<ConnectPolicy> — policy-based TCP client template.
// ---------------------------------------------------------------------------

template <typename ConnectPolicy>
class BasicTcpClient : public Client {
 public:
  // Sync:  TcpClient()
  // Async: AsyncTcpClient({}, evb)
  template <typename... PolicyArgs>
  explicit BasicTcpClient(TcpSocketConfig config = {}, PolicyArgs&&... args)
      : config_(std::move(config)), policy_(std::forward<PolicyArgs>(args)...) {
    auto status = config_.validate();
    if (!status) {
      throw std::invalid_argument(
          "Invalid socket config: " + status.error().toString());
    }
  }

  ~BasicTcpClient() override = default;
  BasicTcpClient(const BasicTcpClient&) = delete;
  BasicTcpClient& operator=(const BasicTcpClient&) = delete;
  BasicTcpClient(BasicTcpClient&&) = delete;
  BasicTcpClient& operator=(BasicTcpClient&&) = delete;

  std::future<std::unique_ptr<Conn>> connect(std::string id) override {
    return policy_.connect(id, config_);
  }

 private:
  TcpSocketConfig config_;
  ConnectPolicy policy_;
};

extern template class BasicTcpClient<SyncConnect>;
extern template class BasicTcpClient<AsyncConnect>;

using TcpClient = BasicTcpClient<SyncConnect>;
using AsyncTcpClient = BasicTcpClient<AsyncConnect>;

} // namespace uniflow::controller
