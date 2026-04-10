// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <chrono>
#include <optional>

#include "comms/uniflow/controller/Controller.h"

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

class TcpServer : public Server {
 public:
  explicit TcpServer(std::string id, TcpSocketConfig config = {});
  TcpServer(const TcpServer&) = delete;
  TcpServer& operator=(const TcpServer&) = delete;
  TcpServer(TcpServer&&) = delete;
  TcpServer& operator=(TcpServer&&) = delete;
  ~TcpServer() override;

  const std::string& getId() const override;
  int getPort() const {
    return port_;
  }
  Status init() override;
  std::unique_ptr<Conn> accept() override;

  // Thread-safe: can be called from a different thread than accept().
  void shutdown();

 private:
  static Result<int> createListenSocket(int domain);
  Status configureAcceptedSocket(int sock);
  Status resolveAndBind(int domain);

  std::string id_;
  std::string host_;
  int port_{-1};
  std::atomic<int> listenSock_{-1};
  TcpSocketConfig config_;
};

class TcpClient : public Client {
 public:
  explicit TcpClient(TcpSocketConfig config = {});
  TcpClient(const TcpClient&) = delete;
  TcpClient& operator=(const TcpClient&) = delete;
  TcpClient(TcpClient&&) = delete;
  TcpClient& operator=(TcpClient&&) = delete;

  std::unique_ptr<Conn> connect(std::string id) override;

 private:
  Status configureClientSocket(int sock);

  TcpSocketConfig config_;
};

} // namespace uniflow::controller
