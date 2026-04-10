// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <chrono>

#include "comms/uniflow/controller/Controller.h"

namespace uniflow::controller {

class TcpConn : public Conn {
 public:
  // Factory: creates a TcpConn and validates the peer via magic exchange.
  // Returns nullptr and closes the socket if the handshake fails.
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
  explicit TcpServer(std::string id, int acceptRetryCnt = 5);
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

  // Gracefully shut down the listening socket. After this call, accept()
  // will return nullptr. Safe to call from a different thread than accept().
  void shutdown();

 private:
  static Result<int> createListenSocket(int domain);
  Status configureAcceptedSocket(int sock);
  Status resolveAndBind(int domain);

  std::string id_;
  std::string host_;
  int port_{-1};
  std::atomic<int> listenSock_{-1};
  int acceptRetryCnt_;
};

class TcpClient : public Client {
 public:
  TcpClient() = default;
  TcpClient(const TcpClient&) = delete;
  TcpClient& operator=(const TcpClient&) = delete;
  TcpClient(TcpClient&&) = delete;
  TcpClient& operator=(TcpClient&&) = delete;

  // Configure retry behavior for connect().
  TcpClient(size_t numRetries, std::chrono::milliseconds retryTimeout)
      : numRetries_(numRetries), retryTimeout_(retryTimeout) {}

  std::unique_ptr<Conn> connect(std::string id) override;

 private:
  static Status configureClientSocket(int sock);

  size_t numRetries_{10};
  std::chrono::milliseconds retryTimeout_{1000};
};

} // namespace uniflow::controller
