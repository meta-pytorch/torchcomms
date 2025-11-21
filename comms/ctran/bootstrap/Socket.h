// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <string>

#include <folly/Expected.h>
#include <folly/IPAddress.h>
#include <folly/SocketAddress.h>
#include "comms/ctran/bootstrap/ISocket.h"
#include "comms/ctran/bootstrap/ISocketFactory.h"

namespace ctran::bootstrap {

/*
 * API to retrieve the address of the interface. Ignores link-local addresses
 * and prefers IPv6 over IPv4 based on passed parameter. Filters the address
 * based on prefix if provided.
 * On error returns appropriate system error code.
 */
folly::Expected<folly::IPAddress, int> getInterfaceAddress(
    const std::string& ifName,
    const std::string& addrPrefix = "",
    bool preferV6ElseV4 = true);

/**
 * C++ wrapper over socket interface for communicating with the peer.
 * All APIs are blocking. All APIs return standard system error codes.
 */
class Socket : public ISocket {
 public:
  /*
   * Create new unconnected socket. `connect(..)` must be called to
   * establish connection and send/recv data.
   */
  Socket() = default;

  /**
   * Construct socket on already accepted socket
   */
  explicit Socket(int sockFd, bool async, folly::SocketAddress peerAddr);

  /**
   * Socket is movable
   */
  Socket(Socket&& other) noexcept;
  Socket& operator=(Socket&& other) noexcept;

  /*
   * But not copyable. This ensures 1:1 maping from underlying
   * socketFd_ to its owner object.
   */
  Socket(const Socket& other) = delete;
  Socket& operator=(const Socket& other) = delete;

  /**
   * Automatically closes socket on destruction if not done
   * explicitly.
   */
  ~Socket();

  /*
   * Connect to specified address and ifName. Sleeps time increases linearly for
   * every retry e.g. `retry x timeout`
   *
   * @param async - Configures the socket as async if set to true
   */
  int connect(
      const folly::SocketAddress& addr,
      const std::string& ifName,
      const std::chrono::milliseconds timeout = std::chrono::milliseconds(1000),
      size_t numRetries = 10,
      bool async = false) override;

  /**
   * Send provided bytes synchronously. Return 0 on success or errno
   */
  int send(const void* buf, const size_t len);

  /**
   * Receive from the socket. Returns 0 on success or errno
   */
  int recv(void* buf, const size_t len);

  /**
   * Receive from the socket. On success, returns the number of bytes read. On
   * failure, returns -1 and sets errno.
   */
  int recvAsync(void* buf, const size_t len);

  int close() override;

  int getFd() const override {
    return fd_;
  }

  folly::SocketAddress getPeerAddress() const override {
    return peerAddr_;
  }

  folly::SocketAddress getLocalAddress() const override {
    return localAddr_;
  }

 private:
  /*
   * Utility helper to set various socket options.
   */
  void prepareSocket(bool async);

  int fd_{-1};
  folly::SocketAddress peerAddr_;
  folly::SocketAddress localAddr_;
};

/**
 * Server socket to bind on the interface
 */
class ServerSocket : public IServerSocket {
 public:
  explicit ServerSocket(int acceptRetryCnt) : acceptRetryCnt_(acceptRetryCnt) {}

  /**
   * Shuts down the socket if not done explicitly
   */
  ~ServerSocket();

  /**
   * Server socket is movable
   */
  ServerSocket(ServerSocket&& other) noexcept;
  ServerSocket& operator=(ServerSocket&& other) noexcept;

  /*
   * But not copyable. This ensures 1:1 maping from underlying
   * socketFd_ to its owner object.
   */
  ServerSocket(const ServerSocket& other) = delete;
  ServerSocket& operator=(const ServerSocket& other) = delete;

  /**
   * Bind the socket to specified address in constructor. If port in addr is
   * 0 then any free available port on the system will be used. It can be
   * retrieved via getListenPort() API.
   * @param reusePort - Configures the socket to re-use the port even if port=0
   */
  int bind(
      const folly::SocketAddress& addr,
      const std::string& ifName,
      bool reusePort = false) override;
  int listen() override;
  int bindAndListen(const folly::SocketAddress& addr, const std::string& ifName)
      override;

  /*
   * Get the listen port of the socket. Socket must be set to startListen
   * before this API call.
   */
  folly::Expected<folly::SocketAddress, int> getListenAddress() override;

  /**
   * @return ISocket on success, errno on error/timeout
   */
  folly::Expected<std::unique_ptr<ISocket>, int> acceptSocket() override;

  /**
   * @return ISocket on success, errno on error/timeout
   */
  folly::Expected<Socket, int> accept(bool async = false);

  int shutdown() override;

  int getFd() const override {
    return fd_;
  }

  inline bool hasShutDown() const override {
    return hasShutDown_;
  }

 private:
  /*
   * Utility helper to set various socket options.
   */
  void prepareSocket();

  bool isV4_{false};
  int acceptRetryCnt_;
  int fd_{-1};
  std::atomic_bool hasShutDown_{false};
};

class SocketFactory : public ISocketFactory {
 public:
  explicit SocketFactory() {};

  std::unique_ptr<ISocket> createClientSocket(
      std::shared_ptr<ctran::utils::Abort> abort = nullptr) override;

  std::unique_ptr<ISocket> createClientSocket(
      int sockFd,
      const folly::SocketAddress& peerAddr,
      std::shared_ptr<ctran::utils::Abort> abort = nullptr) override;

  std::unique_ptr<IServerSocket> createServerSocket(
      int acceptRetryCnt,
      std::shared_ptr<ctran::utils::Abort> abort = nullptr) override;
};

} // namespace ctran::bootstrap
