// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <string>

#include <folly/Expected.h>
#include <folly/IPAddress.h>
#include <folly/SocketAddress.h>

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
class Socket {
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
      bool async = false);

  /**
   * Send provided bytes synchronously. Return 0 on success or errno
   */
  int send(const void* buf, const size_t len);

  /**
   * Receive from the socket. Returns 0 on success or errno
   */
  int recv(void* buf, const size_t len);

  /**
   * Receive from the socket. Returns 0 on success or errno
   */
  int recvAsync(void* buf, const size_t len);

  int close();

  int getFd() const {
    return fd_;
  }

  folly::SocketAddress getPeerAddress() const {
    return peerAddr_;
  }

  folly::SocketAddress getLocalAddress() const {
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
class ServerSocket {
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
      bool reusePort = false);
  int listen();
  int bindAndListen(
      const folly::SocketAddress& addr,
      const std::string& ifName);

  /*
   * Get the listen port of the socket. Socket must be set to startListen
   * before this API call.
   */
  folly::Expected<folly::SocketAddress, int> getListenAddress();

  /**
   * Accept new incoming connection synchronously or an error code
   * Configure the accepted socket as async or not
   */
  folly::Expected<Socket, int> accept(bool async = false);

  int shutdown();

  int getFd() const {
    return fd_;
  }

  inline bool hasShutDown() const {
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

} // namespace ctran::bootstrap
