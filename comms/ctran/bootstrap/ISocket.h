// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fcntl.h>
#include <folly/IPAddress.h>
#include <folly/SocketAddress.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <string>

#include <folly/ScopeGuard.h>
#include <folly/logging/xlog.h>

namespace ctran::bootstrap {

// Abstract interface for async socket operations
class ISocket {
 public:
  virtual ~ISocket() = default;

  virtual int connect(
      const folly::SocketAddress& addr,
      const std::string& ifName,
      const std::chrono::milliseconds timeout = std::chrono::milliseconds(1000),
      size_t numRetries = 10,
      bool async = false) = 0;

  /**
   * Send provided bytes synchronously. Return 0 on success or errno
   */
  virtual int send(const void* buf, const size_t len) = 0;

  /**
   * Receive from the socket. Returns 0 on success or errno
   */
  virtual int recv(void* buf, const size_t len) = 0;

  virtual int close() = 0;

  // Getters
  virtual int getFd() const = 0;
  virtual folly::SocketAddress getPeerAddress() const = 0;
  virtual folly::SocketAddress getLocalAddress() const = 0;
}; // class ISocket

class IServerSocket {
 public:
  virtual ~IServerSocket() = default;

  /**
   * @param reusePort - Allow port reuse
   * @return 0 on success, errno on error
   */
  virtual int bind(
      const folly::SocketAddress& addr,
      const std::string& ifName,
      bool reusePort = false) = 0;

  virtual int listen() = 0;

  virtual int bindAndListen(
      const folly::SocketAddress& addr,
      const std::string& ifName) = 0;

  virtual folly::Expected<std::unique_ptr<ISocket>, int> acceptSocket() = 0;

  /**
   * Shutdown the server socket.
   *
   * This method is thread-safe. If a shutdown close operation is already
   * executing, then this method will return immediately with EAGAIN.
   */
  virtual int shutdown() = 0;

  virtual int getFd() const = 0;

  virtual folly::Expected<folly::SocketAddress, int> getListenAddress() = 0;

  virtual bool hasShutDown() const = 0;
}; // class IServerSocket

} // namespace ctran::bootstrap
