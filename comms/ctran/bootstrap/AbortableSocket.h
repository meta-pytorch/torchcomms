// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <string>

#include <folly/Expected.h>
#include <folly/IPAddress.h>
#include <folly/SocketAddress.h>
#include "comms/ctran/bootstrap/ISocket.h"
#include "comms/ctran/utils/Abort.h"
#include "comms/ctran/utils/Exception.h"

using ctran::utils::Abort;

namespace ctran::bootstrap {

/**
 * C++ wrapper over socket interface, integrated with ctran::utils::Abort
 * to cancel active operations.
 */
class AbortableSocket : public ISocket {
// TODO(T243405238): Improve error reporting.
#define CHECK_ABORT_RETURN() \
  do {                       \
    if (abort_->Test()) {    \
      return ECONNABORTED;   \
    }                        \
  } while (0);

 public:
  /**
   * Create new unconnected AbortableSocket. `connect(..)` must be called to
   * establish connection and send/recv data.
   */
  explicit AbortableSocket(
      std::shared_ptr<Abort> abort =
          ctran::utils::createAbort(/*enabled=*/true))
      : abort_(abort) {
    if (abort_ == nullptr) {
      throw ctran::utils::Exception(
          "Invalid ctran::utils::Abort object", commInvalidArgument);
    }
  };

  /**
   * Construct AbortableSocket on already accepted socket
   */
  explicit AbortableSocket(
      int sockFd,
      folly::SocketAddress peerAddr,
      std::shared_ptr<Abort> abort =
          ctran::utils::createAbort(/*enabled=*/true));

  /**
   * AbortableSocket is movable
   */
  AbortableSocket(AbortableSocket&& other) noexcept;
  AbortableSocket& operator=(AbortableSocket&& other) noexcept;

  /**
   * But not copyable. This ensures 1:1 mapping from underlying
   * socketFd_ to its owner object.
   */
  AbortableSocket(const AbortableSocket& other) = delete;
  AbortableSocket& operator=(const AbortableSocket& other) = delete;

  /**
   * Automatically closes socket on destruction if not done explicitly.
   */
  ~AbortableSocket();

  /**
   * Connect to specified address and ifName asynchronously.
   * Returns immediately and connection may still be in progress.
   * Use isConnected() or waitForConnect() to check/wait for completion.
   *
   * Will close the socket if there is an error while connecting.
   *
   * @param addr - Target address to connect to
   * @param ifName - Interface name to bind to
   * @param timeout - Ignored; exists for consistency with Socket API.
   * @param numRetries - Ignored; exists for consistency with Socket API.
   * @param async - Ignored; exists for consistency with Socket API.
   * @return 0 on success or errno on error
   */
  int connect(
      const folly::SocketAddress& addr,
      const std::string& ifName,
      [[maybe_unused]] const std::chrono::milliseconds timeout =
          std::chrono::milliseconds(-1),
      [[maybe_unused]] size_t numRetries = -1,
      [[maybe_unused]] bool async = false) override;

  /**
   * Send all data. Blocks until all data is sent or timeout/abort.
   * Note: calling close concurrently with send/recv is not
   * thread-safe and may result in send/recv returning EBADF.
   * @return 0 on success, errno on error/timeout
   */
  int send(const void* buf, const size_t len) override;

  /**
   * Receive all data. Blocks until all data is received or timeout/abort.
   * Note: calling close concurrently with send/recv is not
   * thread-safe and may result in send/recv returning EBADF.
   * @return 0 on success, errno on error/timeout
   */
  int recv(void* buf, const size_t len) override;

  /**
   * Shutdown the socket. Calling close concurrently with send/recv is not
   * thread-safe and may result in send/recv returning EBADF.
   * @return 0 on success, errno on error.
   */
  int close() override;

  int getFd() const override {
    return fd_.load();
  }

  folly::SocketAddress getPeerAddress() const override {
    return peerAddr_;
  }

  folly::SocketAddress getLocalAddress() const override {
    return localAddr_;
  }

 private:
  void prepareSocket();

  /**
   * Pass negative timeout to block until success (or abort).
   * @return true if readable, false on timeout or error
   */
  bool waitForReadable(const std::chrono::milliseconds timeout);

  /**
   * Pass negative timeout to block until success (or abort).
   * @return true if writable, false on timeout or error
   */
  bool waitForWritable(const std::chrono::milliseconds timeout);

  /**
   * Pass negative timeout to block until success (or abort).
   * @return 0 if connected, errno on error
   */
  int waitForConnect(const std::chrono::milliseconds timeout);

  /**
   * Pass negative timeout to block until success (or abort).
   * @return true if event occurred, false on timeout or error
   */
  bool waitForEvent(short events, const std::chrono::milliseconds timeout);

  bool setShuttingDown(bool expected, bool target) {
    return shuttingDown_.compare_exchange_strong(expected, target);
  }

  std::atomic<int> fd_{-1};
  std::atomic_bool shuttingDown_{false};
  std::shared_ptr<Abort> abort_;
  folly::SocketAddress peerAddr_;
  folly::SocketAddress localAddr_;
};

class AbortableServerSocket : public IServerSocket {
 public:
  explicit AbortableServerSocket(
      int acceptRetryCnt,
      std::shared_ptr<Abort> abort =
          ctran::utils::createAbort(/*enabled=*/true))
      : acceptRetryCnt_(acceptRetryCnt), abort_(abort) {
    if (abort_ == nullptr) {
      throw ctran::utils::Exception(
          "Invalid ctran::utils::Abort object", commInvalidArgument);
    }
  }
  ~AbortableServerSocket();

  /**
   * AbortableServerSocket is movable
   */
  AbortableServerSocket(AbortableServerSocket&& other) noexcept;
  AbortableServerSocket& operator=(AbortableServerSocket&& other) noexcept;

  /**
   * But not copyable. This ensures 1:1 mapping from underlying
   * socketFd_ to its owner object.
   */
  AbortableServerSocket(const AbortableServerSocket& other) = delete;
  AbortableServerSocket& operator=(const AbortableServerSocket& other) = delete;

  /**
   * @return 0 on success, errno on error
   */
  int bind(
      const folly::SocketAddress& addr,
      const std::string& ifName,
      bool reusePort = false) override;

  int listen() override;

  int bindAndListen(const folly::SocketAddress& addr, const std::string& ifName)
      override;

  folly::Expected<folly::SocketAddress, int> getListenAddress() override;

  /**
   * @return AbortableSocket on success, errno on error/timeout
   */
  folly::Expected<std::unique_ptr<ISocket>, int> acceptSocket() override;

  /**
   * @param shutdown - Optional output param to indicate if socket was closed by
   * this call. Primarily used for testing.
   */
  int shutdown() override;

  int getFd() const override {
    return fd_;
  }

  bool hasShutDown() const override {
    return hasShutDown_;
  }

 private:
  /**
   * @return pointer to ISocket on success, errno on error
   */
  folly::Expected<std::unique_ptr<ISocket>, int> acceptAsync();

  bool setShuttingDown(bool expected, bool target) {
    return shuttingDown_.compare_exchange_strong(expected, target);
  }

  void prepareSocket();

  bool isV4_{false};
  int acceptRetryCnt_;
  int fd_{-1};

  std::shared_ptr<Abort> abort_;
  std::atomic_bool shuttingDown_{false};
  std::atomic_bool hasShutDown_{false};
};

} // namespace ctran::bootstrap
