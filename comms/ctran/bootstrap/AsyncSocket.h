// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <string>

#include <folly/Function.h>
#include <folly/SocketAddress.h>
#include <folly/container/F14Map.h>
#include <folly/futures/Future.h>
#include <folly/io/IOBuf.h>
#include <folly/io/async/AsyncServerSocket.h>
#include <folly/io/async/AsyncSocket.h>
#include <folly/io/async/AsyncTimeout.h>
#include <folly/io/async/DestructorCheck.h>

namespace ctran::bootstrap {

/**
 * Asynchronous fire-and-forget client socket for sending a single message.
 *
 * This class provides a simple interface for asynchronously connecting to a
 * destination, sending one message, and then closing the connection. All
 * operations run on the provided EventBase.
 *
 * Lifecycle:
 *   1. Connect to the destination address
 *   2. Send the message payload
 *   3. Invoke completion callback (optional)
 *   4. Close the socket
 *
 * The class manages its own lifetime through a self-reference during async
 * operations and automatically cleans up upon completion or error.
 *
 * User API:
 *   - send(): Static method that executes the entire lifecycle on the given
 *             EventBase. The completion callback is optional and will be
 *             invoked with nullptr on success or an exception pointer on error.
 *             Supports optional timeout.
 */
class AsyncClientSocket
    : public std::enable_shared_from_this<AsyncClientSocket>,
      private folly::AsyncSocket::ConnectCallback,
      private folly::AsyncTransport::WriteCallback,
      private folly::AsyncTimeout {
 public:
  using CompletionCb =
      folly::Function<void(const folly::AsyncSocketException*)>;

  // Fire-and-forget. Runs entirely on `evb`.
  static void send(
      folly::EventBase& evb,
      const folly::SocketAddress& dst,
      const void* buf,
      const size_t len,
      CompletionCb cb = nullptr,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(0));

 private:
  explicit AsyncClientSocket(folly::EventBase& evb)
      : folly::AsyncTimeout(&evb), evb_(evb) {}

  void start(
      std::unique_ptr<folly::IOBuf> payload,
      const folly::SocketAddress& dst,
      CompletionCb cb,
      std::chrono::milliseconds timeout);

  // ---- ConnectCallback ----
  void connectSuccess() noexcept override;
  void connectErr(const folly::AsyncSocketException& ex) noexcept override;

  // ---- WriteCallback ----
  void writeSuccess() noexcept override;
  void writeErr(
      size_t bytesWritten,
      const folly::AsyncSocketException& ex) noexcept override;

  // ---- AsyncTimeout ----
  void timeoutExpired() noexcept override;

  void finish(const folly::AsyncSocketException* err);

  folly::EventBase& evb_;
  folly::AsyncSocket::UniquePtr sock_;
  std::unique_ptr<folly::IOBuf> out_;
  // Self-reference to keep object alive during async operations
  std::shared_ptr<AsyncClientSocket> selfRef_;
  CompletionCb cb_;
};

/**
 * Asynchronous server socket that handles one-shot message reception.
 *
 * This class provides a simple interface for asynchronously accepting
 * connections, receiving a single message from each client, and then closing
 * the connection. All operations run on the provided EventBase.
 *
 * Lifecycle per connection:
 *   1. Accept incoming connection
 *   2. Receive one message of specified size
 *   3. Invoke receive callback (optional)
 *   4. Close the connection
 *
 * The server can handle multiple concurrent one-shot connections, with each
 * connection managed independently and automatically cleaned up upon
 * completion.
 *
 * User API:
 *   - start(): Binds to the specified address and begins accepting connections.
 *              Returns the actual listen address via SemiFuture.
 *   - stop(): Stops accepting new connections and cleans up existing ones.
 *   - getListenAddress(): Returns the current listen address if started.
 */
class AsyncServerSocket : private folly::AsyncServerSocket::AcceptCallback {
 public:
  using RecvCb = std::function<void(std::unique_ptr<folly::IOBuf>)>;

  AsyncServerSocket(folly::EventBase& evb) : evb_(evb) {}

  folly::SemiFuture<folly::SocketAddress> start(
      folly::SocketAddress bindAddr,
      size_t msgSize,
      RecvCb onRecv,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(0));

  folly::SemiFuture<folly::Unit> stop();

  folly::Expected<folly::SocketAddress, int> getListenAddress() const;

 private:
  // OneShotRecv: reads exactly one message, then closes.
  class OneShotRecv : public folly::AsyncTransport::ReadCallback,
                      public folly::DestructorCheck,
                      private folly::AsyncTimeout {
   public:
    using DoneCb = std::function<void(OneShotRecv*)>;

    OneShotRecv(
        folly::AsyncSocket::UniquePtr sock,
        folly::EventBase& evb,
        RecvCb onRecv,
        DoneCb onDone,
        size_t msgSize,
        std::chrono::milliseconds timeout)
        : folly::AsyncTimeout(&evb),
          msgSize_(msgSize),
          sock_(std::move(sock)),
          evb_(evb),
          onRecv_(std::move(onRecv)),
          onDone_(std::move(onDone)),
          timeout_(timeout) {}

    void begin();

    // ---- ReadCallback ----
    void getReadBuffer(void** buf, size_t* len) override;
    void readDataAvailable(size_t n) noexcept override;
    void readEOF() noexcept override;
    void readErr(const folly::AsyncSocketException& ex) noexcept override;

    // ---- AsyncTimeout ----
    void timeoutExpired() noexcept override;

   private:
    void closeNow();
    void notifyDone();

    const size_t msgSize_;
    folly::AsyncSocket::UniquePtr sock_;
    folly::EventBase& evb_;

    std::unique_ptr<folly::IOBuf> buf_;
    RecvCb onRecv_;
    DoneCb onDone_;
    size_t got_{0};
    std::chrono::milliseconds timeout_;
  };

  // ---- AcceptCallback ----
  void connectionAccepted(
      folly::NetworkSocket ns,
      const folly::SocketAddress& clientAddr,
      AcceptInfo /* info */) noexcept override;

  void acceptError(const std::exception& ex) noexcept override;

  folly::EventBase& evb_;
  size_t msgSize_;
  RecvCb onRecv_;
  folly::SocketAddress bindAddr_;
  std::shared_ptr<folly::AsyncServerSocket> server_;
  std::chrono::milliseconds timeout_;

  // EVB-owned: one-shot connections keyed by pointer (self-erasing on
  // completion)
  folly::F14FastMap<OneShotRecv*, std::unique_ptr<OneShotRecv>> conns_;
};

} // namespace ctran::bootstrap
