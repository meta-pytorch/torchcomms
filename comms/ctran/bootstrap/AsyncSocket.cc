// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AsyncSocket.h"

#include <folly/logging/xlog.h>

namespace ctran::bootstrap {

//
// AsyncClientSocket Implementation
//

void AsyncClientSocket::send(
    folly::EventBase& evb,
    const folly::SocketAddress& dst,
    const void* buf,
    const size_t len,
    CompletionCb cb,
    std::chrono::milliseconds timeout) {
  // Self-owning: lifetime tied to async completion.
  auto self = std::shared_ptr<AsyncClientSocket>(new AsyncClientSocket(evb));
  auto payload = folly::IOBuf::wrapBuffer(buf, len);
  evb.runInEventBaseThread([self,
                            payload = std::move(payload),
                            dst,
                            cb = std::move(cb),
                            timeout]() mutable {
    self->start(std::move(payload), dst, std::move(cb), timeout);
  });
}

void AsyncClientSocket::start(
    std::unique_ptr<folly::IOBuf> payload,
    const folly::SocketAddress& dst,
    CompletionCb cb,
    std::chrono::milliseconds timeout) {
  // Keep ourselves alive until finish() is called.
  selfRef_ = shared_from_this();
  cb_ = std::move(cb);
  out_ = std::move(payload);
  sock_.reset(new folly::AsyncSocket(&evb_));
  sock_->setNoDelay(true);

  // Schedule timeout if specified
  if (timeout.count() > 0) {
    scheduleTimeout(timeout);
  }

  sock_->connect(this, dst);
}

void AsyncClientSocket::connectSuccess() noexcept {
  folly::SocketAddress peer;
  sock_->getPeerAddress(&peer);
  XLOGF(
      INFO,
      "Connected to {}, fd={}",
      peer.describe(),
      sock_->getNetworkSocket().toFd());
  sock_->writeChain(this, std::move(out_));
}

void AsyncClientSocket::connectErr(
    const folly::AsyncSocketException& ex) noexcept {
  XLOGF(ERR, "AsyncClientSocket connect failed: {}", ex.what());
  cancelTimeout();
  finish(&ex);
}

void AsyncClientSocket::writeSuccess() noexcept {
  cancelTimeout();
  finish(nullptr);
}

void AsyncClientSocket::writeErr(
    size_t bytesWritten,
    const folly::AsyncSocketException& ex) noexcept {
  XLOGF(
      ERR,
      "AsyncClientSocket write failed after {} bytes: {}",
      bytesWritten,
      ex.what());
  cancelTimeout();
  finish(&ex);
}

void AsyncClientSocket::timeoutExpired() noexcept {
  XLOGF(ERR, "AsyncClientSocket operation timed out");
  auto ex = folly::AsyncSocketException(
      folly::AsyncSocketException::TIMED_OUT, "Operation timed out");
  finish(&ex);
}

void AsyncClientSocket::finish(const folly::AsyncSocketException* err) {
  if (sock_) {
    if (!err) {
      sock_->close();
    } else {
      sock_->closeNow();
    }
  }
  if (cb_) {
    cb_(err);
  }
  // Release self-reference, allowing destruction
  selfRef_.reset();
}

//
// AsyncServerSocket Implementation
//

folly::SemiFuture<folly::SocketAddress> AsyncServerSocket::start(
    folly::SocketAddress bindAddr,
    size_t msgSize,
    RecvCb onRecv,
    std::chrono::milliseconds timeout) {
  auto [p, f] = folly::makePromiseContract<folly::SocketAddress>();
  evb_.runInEventBaseThread([this,
                             bindAddr = std::move(bindAddr),
                             msgSize,
                             onRecv = std::move(onRecv),
                             timeout,
                             p = std::move(p)]() mutable {
    if (server_) {
      // Server already started, just return its address
      p.setValue(server_->getAddress());
      return;
    }
    msgSize_ = msgSize;
    timeout_ = timeout;
    onRecv_ = std::move(onRecv);
    bindAddr_ = std::move(bindAddr);
    server_ = folly::AsyncServerSocket::newSocket(&evb_);
    server_->setReusePortEnabled(true);
    server_->bind(bindAddr_);
    server_->addAcceptCallback(this, nullptr);
    server_->listen(1024 /* backlog */);
    server_->startAccepting();
    auto addr = server_->getAddress();
    XLOGF(
        INFO,
        "AsyncServerSocket started listening on {}, fd={}",
        addr.describe(),
        server_->getNetworkSocket().toFd());
    p.setValue(addr);
  });
  return std::move(f);
}

folly::SemiFuture<folly::Unit> AsyncServerSocket::stop() {
  auto [p, f] = folly::makePromiseContract<folly::Unit>();
  evb_.runInEventBaseThread([this, p = std::move(p)]() mutable {
    if (server_) {
      auto addr = server_->getAddress();
      XLOGF(
          INFO,
          "AsyncServerSocket is shutting down on {}, fd={}",
          addr.describe(),
          server_->getNetworkSocket().toFd());
      server_->pauseAccepting();
      server_.reset();
    }
    conns_.clear();
    p.setValue(folly::unit); // ← Signal completion
  });
  return std::move(f);
}

folly::Expected<folly::SocketAddress, int> AsyncServerSocket::getListenAddress()
    const {
  if (!server_) {
    return folly::makeUnexpected(EBADF);
  }
  return server_->getAddress();
}

void AsyncServerSocket::connectionAccepted(
    folly::NetworkSocket ns,
    const folly::SocketAddress& clientAddr,
    AcceptInfo /* info */) noexcept {
  auto sock = folly::AsyncSocket::UniquePtr(new folly::AsyncSocket(&evb_, ns));
  folly::SocketAddress localAddr;
  localAddr.setFromLocalAddress(ns);
  XLOGF(
      INFO,
      "Accepted a new connection fd={}, local={}, peer={}. Waiting for 1 message to arrive.",
      ns.toFd(),
      localAddr.describe(),
      clientAddr.describe());
  // Create OneShotRecv and store by raw pointer; it erases itself on
  // completion.
  auto up = std::make_unique<OneShotRecv>(
      std::move(sock),
      evb_,
      onRecv_, // Copy the function
      // onDone: erase from map (already on EVB)
      [this](OneShotRecv* self) {
        conns_.erase(self); // destroys it
      },
      msgSize_,
      timeout_);
  OneShotRecv* raw = up.get();
  conns_.emplace(raw, std::move(up));
  raw->begin();
}

void AsyncServerSocket::acceptError(const std::exception& ex) noexcept {
  XLOGF(ERR, "AsyncServerSocket accept failed: {}", ex.what());
}

//
// AsyncServerSocket::OneShotRecv Implementation
//

void AsyncServerSocket::OneShotRecv::begin() {
  if (!sock_) {
    notifyDone();
    return;
  }
  sock_->setNoDelay(true);
  // Preallocate required IOBuf for receiving.
  if (!buf_) {
    buf_ = folly::IOBuf::create(msgSize_);
  }

  // Schedule timeout if specified
  if (timeout_.count() > 0) {
    scheduleTimeout(timeout_);
  }

  // Registers read handler getReadBuffer, readDataAvailable, readEOF and
  // readErr.
  sock_->setReadCB(this);
}

void AsyncServerSocket::OneShotRecv::getReadBuffer(void** buf, size_t* len) {
  size_t remaining = msgSize_ - got_;
  *buf = buf_->writableTail();
  *len = std::min(remaining, buf_->tailroom());
}

void AsyncServerSocket::OneShotRecv::readDataAvailable(size_t n) noexcept {
  buf_->append(n);
  got_ += n;
  if (got_ >= msgSize_) {
    // We have at least msgSize_ bytes; stop further reads first.
    cancelTimeout();
    sock_->setReadCB(nullptr);
    if (onRecv_) {
      onRecv_(std::move(buf_));
    }
    closeNow();
  }
}

void AsyncServerSocket::OneShotRecv::readEOF() noexcept {
  cancelTimeout();
  closeNow();
}

void AsyncServerSocket::OneShotRecv::readErr(
    const folly::AsyncSocketException& ex) noexcept {
  XLOGF(ERR, "AsyncServerSocket read error: {}", ex.what());
  cancelTimeout();
  closeNow();
}

void AsyncServerSocket::OneShotRecv::closeNow() {
  if (sock_) {
    sock_->closeNow();
    sock_.reset();
  }
  notifyDone();
}

void AsyncServerSocket::OneShotRecv::timeoutExpired() noexcept {
  XLOGF(ERR, "AsyncServerSocket receive operation timed out");
  closeNow();
}

void AsyncServerSocket::OneShotRecv::notifyDone() {
  if (onDone_) {
    auto cb = std::move(onDone_);
    // Ensure this object stays alive during callback
    folly::DestructorCheck::Safety safety(*this);
    cb(this);
  }
}

} // namespace ctran::bootstrap
