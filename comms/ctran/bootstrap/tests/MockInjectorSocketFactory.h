// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "comms/ctran/bootstrap/ISocket.h"
#include "comms/ctran/bootstrap/ISocketFactory.h"
#include "comms/ctran/bootstrap/tests/MockIServerSocket.h"
#include "comms/ctran/bootstrap/tests/MockISocket.h"

namespace ctran::bootstrap::testing {

/**
 * Returns MockISocket and MockIServerSocket instances that
 * are created, configured/setup (e.g., with EXPECT_CALL), and
 * passed to the MockInjectorSocketFactory when it is created.
 */
class MockInjectorSocketFactory : public ISocketFactory {
 public:
  MockInjectorSocketFactory(
      std::vector<std::unique_ptr<MockISocket>> mockSockets)
      : ISocketFactory(),
        mockSockets_(std::move(mockSockets)),
        mockServerSockets_() {}

  MockInjectorSocketFactory(
      std::vector<std::unique_ptr<MockIServerSocket>> mockServerSockets)
      : ISocketFactory(),
        mockSockets_(),
        mockServerSockets_(std::move(mockServerSockets)) {}

  MockInjectorSocketFactory(
      std::vector<std::unique_ptr<MockISocket>> mockSockets,
      std::vector<std::unique_ptr<MockIServerSocket>> mockServerSockets)
      : ISocketFactory(),
        mockSockets_(std::move(mockSockets)),
        mockServerSockets_(std::move(mockServerSockets)) {}

  std::unique_ptr<ISocket> createClientSocket(
      std::shared_ptr<ctran::utils::Abort> abort = nullptr) override {
    if (currentIndex_ >= mockSockets_.size()) {
      throw std::runtime_error(
          "MockInjectorSocketFactory: No more mocked sockets available. "
          "Test attempted to create more sockets than provided.");
    }

    auto& mock = mockSockets_[currentIndex_++];
    return std::move(mock);
  }

  std::unique_ptr<ISocket> createClientSocket(
      int sockFd,
      const folly::SocketAddress& peerAddr,
      std::shared_ptr<ctran::utils::Abort> abort = nullptr) override {
    if (currentIndex_ >= mockSockets_.size()) {
      throw std::runtime_error(
          "MockInjectorSocketFactory: No more mocked sockets available. "
          "Test attempted to create more sockets than provided.");
    }

    auto& mock = mockSockets_[currentIndex_++];
    return std::move(mock);
  }

  std::unique_ptr<IServerSocket> createServerSocket(
      int acceptRetryCnt,
      std::shared_ptr<ctran::utils::Abort> abort = nullptr) override {
    if (currentServerIndex_ >= mockServerSockets_.size()) {
      throw std::runtime_error(
          "MockInjectorSocketFactory: No more mocked server sockets "
          "available. Test attempted to create more server sockets than "
          "provided.");
    }

    auto& mock = mockServerSockets_[currentServerIndex_++];
    return std::move(mock);
  }

 private:
  std::vector<std::unique_ptr<MockISocket>> mockSockets_;
  std::vector<std::unique_ptr<MockIServerSocket>> mockServerSockets_;
  size_t currentIndex_{0};
  size_t currentServerIndex_{0};
};

} // namespace ctran::bootstrap::testing
