// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "comms/ctran/bootstrap/ISocket.h"

namespace ctran::bootstrap::testing {

class MockISocket : public ISocket {
 public:
  ~MockISocket() override = default;

  MOCK_METHOD(
      int,
      connect,
      (const folly::SocketAddress& addr,
       const std::string& ifName,
       const std::chrono::milliseconds timeout,
       size_t numRetries,
       bool async),
      (override));

  MOCK_METHOD(int, send, (const void* buf, const size_t len), (override));

  MOCK_METHOD(int, recv, (void* buf, const size_t len), (override));

  MOCK_METHOD(int, close, (), (override));

  MOCK_METHOD(int, getFd, (), (const, override));

  MOCK_METHOD(folly::SocketAddress, getPeerAddress, (), (const, override));

  MOCK_METHOD(folly::SocketAddress, getLocalAddress, (), (const, override));
};

} // namespace ctran::bootstrap::testing
