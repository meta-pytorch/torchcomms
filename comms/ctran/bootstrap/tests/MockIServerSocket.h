// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "comms/ctran/bootstrap/ISocket.h"

namespace ctran::bootstrap::testing {

class MockIServerSocket : public IServerSocket {
 public:
  ~MockIServerSocket() override = default;

  MOCK_METHOD(
      int,
      bind,
      (const folly::SocketAddress& addr,
       const std::string& ifName,
       bool reusePort),
      (override));

  MOCK_METHOD(int, listen, (), (override));

  MOCK_METHOD(
      int,
      bindAndListen,
      (const folly::SocketAddress& addr, const std::string& ifName),
      (override));

  MOCK_METHOD(
      (folly::Expected<std::unique_ptr<ISocket>, int>),
      acceptSocket,
      (),
      (override));

  MOCK_METHOD(int, shutdown, (), (override));

  MOCK_METHOD(int, getFd, (), (const, override));

  MOCK_METHOD(
      (folly::Expected<folly::SocketAddress, int>),
      getListenAddress,
      (),
      (override));

  MOCK_METHOD(bool, hasShutDown, (), (const, override));
};

} // namespace ctran::bootstrap::testing
