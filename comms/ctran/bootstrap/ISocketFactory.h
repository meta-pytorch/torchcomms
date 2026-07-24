// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/bootstrap/ISocket.h"

namespace ctran::bootstrap {

// Abstract factory for creating ctran::bootstrap::ISocket instances
class ISocketFactory {
 public:
  ISocketFactory() {}

  virtual ~ISocketFactory() = default;

  virtual std::unique_ptr<ISocket> createClientSocket(
      std::shared_ptr<comms::fault_tolerance::Abort> abort = nullptr) = 0;

  virtual std::unique_ptr<ISocket> createClientSocket(
      int sockFd,
      const folly::SocketAddress& peerAddr,
      std::shared_ptr<comms::fault_tolerance::Abort> abort = nullptr) = 0;

  virtual std::unique_ptr<IServerSocket> createServerSocket(
      int acceptRetryCnt,
      std::shared_ptr<comms::fault_tolerance::Abort> abort = nullptr) = 0;
};

} // namespace ctran::bootstrap
