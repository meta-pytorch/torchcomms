// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/tcp_devmem/transport.h"
#include "folly/Singleton.h"

namespace ctran {

class CtranTcpDmSingleton {
 public:
  static std::shared_ptr<::comms::tcp_devmem::TransportInterface>
  getTransport();
};

} // namespace ctran
