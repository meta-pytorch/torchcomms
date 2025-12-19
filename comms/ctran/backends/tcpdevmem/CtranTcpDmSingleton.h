// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/tcp_devmem/transport.h"
#include "folly/Singleton.h"

namespace ctran {

class CtranTcpDmSingleton {
 public:
  // utility function to get the interface name from hacList
  static std::vector<std::vector<std::string>> getIfNames(
      const std::vector<std::string>& hcaList,
      int ifPerRank);

  static bool supportBondTransport();

  static ::comms::tcp_devmem::TransportInterface* getTransport();
};

} // namespace ctran
