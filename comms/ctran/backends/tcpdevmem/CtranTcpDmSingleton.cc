// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/backends/tcpdevmem/CtranTcpDmSingleton.h"

namespace ctran {

folly::Singleton<::comms::tcp_devmem::Transport> tcpTransport;

std::shared_ptr<::comms::tcp_devmem::Transport>
CtranTcpDmSingleton::getTransport() {
  return tcpTransport.try_get();
}

} // namespace ctran
