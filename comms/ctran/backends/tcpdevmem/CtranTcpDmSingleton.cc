// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/backends/tcpdevmem/CtranTcpDmSingleton.h"

namespace ctran {

// Use a LeakySingleton to avoid shutdown order issues where CtranMapper
// objects may outlive the singleton's destruction phase. LeakySingleton
// intentionally leaks the object at shutdown to prevent crashes.
folly::LeakySingleton<std::shared_ptr<::comms::tcp_devmem::Transport>>
    tcpTransportPtr([] {
      return new std::shared_ptr<::comms::tcp_devmem::Transport>(
          std::make_shared<::comms::tcp_devmem::Transport>());
    });

std::shared_ptr<::comms::tcp_devmem::TransportInterface>
CtranTcpDmSingleton::getTransport() {
  return folly::LeakySingleton<
      std::shared_ptr<::comms::tcp_devmem::Transport>>::get();
}

} // namespace ctran
