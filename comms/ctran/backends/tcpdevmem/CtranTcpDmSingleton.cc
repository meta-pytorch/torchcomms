// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/backends/tcpdevmem/CtranTcpDmSingleton.h"
#include <sys/utsname.h>
#include <filesystem>
#include "comms/ctran/utils/Checks.h"
#include "comms/tcp_devmem/bond_transport.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran {

std::vector<std::vector<std::string>> CtranTcpDmSingleton::getIfNames(
    const std::vector<std::string>& hcaList,
    int ifPerRank) {
  std::vector<std::vector<std::string>> devlists;
  std::vector<std::string> devs;
  for (const auto& s : hcaList) {
    std::vector<std::string> tokens;
    folly::split(':', s, tokens);
    if (tokens.empty() || tokens[0].empty()) {
      FB_ERRORTHROW(commInvalidArgument, "TCP-DEVMEM: invalid hcaList");
    }
    std::filesystem::path inputPath =
        "/sys/class/infiniband/" + tokens[0] + "/device/net";
    if (!std::filesystem::is_directory(inputPath)) {
      continue;
    }
    for (const auto& entry : std::filesystem::directory_iterator(inputPath)) {
      auto ifName = entry.path().filename().string();
      if (ifName.starts_with("beth")) {
        devs.push_back(ifName);
        if (devs.size() == ifPerRank) {
          devlists.push_back(std::move(devs));
        }
      }
    }
  }

  return devlists;
}

bool CtranTcpDmSingleton::supportBondTransport() {
  struct utsname uts{};

  if (uname(&uts) != 0) {
    FB_ERRORTHROW(commSystemError, "uname() failed with errno {}", errno);
  }

  if (!std::strcmp("aarch64", uts.machine)) {
    return true;
  }
  return false;
}

// Use a LeakySingleton to avoid shutdown order issues where CtranMapper
// objects may outlive the singleton's destruction phase. LeakySingleton
// intentionally leaks the object at shutdown to prevent crashes.
folly::LeakySingleton<::comms::tcp_devmem::TransportInterface> tcpTransportPtr(
    []() -> ::comms::tcp_devmem::TransportInterface* {
      if (!CtranTcpDmSingleton::supportBondTransport()) {
        return new ::comms::tcp_devmem::Transport();
      }

      auto devs = CtranTcpDmSingleton::getIfNames(
          NCCL_IB_HCA, NCCL_CTRAN_IB_DEVICES_PER_RANK);
      if (devs.size() && devs.front().size() == 1) {
        return new ::comms::tcp_devmem::Transport();
      } else {
        return new ::comms::tcp_devmem::BondTransport(devs);
      }
    });

::comms::tcp_devmem::TransportInterface* CtranTcpDmSingleton::getTransport() {
  return &folly::LeakySingleton<::comms::tcp_devmem::TransportInterface>::get();
}

} // namespace ctran
