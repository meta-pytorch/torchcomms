// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/socket_ext/socket_ext.h"

#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <cstring>

#include "checks.h"
#include "comms/utils/cvars/nccl_cvars.h"

std::string ncclSocketToIPv6String(union ncclSocketAddress* addr) {
  if (addr == nullptr) {
    return {};
  }
  char host[NI_MAXHOST] = "";
  int flag = NI_NUMERICHOST;
  // NI_NUMERICHOST returns the numeric address form only: no name lookup / DNS.
  // patternlint-disable-next-line cpp-dns-deps
  (void)getnameinfo(
      &addr->sa,
      sizeof(union ncclSocketAddress),
      host,
      NI_MAXHOST,
      nullptr,
      0,
      flag);
  return {host};
}

ncclResult_t ncclSocketExtSetTos(struct ncclSocket* sock, int family) {
  if (NCCL_SOCKET_TOS_CONFIG == -1) {
    return ncclSuccess;
  }
  // referenced D77281608
  if (family == AF_INET6) {
    // For IPv6 set the traffic class field
    SYSCHECK(
        setsockopt(
            sock->socketDescriptor,
            IPPROTO_IPV6,
            IPV6_TCLASS,
            (char*)&NCCL_SOCKET_TOS_CONFIG,
            sizeof(int)),
        "setsockopt");
  } else {
    // For IPv4 set the TOS field
    SYSCHECK(
        setsockopt(
            sock->socketDescriptor,
            IPPROTO_IP,
            IP_TOS,
            (char*)&NCCL_SOCKET_TOS_CONFIG,
            sizeof(int)),
        "setsockopt");
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketExtBindToDevice(struct ncclSocket* sock) {
  if (NCCL_CLIENT_SOCKET_IFNAME.empty()) {
    return ncclSuccess;
  }
  const char* localIfName = NCCL_CLIENT_SOCKET_IFNAME.c_str();
  // bind client socket to specified interface
  ifreq ifr{};
  strncpy(ifr.ifr_name, localIfName, sizeof(ifr.ifr_name));
  ifr.ifr_name[sizeof(ifr.ifr_name) - 1] = '\0';
  SYSCHECK(
      setsockopt(
          sock->socketDescriptor,
          SOL_SOCKET,
          SO_BINDTODEVICE,
          (void*)&ifr.ifr_name,
          sizeof(ifr)),
      "setsockopt");
  INFO(NCCL_INIT, "ncclSocketConnect bind to interface %s", localIfName);
  return ncclSuccess;
}
