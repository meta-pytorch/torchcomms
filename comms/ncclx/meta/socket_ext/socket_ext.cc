// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/socket_ext/socket_ext.h"

#include <netdb.h>

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
