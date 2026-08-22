// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

#include "socket.h"

// NCCLX-only socket helpers hoisted out of the forked upstream NCCL socket
// sources (misc/socket.cc, os/linux.cc). Keeping them here shrinks the seam
// that must be reconciled against pristine upstream NCCL on every rebase.

// Numeric-host string form of an IPv4/IPv6 socket address.
std::string ncclSocketToIPv6String(union ncclSocketAddress* addr);

// Applies the NCCL_SOCKET_TOS_CONFIG traffic class (IPv6) / TOS (IPv4) sockopt
// to an initialized socket. No-op when the cvar is left unset (-1). `family` is
// the socket's address family (AF_INET / AF_INET6).
ncclResult_t ncclSocketExtSetTos(struct ncclSocket* sock, int family);

// Binds a client socket to the interface named by NCCL_CLIENT_SOCKET_IFNAME
// via SO_BINDTODEVICE. No-op when the cvar is empty.
ncclResult_t ncclSocketExtBindToDevice(struct ncclSocket* sock);
