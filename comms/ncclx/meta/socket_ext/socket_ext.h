// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

#include "socket.h"

// NCCLX-only socket helpers hoisted out of the forked upstream NCCL socket
// sources (misc/socket.cc, os/linux.cc). Keeping them here shrinks the seam
// that must be reconciled against pristine upstream NCCL on every rebase.

// Numeric-host string form of an IPv4/IPv6 socket address.
std::string ncclSocketToIPv6String(union ncclSocketAddress* addr);
