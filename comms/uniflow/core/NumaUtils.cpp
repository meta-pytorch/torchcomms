// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/core/NumaUtils.h"

#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>

namespace uniflow {

int detectHostNumaNode(const void* addr) {
  if (addr == nullptr) {
    return -1;
  }
  int node = -1;
  long rc = syscall(
      SYS_get_mempolicy,
      &node,
      nullptr,
      0UL,
      const_cast<void*>(addr),
      MPOL_F_NODE | MPOL_F_ADDR);
  return rc == 0 ? node : -1;
}

} // namespace uniflow
