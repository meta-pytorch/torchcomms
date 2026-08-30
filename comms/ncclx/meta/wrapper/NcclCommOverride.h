// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>

#include "comm.h"
#include "nccl.h"

namespace ncclx::wrapper {

// Accessors for per-comm NCCLX override state. Deliberately narrow: reading
// these fields must not require the full ncclxCommExt definition.

inline std::string ncclCommOverrideDesc(ncclComm_t comm) {
  if (comm == nullptr)
    return "null";
  const char* d = comm->config.commDesc;
  if (d == nullptr || d == NCCL_CONFIG_UNDEF_PTR)
    return "";
  return std::string(d);
}

// Whether the comm carries the PAT AVG override. Callers still have to check
// the operation is ncclAvg themselves; this reports comm state only.
inline bool ncclCommUsePatAvg(ncclComm_t comm) {
  if (comm == nullptr)
    return false;
#if defined(IS_NCCLX)
  return comm->usePatAvg_;
#else
  // Pristine 2.31 has no usePatAvg_ on ncclComm; PAT AVG there is expressed
  // as a configured collective instead.
  (void)comm;
  return false;
#endif
}

} // namespace ncclx::wrapper
