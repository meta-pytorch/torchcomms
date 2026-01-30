// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdlib>
#include <cstring>
#include <mutex>
#include "comm.h"
#include "device.h"

namespace ncclx {

// Global flag for pat_postdiv detection (initialized once via call_once)
inline std::once_flag patPostDivInitFlag;
inline bool patPostDivEnabled = false;

// Check if NCCL_ALGO contains "pat_postdiv" (case-insensitive).
// Uses call_once for thread-safe one-time initialization.
inline bool isPatPostDivEnabled() {
  std::call_once(patPostDivInitFlag, []() {
    const char* algoStr = getenv("NCCL_ALGO");
    patPostDivEnabled = (algoStr && strcasestr(algoStr, "pat_postdiv"));
  });
  return patPostDivEnabled;
}

// Check if PAT algorithm should be skipped for a given reduction operation.
// PAT doesn't support PreMulSum or SumPostDiv natively, but when pat_postdiv
// is enabled, both are converted to ncclDevPatAvg and handled by PAT.
inline bool shouldSkipPatForReduceOp(ncclDevRedOp_t op) {
  if (isPatPostDivEnabled()) {
    // When pat_postdiv is enabled, PAT supports all AVG operations
    // (both PreMulSum and SumPostDiv will be converted to PatAvg)
    return false;
  }
  // Without pat_postdiv, skip PAT for all AVG-related ops
  return op == ncclDevPreMulSum || op == ncclDevSumPostDiv;
}

// Switch opDev to ncclDevPatAvg when native PAT AVG is enabled.
// This should be called after algorithm selection in topoGetAlgoInfo().
// nRanks is needed to set scalarArg correctly for FuncPatAvg.
inline void maybeEnablePatAvg(struct ncclTaskColl* info, int nRanks) {
  if (info->algorithm == NCCL_ALGO_PAT && info->func == ncclFuncReduceScatter &&
      (info->opDev.op == ncclDevSumPostDiv ||
       info->opDev.op == ncclDevPreMulSum) &&
      isPatPostDivEnabled()) {
    info->opDev.op = ncclDevPatAvg;
    // FuncPatAvg expects opArg = nRanks (just the integer count)
    info->opDev.scalarArg = static_cast<uint64_t>(nRanks);
  }
}

} // namespace ncclx
