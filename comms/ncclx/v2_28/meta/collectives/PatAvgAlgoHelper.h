// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/utils/cvars/nccl_cvars.h"
#include "device.h"
#include "info.h"

namespace ncclx {

// Check if PAT AVG mode is enabled via dedicated CVAR.
inline bool isPatAvgEnabled() {
  return NCCL_REDUCESCATTER_PAT_AVG_ENABLE;
}

// Check if PAT algorithm should be skipped for a given reduction operation.
// PAT doesn't support PreMulSum or SumPostDiv natively, but when PAT AVG
// is enabled, both are converted to ncclDevPatAvg and handled by PAT.
inline bool shouldSkipPatForReduceOp(ncclDevRedOp_t op) {
  if (isPatAvgEnabled()) {
    // When PAT AVG is enabled, PAT supports all AVG operations
    // (both PreMulSum and SumPostDiv will be converted to PatAvg)
    return false;
  }
  // Without PAT AVG, skip PAT for all AVG-related ops
  return op == ncclDevPreMulSum || op == ncclDevSumPostDiv;
}

// Switch opDev to ncclDevPatAvg when native PAT AVG is enabled.
// This should be called after algorithm selection in topoGetAlgoInfo().
// nRanks is needed to set scalarArg correctly for FuncPatAvg.
inline void maybeEnablePatAvg(struct ncclTaskColl* info, int nRanks) {
  if (info->algorithm == NCCL_ALGO_PAT && info->func == ncclFuncReduceScatter &&
      (info->opDev.op == ncclDevSumPostDiv ||
       info->opDev.op == ncclDevPreMulSum) &&
      isPatAvgEnabled()) {
    info->opDev.op = ncclDevPatAvg;
    // FuncPatAvg expects opArg = nRanks (just the integer count)
    info->opDev.scalarArg = static_cast<uint64_t>(nRanks);
  }
}

} // namespace ncclx
