// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h"

namespace ncclx {
inline const bool commUseCtran() {
  const auto useCtranHint = getTypedGlobalHint<bool>(HintKeys::kCommUseCtran);
  return NCCL_CTRAN_ENABLE || useCtranHint.value_or(false);
}

// Check if PAT AVG should be enabled for this communicator.
// PAT AVG can be enabled via:
//   1. CVAR: NCCL_REDUCESCATTER_PAT_AVG_ENABLE=1
//   2. Hint: ncclx.comm.algo_reducescatter="avg:patavg"
// Returns true if either method enables PAT AVG.
inline const bool commUsePatAvg() {
  if (NCCL_REDUCESCATTER_PAT_AVG_ENABLE) {
    return true;
  }
  const auto algoHint = getGlobalHint(HintKeys::kCommAlgoReduceScatter);
  return algoHint.has_value() && algoHint.value() == "avg:patavg";
}

inline bool commNoLocal() {
  return getTypedGlobalHint<bool>(HintKeys::kCommNoLocal).value_or(false);
}

const std::string getCommUseCtranConfig();
const std::string getCommUsePatAvgConfig();
const std::string getCommNoLocalConfig();
} // namespace ncclx
