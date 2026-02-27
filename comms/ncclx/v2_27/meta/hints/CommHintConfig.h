// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h"

namespace ncclx {
inline const bool commUseCtran() {
  const auto useCtranHint = getTypedGlobalHint<bool>(HintKeys::kCommUseCtran);
  return NCCL_CTRAN_ENABLE || useCtranHint.value_or(false);
}

const std::string getCommUseCtranConfig();
} // namespace ncclx
