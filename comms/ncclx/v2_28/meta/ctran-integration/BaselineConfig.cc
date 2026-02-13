#include "meta/ctran-integration/BaselineConfig.h" // @manual
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h" // @manual

namespace ncclx {

const std::string getCommUseCtranConfig() {
  return fmt::format(
      "NCCL_CTRAN_ENABLE={}, GlobalHint {}={}",
      NCCL_CTRAN_ENABLE,
      HintKeys::kCommUseCtran,
      getTypedGlobalHint<bool>(HintKeys::kCommUseCtran).value_or(false));
}

const std::string getCommUsePatAvgConfig() {
  auto hintVal = getGlobalHint(HintKeys::kCommAlgoReduceScatter);
  return fmt::format(
      "NCCL_REDUCESCATTER_PAT_AVG_ENABLE={}, GlobalHint {}={}",
      NCCL_REDUCESCATTER_PAT_AVG_ENABLE,
      HintKeys::kCommAlgoReduceScatter,
      hintVal.value_or("(not set)"));
}
} // namespace ncclx
