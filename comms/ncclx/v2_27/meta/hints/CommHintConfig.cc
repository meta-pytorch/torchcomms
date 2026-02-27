#include "meta/hints/CommHintConfig.h" // @manual
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
} // namespace ncclx
