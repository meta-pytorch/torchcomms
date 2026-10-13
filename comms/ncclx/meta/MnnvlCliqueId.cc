// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/MnnvlCliqueId.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/NcclxLogger.h"
#include "meta/RankUtil.h"

namespace ncclx {

ncclResult_t assignMnnvlCliqueIdBasedOnCliqueSize(int* cliqueId) {
  NCCLX_LOG_IF(
      FATAL,
      NCCL_MNNVL_CLIQUE_SIZE <= 0,
      "Check failed: NCCL_MNNVL_CLIQUE_SIZE > 0: NCCL_MNNVL_CLIQUE_SIZE must be positive");
  NCCLX_LOG_IF(
      FATAL,
      NCCL_MNNVL_CLIQUE_ID != -1,
      "Check failed: NCCL_MNNVL_CLIQUE_ID == -1: NCCL_MNNVL_CLIQUE_SIZE and NCCL_MNNVL_CLIQUE_ID can NOT be set at the same time");
  auto globalRank = RankUtil::getGlobalRank();
  auto worldSize = RankUtil::getWorldSize();
  NCCLX_LOG_IF(
      FATAL,
      !globalRank.has_value(),
      "Check failed: globalRank.has_value(): RANK is not set");
  NCCLX_LOG_IF(
      FATAL,
      !worldSize.has_value(),
      "Check failed: worldSize.has_value(): WORLD_SIZE is not set");
  NCCLX_LOG_IF(
      FATAL,
      worldSize.value() % NCCL_MNNVL_CLIQUE_SIZE != 0,
      "Check failed: worldSize.value() % NCCL_MNNVL_CLIQUE_SIZE == 0: WORLD_SIZE is not a multiple of NCCL_MNNVL_CLIQUE_SIZE");
  *cliqueId = static_cast<int>(globalRank.value() / NCCL_MNNVL_CLIQUE_SIZE);
  return ncclSuccess;
}

} // namespace ncclx
