// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comm.h"
#include "meta/algoconf/ConfiguredCollective.h"
#include "meta/algoconf/InfoExt.h"

namespace ncclx::algoconf {

// adjustChunkSizeForExt is defined in ConfiguredCollective.h and re-exported
// here via the include above.

// Apply algorithm info override from task->ext to task fields.
// Returns ncclInvalidUsage if isGrouped is true,
// since grouped collectives with ext override are not supported.
// Precondition: task->ext.has_value() == true. see enqueue.cc
inline ncclResult_t infoExtOverride(
    struct ncclTaskColl* task,
    const bool isGrouped) {
  const auto& ext = *task->ext;

  if (isGrouped) {
    WARN("ncclInfoExt: grouped collectives with ext override not supported");
    return ncclInvalidUsage;
  }

  // Apply all fields
  task->algorithm = ext.algorithm;
  task->protocol = ext.protocol;
  task->nMaxChannels = ext.nMaxChannels;
  task->nWarps = ext.nWarps;

  if (ext.opDev.has_value()) {
    task->opDev = *ext.opDev;
  }

  return ncclSuccess;
}

} // namespace ncclx::algoconf
