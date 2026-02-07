// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comm.h"
#include "meta/algoconf/InfoExt.h"

namespace ncclx::algoconf {

// Apply algorithm info override from task->ext to task fields.
// Returns ncclInvalidUsage if partial override detected or isGrouped is true,
// since both cases are not supported.
// Precondition: task->ext.hasOverride() == true. see enqueue.cc
inline ncclResult_t infoExtOverride(
    struct ncclTaskColl* task,
    const bool isGrouped) {
  const auto& ext = task->ext;

  if (isGrouped) {
    WARN("ncclInfoExt: grouped collectives with ext override not supported");
    return ncclInvalidUsage;
  }

  // Partial override is invalid - must set all required fields
  if (!ext.isComplete()) {
    WARN(
        "ncclInfoExt: partial override not supported: must set algorithm, protocol, nMaxChannels, nWarps");
    return ncclInvalidUsage;
  }

  // Full override - apply all fields
  task->algorithm = ext.algorithm;
  task->protocol = ext.protocol;
  task->nMaxChannels = ext.nMaxChannels;
  task->nWarps = ext.nWarps;

  if (ext.opDevSet) {
    task->opDev = ext.opDev;
  }

  return ncclSuccess;
}

} // namespace ncclx::algoconf
