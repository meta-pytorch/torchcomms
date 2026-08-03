// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <optional>

#include "comm.h"
#include "meta/algoconf/InfoExt.h"

namespace ncclx::algoconf {

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

// Quantized collectives transport data in a smaller type (BF16, 2 bytes) than
// the input type (FP32, 4 bytes), so a transport-buffer step holds 2x more
// elements: doubling the chunk size fully utilizes the buffer and halves the
// number of PAT steps. Collectives without a quantize seed are unchanged.
inline int adjustChunkSizeForExt(
    const std::optional<ncclInfoExt>& ext,
    int chunkSize) {
  if (ext.has_value() && ext->quantizeRandomSeedPtr != nullptr) {
    return chunkSize * 2;
  }
  return chunkSize;
}

} // namespace ncclx::algoconf
