// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

static_assert(
    __cplusplus >= 202002L,
    "ncclInfoExt requires C++20 for default member initializers in aggregate initialization");

#include "device.h"

namespace ncclx::algoconf {

// Extension to ncclInfo for per-comm algorithm/protocol override.
// All fields use sentinel values; C++20 default member initializers apply
// when omitted from aggregate initialization.
struct ncclInfoExt {
  int algorithm = NCCL_ALGO_UNDEF;

  int protocol = NCCL_PROTO_UNDEF;

  ncclDevRedOpFull opDev = {};
  bool opDevSet = false;

  int nMaxChannels = 0;
  int nWarps = 0;

  // Returns true if ANY field is set (partial or full override)
  bool hasOverride() const {
    return algorithm != NCCL_ALGO_UNDEF || protocol != NCCL_PROTO_UNDEF ||
        opDevSet || nMaxChannels > 0 || nWarps > 0;
  }

  // Returns true if ALL required fields are set for full override
  // opDevSet is optional - not all algorithms need opDev override
  bool isComplete() const {
    return algorithm != NCCL_ALGO_UNDEF && protocol != NCCL_PROTO_UNDEF &&
        nMaxChannels > 0 && nWarps > 0;
  }
};

} // namespace ncclx::algoconf
