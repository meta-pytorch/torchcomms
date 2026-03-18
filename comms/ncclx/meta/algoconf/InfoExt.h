// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <optional>

#include "device.h"

namespace ncclx::algoconf {

// Extension to ncclInfo for per-comm algorithm/protocol override.
// Constructor enforces that all required fields are provided.
struct ncclInfoExt {
  int algorithm;
  int protocol;
  int nMaxChannels;
  int nWarps;
  std::optional<ncclDevRedOpFull> opDev;

  ncclInfoExt(
      int algorithm,
      int protocol,
      int nMaxChannels,
      int nWarps,
      std::optional<ncclDevRedOpFull> opDev = std::nullopt)
      : algorithm(algorithm),
        protocol(protocol),
        nMaxChannels(nMaxChannels),
        nWarps(nWarps),
        opDev(opDev) {}
};

} // namespace ncclx::algoconf
