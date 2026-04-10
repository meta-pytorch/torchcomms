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
  // Quantized collective fields: seed pointer for stochastic rounding (device
  // memory). Default to nullptr/nullopt so existing callers (PAT AVG) are
  // unaffected.
  uint64_t* quantizeRandomSeedPtr{nullptr};

  ncclInfoExt(
      int algorithm,
      int protocol,
      int nMaxChannels,
      int nWarps,
      std::optional<ncclDevRedOpFull> opDev = std::nullopt,
      uint64_t* quantizeRandomSeedPtr = nullptr)
      : algorithm(algorithm),
        protocol(protocol),
        nMaxChannels(nMaxChannels),
        nWarps(nWarps),
        opDev(opDev),
        quantizeRandomSeedPtr(quantizeRandomSeedPtr) {}
};

} // namespace ncclx::algoconf
