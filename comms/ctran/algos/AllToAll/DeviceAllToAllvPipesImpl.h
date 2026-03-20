// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <string>
#include <unordered_map>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"

namespace ctran::device_alltoallv_pipes {

// Fully-resolved kernel configuration for deviceAllToAllv.
// The constructor merges per-collective hints with env var / cvar defaults
// so that consumers read final values directly — no fallback logic needed.
//
// Resolution priority (highest wins):
//   per-collective hint (from map) > env var > CVAR > built-in default
struct CollectiveConfig {
  unsigned int numBlocks;
  unsigned int numThreads;
  bool blockScheduling; // false=warp(default), true=block scheduling

  // Construct with all defaults resolved from env vars / cvars.
  // Per-collective hints (if any) override those defaults.
  explicit CollectiveConfig(
      int nLocalRanks,
      const std::unordered_map<std::string, std::string>* hints_ptr = nullptr);
};

commResult_t setupKernelConfig(
    const void* sendbuff,
    void* recvbuff,
    const int64_t* sendcounts_d,
    const int64_t* recvcounts_d,
    commDataType_t datatype,
    CtranComm* comm,
    KernelConfig& config,
    ctran::device_alltoallv_pipes::KernArgs& kernArgs,
    int64_t sendcountsMultiplier = 1,
    int64_t recvcountsMultiplier = 1,
    const CollectiveConfig& collConfig = CollectiveConfig(0));

} // namespace ctran::device_alltoallv_pipes
