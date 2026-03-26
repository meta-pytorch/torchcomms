// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"

namespace meta::comms {

/**
 * Extended bootstrap interface for ctran.
 *
 * Adds NVL domain collective operations (allGatherNvlDomain, barrierNvlDomain)
 * needed by ctran for IPC handle exchange and local synchronization.
 * The NVL domain may span multiple nodes when NVL fabric is enabled.
 *
 * Production implementations: BaselineBootstrap (ncclx), CtranAdapter (mccl).
 * Pipes will use this interface too.
 */
class ICtranBootstrap : public IBootstrap {
 public:
  ~ICtranBootstrap() override = default;

  /**
   * AllGather among a subset of ranks identified by `nvlRankToCommRank`.
   *
   * `buf` is a continuous memory segment of size `nvlNranks * len`.
   * `nvlLocalRank` is this rank's index in [0, nvlNranks).
   * `nvlRankToCommRank` maps NVL domain indices to global communicator ranks.
   */
  virtual folly::SemiFuture<int> allGatherNvlDomain(
      void* buf,
      int len,
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) = 0;

  /**
   * Barrier among a subset of ranks identified by `nvlRankToCommRank`.
   *
   * `nvlLocalRank` is this rank's index in [0, nvlNranks).
   * `nvlRankToCommRank` maps NVL domain indices to global communicator ranks.
   */
  virtual folly::SemiFuture<int> barrierNvlDomain(
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) = 0;
};

} // namespace meta::comms
