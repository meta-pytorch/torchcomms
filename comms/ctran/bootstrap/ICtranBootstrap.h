// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"

namespace meta::comms {

/**
 * Extended bootstrap interface for ctran.
 *
 * Adds intra-node collective operations (allGatherIntraNode, barrierIntraNode)
 * needed by ctran for IPC handle exchange and local synchronization.
 *
 * Production implementations: BaselineBootstrap (ncclx), CtranAdapter (mccl).
 * Pipes will use this interface too.
 */
class ICtranBootstrap : public IBootstrap {
 public:
  ~ICtranBootstrap() override = default;

  /**
   * AllGather among a subset of ranks identified by `localRankToCommRank`.
   *
   * `buf` is a continuous memory segment of size `localNranks * len`.
   * `localRank` is this rank's index in [0, localNranks).
   * `localRankToCommRank` maps local indices to global communicator ranks.
   */
  virtual folly::SemiFuture<int> allGatherIntraNode(
      void* buf,
      int len,
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) = 0;

  /**
   * Barrier among a subset of ranks identified by `localRankToCommRank`.
   *
   * `localRank` is this rank's index in [0, localNranks).
   * `localRankToCommRank` maps local indices to global communicator ranks.
   */
  virtual folly::SemiFuture<int> barrierIntraNode(
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) = 0;
};

} // namespace meta::comms
