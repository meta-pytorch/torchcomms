// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "comms/ctran/interfaces/IBootstrap.h"

namespace comms::pipes {

/**
 * IntraNodeBootstrapAdapter - Bootstrap adapter for intra-node NVLink subgroup.
 *
 * Redirects allGather/barrier to their IntraNode variants. Used by
 * MultiPeerTransportStates to give MultiPeerNvlTransport a bootstrap that
 * operates only on intra-node (NVLink-reachable) ranks.
 *
 * Call flow:
 *   GpuMemHandler calls adapter->allGather(buf, len, localRank, localNRanks)
 *   → adapter delegates to underlying_->allGatherIntraNode(
 *         buf, len, localRank, localNRanks, localRankToCommRank_)
 *
 * This allows GpuMemHandler and MultiPeerNvlTransport to operate with
 * local rank indices [0, nvlNRanks) while the underlying bootstrap
 * coordinates using global communicator ranks.
 */
class IntraNodeBootstrapAdapter : public ctran::bootstrap::IBootstrap {
 public:
  /**
   * @param underlying          The global bootstrap instance.
   * @param localRankToCommRank Mapping: NVL local index → global comm rank.
   *                            Size must equal the number of NVL ranks
   *                            (including self).
   */
  IntraNodeBootstrapAdapter(
      std::shared_ptr<ctran::bootstrap::IBootstrap> underlying,
      std::vector<int> localRankToCommRank)
      : underlying_(std::move(underlying)),
        localRankToCommRank_(std::move(localRankToCommRank)) {}

  folly::SemiFuture<int> allGather(void* buf, int len, int rank, int nranks)
      override {
    return underlying_->allGatherIntraNode(
        buf, len, rank, nranks, localRankToCommRank_);
  }

  folly::SemiFuture<int> allGatherIntraNode(
      void* buf,
      int len,
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) override {
    return underlying_->allGatherIntraNode(
        buf, len, localRank, localNranks, std::move(localRankToCommRank));
  }

  folly::SemiFuture<int> barrier(int rank, int nranks) override {
    return underlying_->barrierIntraNode(rank, nranks, localRankToCommRank_);
  }

  folly::SemiFuture<int> barrierIntraNode(
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) override {
    return underlying_->barrierIntraNode(
        localRank, localNranks, std::move(localRankToCommRank));
  }

  folly::SemiFuture<int> send(void* buf, int len, int peer, int tag) override {
    return underlying_->send(buf, len, localRankToCommRank_[peer], tag);
  }

  folly::SemiFuture<int> recv(void* buf, int len, int peer, int tag) override {
    return underlying_->recv(buf, len, localRankToCommRank_[peer], tag);
  }

 private:
  std::shared_ptr<ctran::bootstrap::IBootstrap> underlying_;
  std::vector<int> localRankToCommRank_;
};

} // namespace comms::pipes
