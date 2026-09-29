// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"

namespace comms::prims {

/**
 * NvlBootstrapAdapter - Bootstrap adapter for NVLink domain subgroup.
 *
 * Redirects allGather/barrier to their NvlDomain variants. Used by
 * MultiPeerTransport to give MultiPeerNvlTransport a bootstrap that
 * operates only on NVLink-reachable ranks (which may span multiple hosts
 * on MNNVL systems like GB200 NVL72).
 *
 * Call flow:
 *   GpuMemHandler calls adapter->allGather(buf, len, localRank, localNRanks)
 *   → adapter delegates to underlying_->allGatherNvlDomain(
 *         buf, len, localRank, localNRanks, localRankToCommRank_)
 *
 * This allows GpuMemHandler and MultiPeerNvlTransport to operate with
 * local rank indices [0, nvlNRanks) while the underlying bootstrap
 * coordinates using global communicator ranks.
 */
class NvlBootstrapAdapter : public meta::comms::IBootstrap {
 public:
  /**
   * @param underlying          The global bootstrap instance.
   * @param localRankToCommRank Mapping: NVL local index → global comm rank.
   *                            Size must equal the number of NVL ranks
   *                            (including self).
   */
  NvlBootstrapAdapter(
      std::shared_ptr<meta::comms::IBootstrap> underlying,
      std::vector<int> localRankToCommRank)
      : underlying_(std::move(underlying)),
        localRankToCommRank_(std::move(localRankToCommRank)) {}

  folly::SemiFuture<int> allGather(void* buf, int len, int rank, int nranks)
      override {
    return underlying_->allGatherNvlDomain(
        buf, len, rank, nranks, localRankToCommRank_);
  }

  folly::SemiFuture<int> barrier(int rank, int nranks) override {
    return underlying_->barrierNvlDomain(rank, nranks, localRankToCommRank_);
  }

  folly::SemiFuture<int> allGatherNvlDomain(
      void* buf,
      int len,
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) override {
    return underlying_->allGatherNvlDomain(
        buf,
        len,
        nvlLocalRank,
        nvlNranks,
        translateLocalRanks(std::move(nvlRankToCommRank)));
  }

  folly::SemiFuture<int> barrierNvlDomain(
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) override {
    return underlying_->barrierNvlDomain(
        nvlLocalRank,
        nvlNranks,
        translateLocalRanks(std::move(nvlRankToCommRank)));
  }

  folly::SemiFuture<int> send(void* buf, int len, int peer, int tag) override {
    return underlying_->send(buf, len, localRankToCommRank_[peer], tag);
  }

  folly::SemiFuture<int> recv(void* buf, int len, int peer, int tag) override {
    return underlying_->recv(buf, len, localRankToCommRank_[peer], tag);
  }

 private:
  std::vector<int> translateLocalRanks(std::vector<int> localRanks) const {
    for (auto& rank : localRanks) {
      rank = localRankToCommRank_.at(static_cast<std::size_t>(rank));
    }
    return localRanks;
  }

  std::shared_ptr<meta::comms::IBootstrap> underlying_;
  std::vector<int> localRankToCommRank_;
};

} // namespace comms::prims
