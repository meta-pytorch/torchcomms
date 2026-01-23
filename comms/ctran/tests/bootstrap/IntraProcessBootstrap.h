// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/common/bootstrap/IBootstrap.h"

namespace ctran::testing {

class IntraProcessBootstrap : public meta::comms::IBootstrap {
 public:
  struct State {
    State() {
      tmpBuf.resize(kMaxNRanks * kMaxPerRankAllGatherSize);
    }
    std::atomic<int> nArrivals{0};
    std::atomic<bool> sense{false};
    std::vector<char> tmpBuf;
  };

  static constexpr size_t kMaxNRanks = 8;
  static constexpr size_t kMaxPerRankAllGatherSize = 8192;
  explicit IntraProcessBootstrap(std::shared_ptr<State> state)
      : state_(state) {}

  folly::SemiFuture<int> allGather(void* buf, int len, int rank, int nRanks)
      override;
  folly::SemiFuture<int> allGatherIntraNode(
      void* buf,
      int len,
      int localRank,
      int localNRanks,
      std::vector<int> localRankToCommRank) override;
  folly::SemiFuture<int> barrier(int rank, int nRanks) override;
  folly::SemiFuture<int> barrierIntraNode(
      int localRank,
      int localNRanks,
      std::vector<int> localRankToCommRank) override;
  folly::SemiFuture<int> send(void* buf, int len, int peer, int tag) override;
  folly::SemiFuture<int> recv(void* buf, int len, int peer, int tag) override;

  // helper for bootstrap API impls, with name logging
  void barrierNamed(
      int rank,
      int nRanks,
      int timeoutSeconds,
      const std::string& name = "");

 private:
  std::shared_ptr<State> state_;
};

} // namespace ctran::testing
