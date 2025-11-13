// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BaselineBootstrap.h"
#include "bootstrap.h"

namespace rcclx {

folly::SemiFuture<int> BaselineBootstrap::allGather(
    void* buf,
    int len,
    int /* rank */,
    int /* nranks */) {
  auto res = bootstrapAllGather(comm_->bootstrap, buf, len);
  return folly::makeSemiFuture<int>(static_cast<int>(res));
}

folly::SemiFuture<int> BaselineBootstrap::allGatherIntraNode(
    void* buf,
    int len,
    int localRank,
    int localNranks,
    std::vector<int> localRankToCommRank) {
  auto res = bootstrapIntraNodeAllGather(
      comm_->bootstrap,
      localRankToCommRank.data(),
      localRank,
      localNranks,
      buf,
      len);
  return folly::makeSemiFuture<int>(static_cast<int>(res));
}

folly::SemiFuture<int> BaselineBootstrap::barrier(int rank, int nranks) {
  auto res = bootstrapBarrier(comm_->bootstrap, rank, nranks, 0 /* tag */);
  return folly::makeSemiFuture<int>(static_cast<int>(res));
}

folly::SemiFuture<int> BaselineBootstrap::barrierIntraNode(
    int localRank,
    int localNranks,
    std::vector<int> localRankToCommRank) {
  auto res = bootstrapIntraNodeBarrier(
      comm_->bootstrap,
      localRankToCommRank.data(),
      localRank,
      localNranks,
      localRankToCommRank.at(0) /* tag */);
  return folly::makeSemiFuture<int>(static_cast<int>(res));
}

folly::SemiFuture<int>
BaselineBootstrap::send(void* buf, int len, int peer, int tag) {
  auto res = bootstrapSend(comm_->bootstrap, peer, tag, buf, len);
  return folly::makeSemiFuture<int>(static_cast<int>(res));
}

folly::SemiFuture<int>
BaselineBootstrap::recv(void* buf, int len, int peer, int tag) {
  auto res = bootstrapRecv(comm_->bootstrap, peer, tag, buf, len);
  return folly::makeSemiFuture<int>(static_cast<int>(res));
}

} // namespace rcclx
