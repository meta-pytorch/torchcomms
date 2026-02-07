// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/common/bootstrap/IBootstrap.h"

#include "nccl.h" // @manual

namespace ncclx {

class BaselineBootstrap : public ::meta::comms::IBootstrap {
 public:
  explicit BaselineBootstrap(ncclComm_t comm) : comm_(comm) {}

  virtual folly::SemiFuture<int>
  allGather(void* buf, int len, int rank, int nranks) override;

  virtual folly::SemiFuture<int> allGatherIntraNode(
      void* buf,
      int len,
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) override;

  virtual folly::SemiFuture<int> barrier(int rank, int nranks) override;

  virtual folly::SemiFuture<int> barrierIntraNode(
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) override;

  virtual folly::SemiFuture<int> send(void* buf, int len, int peer, int tag)
      override;

  virtual folly::SemiFuture<int> recv(void* buf, int len, int peer, int tag)
      override;

 private:
  ncclComm_t comm_{nullptr};
};

} // namespace ncclx
