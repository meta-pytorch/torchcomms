// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/interfaces/IBootstrap.h" // @manual

#if defined(USE_ROCM)
#include "rccl.h" // @manual
#else
#include "nccl.h" // @manual
#endif

namespace meta::comms {

// Only used in unit tests for the components in /comms/common
class TestBaselineBootstrap : public ::ctran::bootstrap::IBootstrap {
 public:
  explicit TestBaselineBootstrap(ncclComm_t comm) : comm_(comm) {}
  virtual folly::SemiFuture<int>
  allGather(void* buf, int len, int rank, int nranks) override;

  virtual folly::SemiFuture<int> allGatherIntraNode(
      void* buf,
      int len,
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) override {
    throw std::runtime_error("Not implemented");
  }

  virtual folly::SemiFuture<int> barrier(int rank, int nranks) override {
    throw std::runtime_error("Not implemented");
  }

  virtual folly::SemiFuture<int> barrierIntraNode(
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) override {
    throw std::runtime_error("Not implemented");
  }

  virtual folly::SemiFuture<int> send(void* buf, int len, int peer, int tag)
      override {
    throw std::runtime_error("Not implemented");
  }

  virtual folly::SemiFuture<int> recv(void* buf, int len, int peer, int tag)
      override {
    throw std::runtime_error("Not implemented");
  }

 private:
  ncclComm_t comm_{nullptr};
};

} // namespace meta::comms
