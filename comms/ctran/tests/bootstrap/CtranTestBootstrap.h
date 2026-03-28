// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "comms/ctran/bootstrap/ICtranBootstrap.h"

namespace ctran::testing {

/**
 * Test-only ICtranBootstrap that wraps a single IBootstrap instance.
 * Global operations delegate directly; NVL-domain operations are implemented
 * via pairwise send/recv through the global bootstrap.
 */
class CtranTestBootstrap : public meta::comms::ICtranBootstrap {
 public:
  explicit CtranTestBootstrap(
      std::unique_ptr<meta::comms::IBootstrap> bootstrap)
      : bootstrap_(std::move(bootstrap)) {}

  // -- Global operations (delegated to bootstrap_) --

  folly::SemiFuture<int> allGather(void* buf, int len, int rank, int nranks)
      override {
    return bootstrap_->allGather(buf, len, rank, nranks);
  }

  folly::SemiFuture<int> barrier(int rank, int nranks) override {
    return bootstrap_->barrier(rank, nranks);
  }

  folly::SemiFuture<int> send(void* buf, int len, int peer, int tag) override {
    return bootstrap_->send(buf, len, peer, tag);
  }

  folly::SemiFuture<int> recv(void* buf, int len, int peer, int tag) override {
    return bootstrap_->recv(buf, len, peer, tag);
  }

  folly::SemiFuture<int>
  broadcast(void* buf, int len, int root, int rank, int nranks) override {
    return bootstrap_->broadcast(buf, len, root, rank, nranks);
  }

  // -- NvlDomain operations (pairwise sendrecv via bootstrap_) --

  folly::SemiFuture<int> allGatherNvlDomain(
      void* buf,
      int len,
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) override {
    int myGlobalRank = nvlRankToCommRank[nvlLocalRank];
    for (int nr = 0; nr < nvlNranks; ++nr) {
      if (nr == nvlLocalRank) {
        continue;
      }
      int peer = nvlRankToCommRank[nr];
      int tag = nextTag(peer);
      auto* myChunk =
          static_cast<char*>(buf) + static_cast<size_t>(nvlLocalRank) * len;
      auto* peerChunk = static_cast<char*>(buf) + static_cast<size_t>(nr) * len;
      if (myGlobalRank < peer) {
        auto rc = bootstrap_->send(myChunk, len, peer, tag).get();
        if (rc != 0) {
          return folly::makeSemiFuture(rc);
        }
        rc = bootstrap_->recv(peerChunk, len, peer, tag).get();
        if (rc != 0) {
          return folly::makeSemiFuture(rc);
        }
      } else {
        auto rc = bootstrap_->recv(peerChunk, len, peer, tag).get();
        if (rc != 0) {
          return folly::makeSemiFuture(rc);
        }
        rc = bootstrap_->send(myChunk, len, peer, tag).get();
        if (rc != 0) {
          return folly::makeSemiFuture(rc);
        }
      }
    }
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> barrierNvlDomain(
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) override {
    int myGlobalRank = nvlRankToCommRank[nvlLocalRank];
    uint8_t dummy = 0;
    for (int nr = 0; nr < nvlNranks; ++nr) {
      if (nr == nvlLocalRank) {
        continue;
      }
      int peer = nvlRankToCommRank[nr];
      int tag = nextTag(peer);
      if (myGlobalRank < peer) {
        auto rc = bootstrap_->send(&dummy, sizeof(dummy), peer, tag).get();
        if (rc != 0) {
          return folly::makeSemiFuture(rc);
        }
        rc = bootstrap_->recv(&dummy, sizeof(dummy), peer, tag).get();
        if (rc != 0) {
          return folly::makeSemiFuture(rc);
        }
      } else {
        auto rc = bootstrap_->recv(&dummy, sizeof(dummy), peer, tag).get();
        if (rc != 0) {
          return folly::makeSemiFuture(rc);
        }
        rc = bootstrap_->send(&dummy, sizeof(dummy), peer, tag).get();
        if (rc != 0) {
          return folly::makeSemiFuture(rc);
        }
      }
    }
    return folly::makeSemiFuture(0);
  }

 private:
  // IBootstrap internally uses (peer, tag) as the key to match send/recv
  // pairs. By maintaining a per-peer monotonically increasing tag, concurrent
  // communications across different subgroups (intra-node, NVL-domain) are
  // guaranteed not to mismatch, since each peer pair always gets a unique tag.
  int nextTag(int peer) {
    return peerTags_[peer]++;
  }

  std::unique_ptr<meta::comms::IBootstrap> bootstrap_;
  std::unordered_map<int, int> peerTags_;
};

} // namespace ctran::testing
