// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stdexcept>
#include <vector>

#include <folly/futures/Future.h>

namespace meta::comms {

/*
 * Abstract class for all bootstrap operations. This is used
 * by CTRAN to perform various control plane operations for a
 * given communicator.
 *
 * APIs are designed to be non-blocking for performance and
 * parallel operations. It can be designed to be blocking
 * as well.
 *
 * All APIs return standard system error codes.
 */
class IBootstrap {
 public:
  virtual ~IBootstrap() = default;

  /**
   * `buf` refers to a continuous memory segment that is of size
   * `nranks * len`. `rank` must be a valid value between 0 to `nranks -1`
   */
  virtual folly::SemiFuture<int>
  allGather(void* buf, int len, int rank, int nranks) = 0;

  /*
   * `rank` must be a valid value between 0 and `nranks - 1`
   */
  virtual folly::SemiFuture<int> barrier(int rank, int nranks) = 0;

  /**
   * AllGather within an NVLink domain, which may span multiple hosts (MNNVL).
   */
  virtual folly::SemiFuture<int> allGatherNvlDomain(
      void* buf,
      int len,
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) {
    throw std::runtime_error("allGatherNvlDomain not implemented");
  }

  /**
   * Barrier within an NVLink domain, which may span multiple hosts (MNNVL).
   */
  virtual folly::SemiFuture<int> barrierNvlDomain(
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) {
    throw std::runtime_error("barrierNvlDomain not implemented");
  }

  /*
   * `buf` refers to a continuous memory segment that is of size `len`
   * `peer` must be a valid value between 0 and `nranks - 1`
   * `tag` must be a unique valid for each rank
   */
  virtual folly::SemiFuture<int>
  send(void* buf, int len, int peer, int tag) = 0;

  /*
   * `buf` refers to a continuous memory segment that is of size `len`
   * `peer` must be a valid value between 0 and `nranks - 1`
   * `tag` must be a unique valid for each rank
   */
  virtual folly::SemiFuture<int>
  recv(void* buf, int len, int peer, int tag) = 0;

  // Broadcast `len` bytes from `root` to all ranks.
  // `buf` is input on root, output on all other ranks.
  virtual folly::SemiFuture<int>
  broadcast(void* buf, int len, int root, int rank, int nranks) {
    constexpr int kBcastTag = 0x42;
    if (rank != root) {
      return recv(buf, len, root, kBcastTag);
    }
    for (int r = 0; r < nranks; r++) {
      if (r != root) {
        auto rc = send(buf, len, r, kBcastTag).get();
        if (rc != 0) {
          return folly::makeSemiFuture(rc);
        }
      }
    }
    return folly::makeSemiFuture(0);
  }
};

} // namespace meta::comms
