// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

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
   * `buf` refers to a continuous memory segment that is of size
   * `localNranks * len`.  `localRank` must be a valid value between 0 to
   * `localNranks -1`. The `localRankToCommRank` must be an array of size
   * `localNranks` and provides mapping from local rank (on host) to
   * rank in a communicator.
   */
  virtual folly::SemiFuture<int> allGatherIntraNode(
      void* buf,
      int len,
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) = 0;

  /*
   * `rank` must be a valid value between 0 and `nranks - 1`
   */
  virtual folly::SemiFuture<int> barrier(int rank, int nranks) = 0;

  /**
   * `localRank` must be a valid value between 0 and `nranks -1`. The
   * `localRankToCommRank` must be an array of size `localNranks` and
   * provides mapping from local rank (on host) to rank in a
   * communicator
   */
  virtual folly::SemiFuture<int> barrierIntraNode(
      int localRank,
      int localNranks,
      std::vector<int> localRankToCommRank) = 0;

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
};

} // namespace meta::comms
