// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stdexcept>
#include <vector>

#include <folly/futures/Future.h>

namespace ctran::bootstrap {

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

  /**
   * AllGather within an NVLink domain, which may span multiple hosts (MNNVL).
   *
   * `buf` refers to a continuous memory segment of size `nvlNranks * len`.
   * `nvlLocalRank` is this rank's index within the NVL domain [0, nvlNranks).
   * `nvlRankToCommRank` maps NVL-local indices to global communicator ranks.
   *
   * Unlike allGatherIntraNode (which uses a host-scoped communicator),
   * this creates a dynamic subcommunicator from the specified global ranks,
   * supporting cross-host NVLink domains like GB200 NVL72.
   *
   * Subclasses must override this if NVL domain operations are needed.
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
   *
   * `nvlLocalRank` is this rank's index within the NVL domain [0, nvlNranks).
   * `nvlRankToCommRank` maps NVL-local indices to global communicator ranks.
   *
   * Subclasses must override this if NVL domain operations are needed.
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
};

} // namespace ctran::bootstrap
