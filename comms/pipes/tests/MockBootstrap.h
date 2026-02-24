// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <vector>

#include <gmock/gmock.h>

#include "comms/ctran/interfaces/IBootstrap.h"

namespace comms::pipes::testing {

/// GMock-based mock of IBootstrap for unit testing.
class MockBootstrap : public ctran::bootstrap::IBootstrap {
 public:
  MOCK_METHOD(
      folly::SemiFuture<int>,
      allGather,
      (void* buf, int len, int rank, int nRanks),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      allGatherIntraNode,
      (void* buf,
       int len,
       int localRank,
       int localNRanks,
       std::vector<int> localRankToCommRank),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      barrier,
      (int rank, int nRanks),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      barrierIntraNode,
      (int localRank, int localNRanks, std::vector<int> localRankToCommRank),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      allGatherNvlDomain,
      (void* buf,
       int len,
       int nvlLocalRank,
       int nvlNRanks,
       std::vector<int> nvlRankToCommRank),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      barrierNvlDomain,
      (int nvlLocalRank, int nvlNRanks, std::vector<int> nvlRankToCommRank),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      send,
      (void* buf, int len, int peer, int tag),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      recv,
      (void* buf, int len, int peer, int tag),
      (override));
};

} // namespace comms::pipes::testing
