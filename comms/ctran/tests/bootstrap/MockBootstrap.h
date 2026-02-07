// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/common/bootstrap/IBootstrap.h"

namespace ctran::testing {

// Mock IBootstrap for testing
class MockBootstrap : public meta::comms::IBootstrap {
 public:
  MOCK_METHOD(
      folly::SemiFuture<int>,
      allGather,
      (void* buf, int len, int rank, int nranks),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      allGatherIntraNode,
      (void* buf,
       int len,
       int localRank,
       int localNranks,
       std::vector<int> localRankToCommRank),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      barrier,
      (int rank, int nranks),
      (override));
  MOCK_METHOD(
      folly::SemiFuture<int>,
      barrierIntraNode,
      (int localRank, int localNranks, std::vector<int> localRankToCommRank),
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

  void expectSuccessfulCtranInitCalls();
};

} // namespace ctran::testing
