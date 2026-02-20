// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/tests/bootstrap/MockBootstrap.h"

namespace ctran::testing {

using ::testing::_;

void MockBootstrap::expectSuccessfulCtranInitCalls() {
  EXPECT_CALL(*this, allGather(_, _, _, _))
      .WillRepeatedly(
          [](void* buf, int len, int rank, int nranks) { return 0; });
  EXPECT_CALL(*this, allGatherIntraNode(_, _, _, _, _))
      .WillRepeatedly([](void* buf,
                         int len,
                         int localRank,
                         int localNRanks,
                         std::vector<int> localRankToCommRank) { return 0; });
  EXPECT_CALL(*this, barrierIntraNode(_, _, _))
      .WillRepeatedly([](int localRank,
                         int localNRanks,
                         std::vector<int> localRankToCommRank) { return 0; });
  EXPECT_CALL(*this, allGatherNvlDomain(_, _, _, _, _))
      .WillRepeatedly([](void* buf,
                         int len,
                         int nvlLocalRank,
                         int nvlNranks,
                         std::vector<int> nvlRankToCommRank) { return 0; });
  EXPECT_CALL(*this, barrierNvlDomain(_, _, _))
      .WillRepeatedly([](int nvlLocalRank,
                         int nvlNranks,
                         std::vector<int> nvlRankToCommRank) { return 0; });
}

} // namespace ctran::testing
