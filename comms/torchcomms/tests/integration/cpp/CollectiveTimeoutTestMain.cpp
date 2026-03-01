// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Timeout test for all 16 collective operations.
// - Timeout logic: TimeoutTestHelper::launch() forks child processes.
//   Rank 0 only runs warmup ops then sleeps, so rank 1's final op
//   (issued with a short timeout) has no matching peer and times out.
// - Call path: TEST_P → testTimeout → helper_.launch(childBody) where
//   childBody builds an ops vector and calls helper_.exec(mode, ops).
//   exec() handles eager vs graph mode dispatch transparently.

// TODO: Port GraphEventTracker from ncclx to nccl for graph mode timeout.

#include <gtest/gtest.h>
#include "CollectiveTimeoutTest.hpp"

using ExecMode = torch::comms::test::TimeoutTestHelper::ExecMode;
using CollectiveType = CollectiveTimeoutTest::CollectiveType;

TEST_P(CollectiveTimeoutTest, Timeout) {
  const auto [type_int, mode_int] = GetParam();
  testTimeout(
      static_cast<CollectiveType>(type_int), static_cast<ExecMode>(mode_int));
}

INSTANTIATE_TEST_SUITE_P(
    AllCollectives,
    CollectiveTimeoutTest,
    ::testing::Combine(
        ::testing::Values(
            static_cast<int>(CollectiveType::kSendRecv),
            static_cast<int>(CollectiveType::kBatchSendRecv),
            static_cast<int>(CollectiveType::kBroadcast),
            static_cast<int>(CollectiveType::kAllReduce),
            static_cast<int>(CollectiveType::kReduce),
            static_cast<int>(CollectiveType::kAllGather),
            static_cast<int>(CollectiveType::kAllGatherSingle),
            static_cast<int>(CollectiveType::kAllGatherV),
            static_cast<int>(CollectiveType::kReduceScatter),
            static_cast<int>(CollectiveType::kReduceScatterSingle),
            static_cast<int>(CollectiveType::kReduceScatterV),
            static_cast<int>(CollectiveType::kAllToAllSingle),
            static_cast<int>(CollectiveType::kAllToAll),
            static_cast<int>(CollectiveType::kBarrier),
            static_cast<int>(CollectiveType::kScatter),
            static_cast<int>(CollectiveType::kGather)),
        ::testing::Values(
            static_cast<int>(ExecMode::kEager),
            static_cast<int>(ExecMode::kMultiGraphSequential),
            static_cast<int>(ExecMode::kMultiGraphConcurrent))),
    [](const ::testing::TestParamInfo<std::tuple<int, int>>& info) {
      return CollectiveTimeoutTest::collectiveTypeName(
                 static_cast<CollectiveType>(std::get<0>(info.param))) +
          "_" +
          torch::comms::test::TimeoutTestHelper::execModeName(
                 static_cast<ExecMode>(std::get<1>(info.param)));
    });

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
