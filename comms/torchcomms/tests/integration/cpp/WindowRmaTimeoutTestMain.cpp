// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Timeout test for Window RMA operations (wait_signal, put+wait_signal).
// Gated by TEST_BACKEND=ncclx and NCCL_CTRAN_ENABLE=true.
// - Timeout logic: same as CollectiveTimeoutTest — rank 0 only runs
//   warmup ops then sleeps, so rank 1's final op times out.
// - Call path: TEST_P → testTimeout → helper_.launch(childBody) where
//   childBody builds an ops vector and calls helper_.exec(mode, ops).
//   exec() handles eager vs graph mode dispatch transparently.

#include <gtest/gtest.h>
#include "WindowRmaTimeoutTest.hpp"

using ExecMode = torch::comms::test::TimeoutTestHelper::ExecMode;
using RmaType = WindowRmaTimeoutTest::RmaType;

TEST_P(WindowRmaTimeoutTest, Timeout) {
  const auto [type_int, mode_int] = GetParam();
  testTimeout(static_cast<RmaType>(type_int), static_cast<ExecMode>(mode_int));
}

INSTANTIATE_TEST_SUITE_P(
    WindowRma,
    WindowRmaTimeoutTest,
    ::testing::Combine(
        ::testing::Values(
            static_cast<int>(RmaType::kWaitSignal),
            static_cast<int>(RmaType::kPutWaitSignal)),
        ::testing::Values(
            static_cast<int>(ExecMode::kEager),
            static_cast<int>(ExecMode::kMultiGraphSequential),
            static_cast<int>(ExecMode::kMultiGraphConcurrent))),
    [](const ::testing::TestParamInfo<std::tuple<int, int>>& info) {
      return WindowRmaTimeoutTest::rmaTypeName(
                 static_cast<RmaType>(std::get<0>(info.param))) +
          "_" +
          torch::comms::test::TimeoutTestHelper::execModeName(
                 static_cast<ExecMode>(std::get<1>(info.param)));
    });

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
