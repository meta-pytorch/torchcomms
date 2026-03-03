// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fmt/core.h>
#include <folly/Conv.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "nccl.h" // @manual

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/NcclxInfo.h" // @manual

TEST(NcclxInfoUT, TestBasicInfo) {
  auto info = ncclx::getNcclxInfo();

  EXPECT_TRUE(info->contains("ncclx_version"));

  EXPECT_THAT(
      info->at("ncclx_version"),
      ::testing::HasSubstr(
          fmt::format("{}.{}.{}", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH)));
}

TEST(NcclxInfoUT, TestCollTraceSupport) {
  auto info = ncclx::getNcclxInfo();
  EXPECT_TRUE(info->contains("colltrace_supports_check_async_error"));
  EXPECT_EQ(
      folly::to<bool>(info->at("colltrace_supports_check_async_error")), true);
}

TEST(NcclxInfoUT, TestCollTrace) {
  // Use getNcclxInfo once to initialize
  ncclx::getNcclxInfo();

  auto guard = EnvRAII(NCCL_COLLTRACE, {});
  // Tests may be run in the same process, so we need to gather the info again
  auto info = ncclx::testOnlyGatherNcclxInfo();
  EXPECT_TRUE(info.contains("colltrace_enabled"));

  EXPECT_EQ(folly::to<bool>(info.at("colltrace_enabled")), false);
}

TEST(NcclxInfoUT, TestCollTraceEnabled) {
  // Run getNcclxInfo once to initialize
  ncclx::getNcclxInfo();

  auto guard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto info = ncclx::testOnlyGatherNcclxInfo();
  EXPECT_TRUE(info.contains("colltrace_enabled"));

  EXPECT_EQ(folly::to<bool>(info.at("colltrace_enabled")), true);
}

TEST(NcclxInfoUT, TestNewCollTraceEnabled) {
  // Run getNcclxInfo once to initialize
  ncclx::getNcclxInfo();

  auto guard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto newGuard = EnvRAII(NCCL_COLLTRACE_USE_NEW_COLLTRACE, true);
  auto info = ncclx::testOnlyGatherNcclxInfo();
  EXPECT_TRUE(info.contains("colltrace_enabled"));
  EXPECT_TRUE(info.contains("colltrace_new_colltrace"));

  EXPECT_EQ(folly::to<bool>(info.at("colltrace_enabled")), true);
  EXPECT_EQ(folly::to<bool>(info.at("colltrace_new_colltrace")), true);
}

TEST(NcclxInfoUT, TestWatchdogSupport) {
  // Run getNcclxInfo once to initialize
  ncclx::getNcclxInfo();

  auto guard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto newGuard = EnvRAII(NCCL_COLLTRACE_USE_NEW_COLLTRACE, true);
  auto info = ncclx::testOnlyGatherNcclxInfo();
  EXPECT_TRUE(info.contains("colltrace_enabled"));
  EXPECT_TRUE(info.contains("colltrace_new_colltrace"));
  EXPECT_TRUE(info.contains("colltrace_supports_check_timeout"));

  EXPECT_EQ(folly::to<bool>(info.at("colltrace_enabled")), true);
  EXPECT_EQ(folly::to<bool>(info.at("colltrace_new_colltrace")), true);
  EXPECT_EQ(folly::to<bool>(info.at("colltrace_supports_check_timeout")), true);
}

TEST(NcclxInfoUT, TestWatchdogSupportNoNewCollTrace) {
  // Run getNcclxInfo once to initialize
  ncclx::getNcclxInfo();

  auto guard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto newGuard = EnvRAII(NCCL_COLLTRACE_USE_NEW_COLLTRACE, false);

  auto info = ncclx::testOnlyGatherNcclxInfo();
  EXPECT_TRUE(info.contains("colltrace_enabled"));
  EXPECT_TRUE(info.contains("colltrace_new_colltrace"));
  EXPECT_TRUE(info.contains("colltrace_supports_check_timeout"));

  EXPECT_EQ(folly::to<bool>(info.at("colltrace_enabled")), true);
  EXPECT_EQ(folly::to<bool>(info.at("colltrace_new_colltrace")), false);
  EXPECT_EQ(
      folly::to<bool>(info.at("colltrace_supports_check_timeout")), false);
}
