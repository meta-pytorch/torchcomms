// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/Debug.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>

#include "comms/ctran/utils/LogInit.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace testing;
using namespace ncclx;

// Test fixture for Debug tests
class DebugTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ncclDebugSave = NCCL_DEBUG;
    ncclDebugSubsysSave = NCCL_DEBUG_SUBSYS;
    NCCL_DEBUG = "INFO";
    NCCL_DEBUG_SUBSYS = "ALL";
    ctran::logging::initCtranLogging();
  }

  void TearDown() override {
    NCCL_DEBUG = ncclDebugSave;
    NCCL_DEBUG_SUBSYS = ncclDebugSubsysSave;
  }

  std::string ncclDebugSave = "";
  std::string ncclDebugSubsysSave = "";
};

// Test commNamedThreadStart function
TEST_F(DebugTest, CommNamedThreadStart) {
  // Since this function primarily logs, we can only verify it doesn't crash
  // A more comprehensive test would require capturing log output
  EXPECT_NO_THROW(commNamedThreadStart("TestThread"));
}

constexpr int kRank = 42;
constexpr uint64_t kCommHash = 205;
constexpr std::string_view kCommDesc = "UTComm";

// Test commNamedThreadStart function with arguments
TEST_F(DebugTest, CommNamedThreadStartExtWithArgs) {
  testing::internal::CaptureStdout();

  // we know kCommDesc is null terminated here.
  EXPECT_NO_THROW(
      commNamedThreadStart("TestThread", kRank, kCommHash, kCommDesc.data()));

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, HasSubstr(fmt::format("rank {}", kRank)));
  EXPECT_THAT(output, HasSubstr(fmt::format("commHash {:x}", kCommHash)));
  EXPECT_THAT(output, HasSubstr(fmt::format("commDesc {}", kCommDesc)));
}

// Test commNamedThreadStart(Ext) without args
TEST_F(DebugTest, CommNamedThreadStartNoArgs) {
  // This test is a placeholder - in a real environment, we would need to
  // verify that meta::comms::logger::initThreadMetaData is called with the
  // correct parameters This would typically require a mock or a test-specific
  // implementation of the logger

  // For now, we just verify the functions don't crash
  EXPECT_NO_THROW(commNamedThreadStart("TestThread1"));
}
