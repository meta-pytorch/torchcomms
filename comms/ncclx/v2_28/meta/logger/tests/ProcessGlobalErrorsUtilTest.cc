// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdlib>

#include <gtest/gtest.h>

#include "ScubaLoggerTestMixin.h"
#include "comms/ctran/utils/ErrorStackTraceUtil.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"
#include "param.h" // @manual

class ProcessGlobalErrorsUtilTest : public ::testing::Test,
                                    public ScubaLoggerTestMixin {
 public:
  void SetUp() override {
    ScubaLoggerTestMixin::SetUp();
    setenv("NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES", "20", 1);
    initEnv();
  }
};

TEST_F(ProcessGlobalErrorsUtilTest, SetNic) {
  auto state1 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_TRUE(state1.badNics.empty());

  ProcessGlobalErrorsUtil::setNic("beth0", 8000, "bad");
  ProcessGlobalErrorsUtil::setNic("beth1", 8000, "bad");
  ProcessGlobalErrorsUtil::setNic("beth0", 8001, "bad");

  auto state2 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_TRUE(state2.badNics.contains("beth0"));
  ASSERT_TRUE(state2.badNics["beth0"].contains(8000));
  ASSERT_TRUE(state2.badNics["beth0"].contains(8001));
  ASSERT_TRUE(state2.badNics.contains("beth1"));
  ASSERT_TRUE(state2.badNics["beth1"].contains(8000));
  for (const auto& [device, portMap] : state2.badNics) {
    for (const auto& [port, nicError] : portMap) {
      ASSERT_GT(nicError.timestampMs.count(), 0);
      ASSERT_EQ(nicError.errorMessage, "bad");
    }
  }

  ProcessGlobalErrorsUtil::setNic("beth0", 8000, std::nullopt);
  ProcessGlobalErrorsUtil::setNic("beth1", 8000, std::nullopt);
  ProcessGlobalErrorsUtil::setNic("beth0", 8001, std::nullopt);

  auto state3 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_TRUE(state3.badNics.empty());
}

TEST_F(ProcessGlobalErrorsUtilTest, AddErrorAndStackTrace) {
  auto state1 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_TRUE(state1.errorAndStackTraces.empty());

  std::vector<std::string> stackTrace{"s1", "s2"};

  // Ensure we keep only the last N errors
  for (int i = 0; i < NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES * 2; ++i) {
    ProcessGlobalErrorsUtil::addErrorAndStackTrace("error", stackTrace);
  }

  auto state2 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_EQ(
      NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES,
      state2.errorAndStackTraces.size());
  for (const auto& errorAndStackTrace : state2.errorAndStackTraces) {
    ASSERT_GT(errorAndStackTrace.timestampMs.count(), 0);
    ASSERT_EQ("error", errorAndStackTrace.errorMessage);
    ASSERT_EQ(stackTrace, errorAndStackTrace.stackTrace);
  }
}

TEST_F(ProcessGlobalErrorsUtilTest, LogStackTraceOnErrorReturn) {
  ErrorStackTraceUtil::log(commInternalError);
  auto state1 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_EQ(1, state1.errorAndStackTraces.size());

  ErrorStackTraceUtil::logErrorMessage("error3");

  auto state2 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_EQ(2, state2.errorAndStackTraces.size());
}
