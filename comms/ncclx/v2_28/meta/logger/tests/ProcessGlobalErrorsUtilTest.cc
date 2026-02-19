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

TEST_F(ProcessGlobalErrorsUtilTest, AddIbCompletionError) {
  auto state1 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_TRUE(state1.ibCompletionErrors.empty());

  // Ensure we keep only the last N errors
  for (int i = 0; i < NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES * 2; ++i) {
    ProcessGlobalErrorsUtil::IbCompletionError ibErr;
    ibErr.peer = "peer_addr";
    ibErr.statusStr = "IBV_WC_REM_ACCESS_ERR";
    ibErr.status = 5;
    ibErr.opcodeStr = "IBV_WC_RDMA_WRITE";
    ibErr.opcode = 0;
    ibErr.reqSize = 1024;
    ibErr.vendorErr = 42;
    ibErr.reqType = "send";
    ibErr.localGid = "fe80::1";
    ibErr.remoteGid = "fe80::2";
    ibErr.hcaName = "mlx5_0";
    ProcessGlobalErrorsUtil::addIbCompletionError(std::move(ibErr));
  }

  auto state2 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_EQ(
      NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES,
      state2.ibCompletionErrors.size());
  for (const auto& ibErr : state2.ibCompletionErrors) {
    ASSERT_GT(ibErr.timestampMs.count(), 0);
    ASSERT_EQ("peer_addr", ibErr.peer);
    ASSERT_EQ("IBV_WC_REM_ACCESS_ERR", ibErr.statusStr);
    ASSERT_EQ(5, ibErr.status);
    ASSERT_EQ("IBV_WC_RDMA_WRITE", ibErr.opcodeStr);
    ASSERT_EQ(0, ibErr.opcode);
    ASSERT_EQ(1024, ibErr.reqSize);
    ASSERT_EQ(42u, ibErr.vendorErr);
    ASSERT_EQ("send", ibErr.reqType);
    ASSERT_EQ("fe80::1", ibErr.localGid);
    ASSERT_EQ("fe80::2", ibErr.remoteGid);
    ASSERT_EQ("mlx5_0", ibErr.hcaName);
  }
}

TEST_F(ProcessGlobalErrorsUtilTest, AddIbCompletionErrorNoGids) {
  ProcessGlobalErrorsUtil::IbCompletionError ibErr;
  ibErr.peer = "peer_addr";
  ibErr.statusStr = "IBV_WC_RETRY_EXC_ERR";
  ibErr.status = 12;
  ibErr.opcodeStr = "IBV_WC_SEND";
  ibErr.opcode = 1;
  ibErr.reqSize = 512;
  ibErr.vendorErr = 0;
  ibErr.reqType = "recv";
  ibErr.hcaName = "mlx5_1";
  // localGid and remoteGid intentionally left empty
  ProcessGlobalErrorsUtil::addIbCompletionError(std::move(ibErr));

  auto state = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_EQ(1, state.ibCompletionErrors.size());
  const auto& stored = state.ibCompletionErrors.front();
  ASSERT_GT(stored.timestampMs.count(), 0);
  ASSERT_EQ("peer_addr", stored.peer);
  ASSERT_EQ("", stored.localGid);
  ASSERT_EQ("", stored.remoteGid);
  ASSERT_EQ("mlx5_1", stored.hcaName);
}

TEST_F(ProcessGlobalErrorsUtilTest, ClearIbCompletionErrors) {
  // Add some errors
  for (int i = 0; i < 3; ++i) {
    ProcessGlobalErrorsUtil::IbCompletionError ibErr;
    ibErr.peer = "peer_addr";
    ibErr.statusStr = "IBV_WC_REM_ACCESS_ERR";
    ibErr.status = 5;
    ibErr.opcodeStr = "IBV_WC_RDMA_WRITE";
    ibErr.opcode = 0;
    ibErr.reqSize = 1024;
    ibErr.vendorErr = 0;
    ibErr.reqType = "send";
    ibErr.hcaName = "mlx5_0";
    ProcessGlobalErrorsUtil::addIbCompletionError(std::move(ibErr));
  }

  auto state1 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_EQ(3, state1.ibCompletionErrors.size());

  // Clear and verify empty
  ProcessGlobalErrorsUtil::clearIbCompletionErrors();
  auto state2 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_TRUE(state2.ibCompletionErrors.empty());

  // Verify other error types are unaffected
  ProcessGlobalErrorsUtil::setNic("beth0", 8000, "bad");
  ProcessGlobalErrorsUtil::addErrorAndStackTrace("error", {"s1"});

  ProcessGlobalErrorsUtil::IbCompletionError ibErr;
  ibErr.peer = "peer_addr";
  ibErr.statusStr = "IBV_WC_REM_ACCESS_ERR";
  ibErr.status = 5;
  ibErr.opcodeStr = "IBV_WC_RDMA_WRITE";
  ibErr.opcode = 0;
  ibErr.reqSize = 1024;
  ibErr.vendorErr = 0;
  ibErr.reqType = "send";
  ibErr.hcaName = "mlx5_0";
  ProcessGlobalErrorsUtil::addIbCompletionError(std::move(ibErr));

  ProcessGlobalErrorsUtil::clearIbCompletionErrors();

  auto state3 = ProcessGlobalErrorsUtil::getAllState();
  ASSERT_TRUE(state3.ibCompletionErrors.empty());
  ASSERT_FALSE(state3.badNics.empty());
  ASSERT_FALSE(state3.errorAndStackTraces.empty());
}
