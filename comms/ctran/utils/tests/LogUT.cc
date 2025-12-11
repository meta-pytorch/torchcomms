// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <fmt/core.h>
#include <fmt/std.h>
#include <gtest/gtest.h>

#include <gmock/gmock.h>
#include "TestLogCategory.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/Logger.h"

class CtranUtilsLogTest : public ::testing::Test {
 public:
  CtranUtilsLogTest() = default;

  void SetUp() {
    ctran::logging::initCtranLogging(true /*alwaysInit*/);

    // Set up a test category
    auto* category =
        folly::LoggerDB::get().getCategory(XLOG_GET_CATEGORY_NAME());
    ASSERT_TRUE(testCategory_.setup(category));
  }

  void TearDown() {
    NcclLogger::close();
    testCategory_.reset();
  }

  const std::vector<std::string>& getMessages() const {
    return testCategory_.getMessages();
  }

  const int getCurrentGpuIndex() {
    int gpuIndex = -1;
    cudaGetDevice(&gpuIndex);
    return gpuIndex;
  }

  bool messageContainsGpuIndex(const std::string& message, int gpuIndex) const {
    std::string expectedPrefix = fmt::format("[{}]", gpuIndex);
    return message.find(expectedPrefix) != std::string::npos;
  }

 private:
  TestLogCategory testCategory_;
};

TEST_F(CtranUtilsLogTest, TestCLOGF) {
  CLOGF(INFO, "Test message with value: {}", 42);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);

  EXPECT_THAT(
      messages[0],
      ::testing::HasSubstr(fmt::format("[{}]", getCurrentGpuIndex())));

  std::string expectedContent = "Test message with value: 42";
  EXPECT_NE(messages[0].find(expectedContent), std::string::npos);
}

TEST_F(CtranUtilsLogTest, TestCLOGF_IF) {
  CLOGF_IF(INFO, true, "Conditional message with value: {}", 42);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);

  CLOGF_IF(INFO, false, "Conditional message with value: {}", 43);

  ASSERT_EQ(getMessages().size(), 1);

  EXPECT_THAT(
      messages[0],
      ::testing::HasSubstr(fmt::format("[{}]", getCurrentGpuIndex())));

  // Check that the message contains the formatted content
  std::string expectedContent = "Conditional message with value: 42";
  EXPECT_NE(messages[0].find(expectedContent), std::string::npos);
}

TEST_F(CtranUtilsLogTest, TestCLOGF_Threaded) {
  int tid0 = 0, tid1 = 0;
  std::thread t1([&tid0]() {
    CLOGF(INFO, "Test message with value: {}", 42);
    tid0 = syscall(SYS_gettid);
  });
  // Ensure the ordering of thread 1 and thread 2
  t1.join();
  std::thread t2([&tid1]() {
    CLOGF(INFO, "Test message with value: {}", 43);
    tid1 = syscall(SYS_gettid);
  });
  t2.join();

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 2);

  std::string hostname = ctran::utils::getHostname();
  int pid = getpid();

  EXPECT_THAT(
      messages[0],
      testing::HasSubstr(fmt::format("{}:{}:{}", hostname, pid, tid0)));
  EXPECT_THAT(
      messages[1],
      testing::HasSubstr(fmt::format("{}:{}:{}", hostname, pid, tid1)));
}

TEST_F(CtranUtilsLogTest, TestTraceOn) {
  EnvRAII env = EnvRAII(NCCL_CTRAN_ENABLE_TRACE_LOG, true);
  meta::comms::logger::setSubSystemMask(::meta::comms::logger::SubSystem::COLL);

  CLOGF_TRACE(COLL, "Conditional trace message with value: {}", 42);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);

  std::string expectedPrefix = "[TRACE]";
  std::string expectedContent = "Conditional trace message with value: 42";
  EXPECT_THAT(messages[0], testing::HasSubstr(expectedPrefix));
  EXPECT_THAT(messages[0], testing::HasSubstr(expectedContent));
}

TEST_F(CtranUtilsLogTest, TestTraceOff) {
  EnvRAII env = EnvRAII(NCCL_CTRAN_ENABLE_TRACE_LOG, false);
  meta::comms::logger::setSubSystemMask(::meta::comms::logger::SubSystem::COLL);

  CLOGF_TRACE(COLL, "Conditional trace message with value: {}", 42);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 0);
}

TEST_F(CtranUtilsLogTest, TestCLOGF_ENABLEDOneSystem) {
  // Enable only the ALLOC subsystem
  meta::comms::logger::setSubSystemMask(
      ::meta::comms::logger::SubSystem::ALLOC);

  // Test that CLOGF_ENABLED returns true for enabled subsystem
  EXPECT_TRUE(CLOGF_ENABLED(ALLOC));

  // Test that CLOGF_ENABLED returns false for disabled subsystem
  EXPECT_FALSE(CLOGF_ENABLED(COLL));
  EXPECT_FALSE(CLOGF_ENABLED(NET));

  // Test using CLOGF_ENABLED in conditional logging
  CLOGF_SUBSYS(INFO, ALLOC, "This msg should be logged {}", 42);
  CLOGF_SUBSYS(INFO, COLL, "This msg should NOT be logged {}", 43);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(
      messages[0],
      testing::HasSubstr(fmt::format("This msg should be logged {}", 42)));
}

TEST_F(CtranUtilsLogTest, TestCLOGF_ENABLEDTwoSystems) {
  // Test with multiple subsystems
  meta::comms::logger::setSubSystemMask(
      ::meta::comms::logger::SubSystem::ALLOC |
      ::meta::comms::logger::SubSystem::NET);

  EXPECT_TRUE(CLOGF_ENABLED(ALLOC));
  EXPECT_TRUE(CLOGF_ENABLED(NET));
  EXPECT_TRUE(CLOGF_ENABLED(NET | COLL));
  EXPECT_FALSE(CLOGF_ENABLED(COLL));
  EXPECT_FALSE(CLOGF_ENABLED(P2P));

  CLOGF_SUBSYS(
      INFO, ALLOC | COLL, "This should be logged (ALLOC enabled) {}", 44);

  CLOGF_SUBSYS(
      INFO, COLL | P2P, "This should NOT be logged (both disabled) {}", 45);

  const auto& messagesMulti = getMessages();
  ASSERT_EQ(messagesMulti.size(), 1);
  EXPECT_THAT(
      messagesMulti[0],
      testing::HasSubstr(
          fmt::format("This should be logged (ALLOC enabled) {}", 44)));
}

TEST_F(CtranUtilsLogTest, TestCLOGF_ENABLEDAllSystems) {
  // Test with multiple subsystems
  meta::comms::logger::setSubSystemMask(::meta::comms::logger::SubSystem::ALL);

  EXPECT_TRUE(CLOGF_ENABLED(ALLOC));
  EXPECT_TRUE(CLOGF_ENABLED(NET));
  EXPECT_TRUE(CLOGF_ENABLED(NET | COLL));

  CLOGF_SUBSYS(INFO, ALLOC, "This should be logged (All enabled) {}", 102);

  CLOGF_SUBSYS(INFO, COLL | P2P, "This should be logged {}", 103);

  const auto& messagesMulti = getMessages();
  ASSERT_EQ(messagesMulti.size(), 2);
  EXPECT_THAT(
      messagesMulti[0],
      testing::HasSubstr(
          fmt::format("This should be logged (All enabled) {}", 102)));

  EXPECT_THAT(
      messagesMulti[1],
      testing::HasSubstr(fmt::format("This should be logged {}", 103)));
}

TEST_F(CtranUtilsLogTest, TestLogPrefixCtran) {
  CLOGF(INFO, "Test message");

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);

  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN INFO"));
}
