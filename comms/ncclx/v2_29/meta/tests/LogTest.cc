// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <filesystem>
#include <fstream>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/Logger.h"
#include "debug.h"

class LogTest : public ::testing::Test {
 public:
  LogTest() = default;

  void finishLogging() {
    sleep(1); // wait for logging to finish
    NcclLogger::close();
  }

  void initLogging() {
    ncclDebugLevel = -1;
    ncclDebugFile = nullptr;
    initNcclLogger();
  }
};

TEST_F(LogTest, Info) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("INFO"));
  initLogging();
  const std::string kTestStr = "Testing INFO";

  testing::internal::CaptureStdout();
  INFO(NCCL_ALL, "%s", kTestStr.c_str());
  finishLogging();
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(LogTest, LongInfo) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("INFO"));
  initLogging();
  std::string kTestStr = "Testing long INFO,";

  // prepare log longer than 1024 chars as statically set previously
  do {
    kTestStr += kTestStr;
  } while (kTestStr.size() < 3000);

  testing::internal::CaptureStdout();
  INFO(NCCL_ALL, "%s", kTestStr.c_str());
  finishLogging();
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(LogTest, Warn) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("WARN"));
  const std::string kTestStr = "Testing WARN";
  initLogging();
  testing::internal::CaptureStdout();
  WARN("%s", kTestStr.c_str());
  finishLogging();
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(LogTest, LongWarn) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("WARN"));
  std::string kTestStr = "Testing long WARN,";
  initLogging();
  // prepare log longer than 1024 chars as statically set previously
  do {
    kTestStr += kTestStr;
  } while (kTestStr.size() < 3000);

  testing::internal::CaptureStdout();
  WARN("%s", kTestStr.c_str());
  finishLogging();
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(LogTest, InfoOff) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("WARN"));
  const std::string kTestStr = "Testing INFO when NCCL_DEBUG=WARN";
  initLogging();
  testing::internal::CaptureStdout();
  INFO(NCCL_ALL, "%s", kTestStr.c_str());
  finishLogging();
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, Not(testing::HasSubstr(kTestStr)));
}

TEST_F(LogTest, InfoToFile) {
  const std::string kDebugFile = getTestFilePath("nccl_logtest", ".log");
  const std::string kTestStr = "Testing INFO to FILE";
  auto envGuard0 = EnvRAII(NCCL_DEBUG_FILE, kDebugFile);
  auto envGuard1 = EnvRAII(NCCL_DEBUG, std::string("INFO"));
  initLogging();
  INFO(NCCL_ALL, "%s", kTestStr.c_str());
  finishLogging();
  // check file exists and has the expected log
  EXPECT_TRUE(std::filesystem::exists(kDebugFile));
  std::ifstream filein(kDebugFile);
  std::string fileContent(
      (std::istreambuf_iterator<char>(filein)),
      std::istreambuf_iterator<char>());
  EXPECT_THAT(fileContent, testing::HasSubstr(kTestStr));

  std::filesystem::remove(kDebugFile);
}

TEST_F(LogTest, InfoWarnToFile) {
  const std::string kDebugFile = getTestFilePath("nccl_logtest", ".log");
  const std::string kTestInfoStr = "Testing INFO to FILE";
  const std::string kTestWarnStr = "Testing WARN to FILE";

  XLOG(WARN) << kDebugFile;
  auto envGuard0 = EnvRAII(NCCL_DEBUG_FILE, kDebugFile);
  auto envGuard1 = EnvRAII(NCCL_DEBUG, std::string("INFO"));

  initLogging();
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();

  INFO(NCCL_ALL, "%s", kTestInfoStr.c_str());
  WARN("%s", kTestWarnStr.c_str());

  finishLogging();
  std::string stdoutContent = testing::internal::GetCapturedStdout();
  std::string stderrContent = testing::internal::GetCapturedStderr();

  // check both INFO and WARN logs are in the file
  EXPECT_TRUE(std::filesystem::exists(kDebugFile));
  std::ifstream filein(kDebugFile);
  std::string fileContent(
      (std::istreambuf_iterator<char>(filein)),
      std::istreambuf_iterator<char>());
  EXPECT_THAT(fileContent, testing::HasSubstr(kTestInfoStr));
  EXPECT_THAT(fileContent, testing::HasSubstr(kTestWarnStr));

  // check INFO is NOT printed to stdout, and WARN is printed to stderr
  EXPECT_THAT(stdoutContent, Not(testing::HasSubstr(kTestInfoStr)));
  EXPECT_THAT(stderrContent, testing::HasSubstr(kTestWarnStr));

  std::filesystem::remove(kDebugFile);
}
