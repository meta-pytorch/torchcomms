// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <fmt/core.h>
#include <fmt/std.h>
#include <gtest/gtest.h>

#include <gmock/gmock.h>
#include "TestLogCategory.h"
#include "comms/ctran/utils/ArgCheck.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/Logger.h"

class CtranUtilsCheckTest : public ::testing::Test {
 public:
  CtranUtilsCheckTest() = default;

  void SetUp() override {
    ctran::logging::initCtranLogging(true /*alwaysInit*/);

    // Set up a test category
    auto* category =
        folly::LoggerDB::get().getCategory(XLOG_GET_CATEGORY_NAME());
    ASSERT_TRUE(testCategory_.setup(category));
  }

  void TearDown() override {
    NcclLogger::close();
    testCategory_.reset();
  }

  const std::vector<std::string>& getMessages() const {
    return testCategory_.getMessages();
  }

  const int getCurrentGpuIndex() {
    int gpuIndex = -1;
    const auto res = cudaGetDevice(&gpuIndex);
    if (res != cudaSuccess) {
      return -1;
    }
    return gpuIndex;
  }

  bool messageContainsGpuIndex(const std::string& message, int gpuIndex) const {
    std::string expectedPrefix = fmt::format("[{}]", gpuIndex);
    return message.find(expectedPrefix) != std::string::npos;
  }

 private:
  TestLogCategory testCategory_;
};

TEST_F(CtranUtilsCheckTest, CudaCheck) {
  auto dummyFn = []() {
    FB_CUDACHECK(cudaErrorInvalidValue);
    return commSuccess;
  };

  auto res = dummyFn();
  ASSERT_EQ(res, commUnhandledCudaError);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN ERROR"));
  EXPECT_THAT(messages[0], ::testing::HasSubstr("invalid argument"));
}

TEST_F(CtranUtilsCheckTest, CudaCheckGoto) {
  auto dummyFn = []() {
    commResult_t res = commSuccess;
    FB_CUDACHECKGOTO(cudaErrorLaunchFailure, res, exit);
    return commSuccess;
  exit:
    return res;
  };
  auto res = dummyFn();
  ASSERT_EQ(res, commUnhandledCudaError);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN ERROR"));
  EXPECT_THAT(messages[0], ::testing::HasSubstr("unspecified launch failure"));
}

TEST_F(CtranUtilsCheckTest, CudaCheckIgnore) {
  auto dummyFn = []() {
    FB_CUDACHECKIGNORE(cudaErrorInvalidValue);
    return commSuccess;
  };
  auto res = dummyFn();
  ASSERT_EQ(res, commSuccess);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN WARN"));
  EXPECT_THAT(messages[0], ::testing::HasSubstr("invalid argument"));
}

TEST_F(CtranUtilsCheckTest, SysCheck) {
  auto dummyFn = []() {
    FB_SYSCHECK(-1, "sysTestFn");
    return commSuccess;
  };

  auto res = dummyFn();
  ASSERT_EQ(res, commSystemError);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN ERROR"));
  EXPECT_THAT(messages[0], ::testing::HasSubstr("Call to sysTestFn failed"));
}

TEST_F(CtranUtilsCheckTest, SysCheckGoto) {
  auto dummyFn = []() {
    commResult_t res = commSuccess;
    FB_SYSCHECKGOTO(-1, "sysTestFn", res, exit);
    return commSuccess;
  exit:
    return res;
  };
  auto res = dummyFn();
  ASSERT_EQ(res, commSystemError);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN ERROR"));
  EXPECT_THAT(messages[0], ::testing::HasSubstr("Call to sysTestFn failed"));
}

TEST_F(CtranUtilsCheckTest, ErrorReturn) {
  auto dummyFn = []() {
    FB_ERRORRETURN(commInvalidUsage, "test ErrorReturn failure");
    return commSuccess;
  };
  auto res = dummyFn();
  ASSERT_EQ(res, commInvalidUsage);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN ERROR"));
  EXPECT_THAT(messages[0], ::testing::HasSubstr("test ErrorReturn failure"));
}

TEST_F(CtranUtilsCheckTest, ErrorThrow) {
  auto dummyFn = []() {
    FB_ERRORTHROW(commInternalError, "test ErrorThrow failure");
    return commSuccess;
  };

  bool caughtException = false;
  try {
    dummyFn();
  } catch (const std::runtime_error& e) {
    auto errMsg = std::string(e.what());
    EXPECT_THAT(errMsg, ::testing::HasSubstr("COMM internal failure:"));
    auto errStr =
        std::string(::meta::comms::commCodeToString(commInternalError));
    EXPECT_THAT(errMsg, ::testing::HasSubstr(errStr));
    caughtException = true;
  }

  ASSERT_TRUE(caughtException) << "Expected std::runtime_error";

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN ERROR"));
  EXPECT_THAT(messages[0], ::testing::HasSubstr("test ErrorThrow failure"));
}

TEST_F(CtranUtilsCheckTest, ErrorThrowEx) {
  auto dummyFn = [](int rank, uint64_t commHash) {
    FB_COMMCHECKTHROW_EX(commSystemError, rank, commHash);
    return commSuccess;
  };

  const auto rank = 1;
  const auto commHash = 0x1234;
  bool caughtException = false;
  try {
    dummyFn(rank, commHash);
  } catch (const ctran::utils::Exception& e) {
    auto errMsg = std::string(e.what());
    EXPECT_THAT(errMsg, ::testing::HasSubstr("COMM internal failure:"));
    auto errStr = std::string(::meta::comms::commCodeToString(commSystemError));
    EXPECT_THAT(errMsg, ::testing::HasSubstr(errStr));
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    caughtException = true;
  }

  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN ERROR"));
}

TEST_F(CtranUtilsCheckTest, ArgCheckNull) {
  auto dummyFn = []() {
    ARGCHECK_NULL_COMM(nullptr, "ArgCheckNull ptr");
    return commSuccess;
  };
  auto res = dummyFn();
  ASSERT_EQ(res, commInvalidArgument);

  const auto& messages = getMessages();
  ASSERT_EQ(messages.size(), 1);
  EXPECT_THAT(messages[0], ::testing::HasSubstr("CTRAN ERROR"));
  EXPECT_THAT(
      messages[0], ::testing::HasSubstr("ArgCheckNull ptr argument is NULL"));
}

TEST_F(CtranUtilsCheckTest, FB_SYSCHECKTHROW) {
  EXPECT_NO_THROW(FB_SYSCHECKTHROW(0));
  EXPECT_THROW(FB_SYSCHECKTHROW(1), std::runtime_error);
}

TEST_F(CtranUtilsCheckTest, FB_CHECKTHROW) {
  EXPECT_NO_THROW(FB_CHECKTHROW(true, "test FB_CHECKTHROW -> NO throw"));
  EXPECT_THROW(
      FB_CHECKTHROW(false, "test FB_CHECKTHROW -> throw"), std::runtime_error);
}

TEST_F(CtranUtilsCheckTest, FB_SYSCHECKRETURN) {
  const int NOERROR = 0;
  const int ERRORCODE1 = 1;
  const int ERRORCODE2 = 1;

  auto testFn = [](const bool makeError) {
    const int commandErrVal = makeError ? ERRORCODE1 : NOERROR;
    FB_SYSCHECKRETURN(commandErrVal, ERRORCODE2);
    return NOERROR;
  };

  EXPECT_EQ(testFn(false), NOERROR);
  EXPECT_EQ(testFn(true), ERRORCODE2);
}
