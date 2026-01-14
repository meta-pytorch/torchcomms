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
  auto dummyFn = [](int rank, uint64_t commHash, std::string commDesc) {
    FB_COMMCHECKTHROW_EX(commSystemError, rank, commHash, commDesc);
    return commSuccess;
  };

  const auto rank = 1;
  const auto commHash = 0x1234;
  const std::string commDesc = "testDesc";
  bool caughtException = false;
  try {
    dummyFn(rank, commHash, commDesc);
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

TEST_F(CtranUtilsCheckTest, FB_SYSCHECKTHROW_EX_DIRECT) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "testDesc";

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_SYSCHECKTHROW_EX_DIRECT(0, rank, commHash, desc));

  // Failure case: ctran::utils::Exception thrown with correct properties
  bool caughtException = false;
  try {
    FB_SYSCHECKTHROW_EX_DIRECT(EINVAL, rank, commHash, desc);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("System error:"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}

TEST_F(CtranUtilsCheckTest, FB_SYSCHECKTHROW_EX_LOGDATA) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "testDesc";

  CommLogData logData = {
      .rank = rank,
      .commHash = commHash,
      .commDesc = desc,
  };

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_SYSCHECKTHROW_EX_LOGDATA(cudaSuccess, logData));

  // Failure case: ctran::utils::Exception thrown with correct properties
  bool caughtException = false;
  try {
    FB_SYSCHECKTHROW_EX_LOGDATA(cudaErrorInvalidValue, logData);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("System error:"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}

TEST_F(CtranUtilsCheckTest, FB_SYSCHECKTHROW_EX_4ARGS) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "testDesc";

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_SYSCHECKTHROW_EX(0, rank, commHash, desc));

  // Failure case: ctran::utils::Exception thrown with correct properties
  bool caughtException = false;
  try {
    FB_SYSCHECKTHROW_EX(EINVAL, rank, commHash, desc);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("System error:"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}

TEST_F(CtranUtilsCheckTest, FB_SYSCHECKTHROW_EX_2ARGS) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "testDesc";

  CommLogData logData = {
      .rank = rank,
      .commHash = commHash,
      .commDesc = desc,
  };

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_SYSCHECKTHROW_EX(cudaSuccess, logData));

  // Failure case: ctran::utils::Exception thrown with correct properties
  bool caughtException = false;
  try {
    FB_SYSCHECKTHROW_EX(cudaErrorInvalidValue, logData);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("System error:"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}

namespace {
struct MockError {
  int errNum;
  std::string errStr;
};
} // namespace

TEST_F(CtranUtilsCheckTest, FOLLY_EXPECTED_CHECKTHROW_EX_NOCOMM) {
  // Success case: no exception thrown
  auto successResult = folly::Expected<int, MockError>(42);
  EXPECT_NO_THROW(FOLLY_EXPECTED_CHECKTHROW_EX_NOCOMM(successResult));

  // Failure case: ctran::utils::Exception thrown with correct properties
  // Failure case: ctran::utils::Exception thrown with correct properties
  auto errorResult = folly::Expected<int, MockError>(folly::makeUnexpected(
      MockError{
          .errNum = EINVAL,
          .errStr = "mock error message",
      }));

  bool caughtException = false;
  try {
    FOLLY_EXPECTED_CHECKTHROW_EX_NOCOMM(errorResult);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_THAT(
        std::string(e.what()), ::testing::HasSubstr("COMM internal failure:"));
    EXPECT_THAT(
        std::string(e.what()), ::testing::HasSubstr("mock error message"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}

TEST_F(CtranUtilsCheckTest, FOLLY_EXPECTED_CHECKTHROW_EX) {
  const int rank = 3;
  const uint64_t commHash = 0xCAFEBABE;
  const std::string desc = "testDesc";

  // Success case: no exception thrown
  auto successResult = folly::Expected<int, MockError>(42);
  EXPECT_NO_THROW(
      FOLLY_EXPECTED_CHECKTHROW_EX(successResult, rank, commHash, desc));

  // Failure case: ctran::utils::Exception thrown with correct properties
  auto errorResult = folly::Expected<int, MockError>(folly::makeUnexpected(
      MockError{
          .errNum = EINVAL,
          .errStr = "mock error message",
      }));

  bool caughtException = false;
  try {
    FOLLY_EXPECTED_CHECKTHROW_EX(errorResult, rank, commHash, desc);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_THAT(
        std::string(e.what()), ::testing::HasSubstr("COMM internal failure:"));
    EXPECT_THAT(
        std::string(e.what()), ::testing::HasSubstr("mock error message"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
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

TEST_F(CtranUtilsCheckTest, FB_COMMCHECKTHROW_EX_DIRECT) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string commDesc = "testDesc";

  // Success case: no exception thrown
  EXPECT_NO_THROW(
      FB_COMMCHECKTHROW_EX_DIRECT(commSuccess, rank, commHash, commDesc));

  for (size_t i = 0; i < commNumResults; i++) {
    const auto commResult = static_cast<commResult_t>(i);
    if (commResult == commSuccess || commResult == commInProgress) {
      continue;
    }

    // Failure case: ctran::utils::Exception thrown with correct properties
    bool caughtException = false;
    try {
      FB_COMMCHECKTHROW_EX_DIRECT(commResult, rank, commHash, commDesc);
    } catch (const ctran::utils::Exception& e) {
      EXPECT_EQ(e.result(), commResult);
      EXPECT_EQ(e.rank(), rank);
      EXPECT_EQ(e.commHash(), commHash);
      EXPECT_THAT(
          std::string(e.what()),
          ::testing::HasSubstr("COMM internal failure:"));
      caughtException = true;
    }
    ASSERT_TRUE(caughtException)
        << "Expected ctran::utils::Exception for commResult=" << commResult;
  }
}

TEST_F(CtranUtilsCheckTest, FB_COMMCHECKTHROW_EX_LOGDATA) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string commDesc = "testDesc";

  CommLogData logData = {
      .rank = rank,
      .commHash = commHash,
      .commDesc = commDesc,
  };

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_COMMCHECKTHROW_EX_LOGDATA(commSuccess, logData));

  for (size_t i = 0; i < commNumResults; i++) {
    const auto commResult = static_cast<commResult_t>(i);
    if (commResult == commSuccess || commResult == commInProgress) {
      continue;
    }

    // Failure case: ctran::utils::Exception thrown with correct properties
    bool caughtException = false;
    try {
      FB_COMMCHECKTHROW_EX_LOGDATA(commResult, logData);
    } catch (const ctran::utils::Exception& e) {
      EXPECT_EQ(e.result(), commResult);
      EXPECT_EQ(e.rank(), rank);
      EXPECT_EQ(e.commHash(), commHash);
      EXPECT_THAT(
          std::string(e.what()),
          ::testing::HasSubstr("COMM internal failure:"));
      caughtException = true;
    }
    ASSERT_TRUE(caughtException)
        << "Expected ctran::utils::Exception for commResult=" << commResult;
  }
}

TEST_F(CtranUtilsCheckTest, FB_COMMCHECKTHROW_EX_3ARGS) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string commDesc = "testDesc";

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_COMMCHECKTHROW_EX(commSuccess, rank, commHash, commDesc));

  for (size_t i = 0; i < commNumResults; i++) {
    const auto commResult = static_cast<commResult_t>(i);
    if (commResult == commSuccess || commResult == commInProgress) {
      continue;
    }

    // Failure case: ctran::utils::Exception thrown with correct properties
    bool caughtException = false;
    try {
      FB_COMMCHECKTHROW_EX(commResult, rank, commHash, commDesc);
    } catch (const ctran::utils::Exception& e) {
      EXPECT_EQ(e.result(), commResult);
      EXPECT_EQ(e.rank(), rank);
      EXPECT_EQ(e.commHash(), commHash);
      EXPECT_THAT(
          std::string(e.what()),
          ::testing::HasSubstr("COMM internal failure:"));
      caughtException = true;
    }
    ASSERT_TRUE(caughtException)
        << "Expected ctran::utils::Exception for commResult=" << commResult;
  }
}

TEST_F(CtranUtilsCheckTest, FB_COMMCHECKTHROW_EX_2ARGS) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string commDesc = "testDesc";

  CommLogData logData = {
      .rank = rank,
      .commHash = commHash,
      .commDesc = commDesc,
  };

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_COMMCHECKTHROW_EX(commSuccess, logData));

  for (size_t i = 0; i < commNumResults; i++) {
    const auto commResult = static_cast<commResult_t>(i);
    if (commResult == commSuccess || commResult == commInProgress) {
      continue;
    }

    // Failure case: ctran::utils::Exception thrown with correct properties
    bool caughtException = false;
    try {
      FB_COMMCHECKTHROW_EX(commResult, logData);
    } catch (const ctran::utils::Exception& e) {
      EXPECT_EQ(e.result(), commResult);
      EXPECT_EQ(e.rank(), rank);
      EXPECT_EQ(e.commHash(), commHash);
      EXPECT_THAT(
          std::string(e.what()),
          ::testing::HasSubstr("COMM internal failure:"));
      caughtException = true;
    }
    ASSERT_TRUE(caughtException)
        << "Expected ctran::utils::Exception for commResult=" << commResult;
  }
}

TEST_F(CtranUtilsCheckTest, FB_COMMCHECKTHROW_EX_NOCOMM) {
  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_COMMCHECKTHROW_EX_NOCOMM(commSuccess));

  for (size_t i = 0; i < commNumResults; i++) {
    const auto commResult = static_cast<commResult_t>(i);
    if (commResult == commSuccess || commResult == commInProgress) {
      continue;
    }

    // Failure case: ctran::utils::Exception thrown with correct properties
    bool caughtException = false;
    try {
      FB_COMMCHECKTHROW_EX_NOCOMM(commResult);
    } catch (const ctran::utils::Exception& e) {
      EXPECT_EQ(e.result(), commResult);
      EXPECT_THAT(
          std::string(e.what()),
          ::testing::HasSubstr("COMM internal failure:"));
      caughtException = true;
    }
    ASSERT_TRUE(caughtException)
        << "Expected ctran::utils::Exception for commResult=" << commResult;
  }
}

TEST_F(CtranUtilsCheckTest, FB_CUDACHECKTHROW_EX_DIRECT) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "testDesc";

  // Success case: no exception thrown
  EXPECT_NO_THROW(
      FB_CUDACHECKTHROW_EX_DIRECT(cudaSuccess, rank, commHash, desc));

  // Failure case: ctran::utils::Exception thrown with correct properties
  bool caughtException = false;
  try {
    FB_CUDACHECKTHROW_EX_DIRECT(cudaErrorInvalidValue, rank, commHash, desc);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_EQ(e.result(), cudaErrorInvalidValue);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("Cuda failure:"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}

TEST_F(CtranUtilsCheckTest, FB_CUDACHECKTHROW_EX_LOGDATA) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "testDesc";

  CommLogData logData = {
      .rank = rank,
      .commHash = commHash,
      .commDesc = desc,
  };

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_CUDACHECKTHROW_EX_LOGDATA(cudaSuccess, logData));

  // Failure case: ctran::utils::Exception thrown with correct properties
  bool caughtException = false;
  try {
    FB_CUDACHECKTHROW_EX_LOGDATA(cudaErrorInvalidValue, logData);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_EQ(e.result(), cudaErrorInvalidValue);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("Cuda failure:"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}

TEST_F(CtranUtilsCheckTest, FB_CUDACHECKTHROW_EX_3ARGS) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "testDesc";

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_CUDACHECKTHROW_EX(cudaSuccess, rank, commHash, desc));

  // Failure case: ctran::utils::Exception thrown with correct properties
  bool caughtException = false;
  try {
    FB_CUDACHECKTHROW_EX(cudaErrorInvalidValue, rank, commHash, desc);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_EQ(e.result(), cudaErrorInvalidValue);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("Cuda failure:"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}

TEST_F(CtranUtilsCheckTest, FB_CUDACHECKTHROW_EX_2ARGS) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "testDesc";

  CommLogData logData = {
      .rank = rank,
      .commHash = commHash,
      .commDesc = desc,
  };

  // Success case: no exception thrown
  EXPECT_NO_THROW(FB_CUDACHECKTHROW_EX(cudaSuccess, logData));

  // Failure case: ctran::utils::Exception thrown with correct properties
  bool caughtException = false;
  try {
    FB_CUDACHECKTHROW_EX(cudaErrorInvalidValue, logData);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_EQ(e.result(), cudaErrorInvalidValue);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("Cuda failure:"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}

TEST_F(
    CtranUtilsCheckTest,
    FB_CUDACHECKTHROW_EX_NOCOMM) { // Success case: no exception thrown
  EXPECT_NO_THROW(FB_CUDACHECKTHROW_EX_NOCOMM(cudaSuccess));

  // Failure case: ctran::utils::Exception thrown with correct properties
  bool caughtException = false;
  try {
    FB_CUDACHECKTHROW_EX_NOCOMM(cudaErrorInvalidValue);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_EQ(e.result(), cudaErrorInvalidValue);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("Cuda failure:"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
}
TEST_F(CtranUtilsCheckTest, FB_ERRORTHROW_EX) {
  const int rank = 7;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string commDesc = "testDesc";

  CommLogData logData = {
      .rank = rank,
      .commHash = commHash,
      .commDesc = commDesc,
  };

  for (size_t i = 0; i < commNumResults; i++) {
    const auto commResult = static_cast<commResult_t>(i);
    if (commResult == commSuccess || commResult == commInProgress) {
      continue;
    }

    auto dummyFn = [logData, commResult]() {
      FB_ERRORTHROW_EX(
          commResult, logData, "test FB_ERRORTHROW_EX - no failure");
      return commSuccess;
    };

    bool caughtException = false;
    try {
      dummyFn();
    } catch (const ctran::utils::Exception& e) {
      EXPECT_EQ(e.rank(), rank);
      EXPECT_EQ(e.commHash(), commHash);
      EXPECT_EQ(e.result(), commResult);
      EXPECT_THAT(
          std::string(e.what()),
          ::testing::HasSubstr("COMM internal failure:"));
      caughtException = true;
    }

    ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
  }
}

TEST_F(CtranUtilsCheckTest, FB_ERRORTHROW_EX_NOCOMM) {
  for (size_t i = 0; i < commNumResults; i++) {
    const auto commResult = static_cast<commResult_t>(i);
    if (commResult == commSuccess || commResult == commInProgress) {
      continue;
    }

    auto dummyFn = [commResult]() {
      FB_ERRORTHROW_EX_NOCOMM(
          commResult, "test FB_ERRORTHROW_EX_NOCOMM - no failure");
      return commSuccess;
    };

    bool caughtException = false;
    try {
      dummyFn();
    } catch (const ctran::utils::Exception& e) {
      EXPECT_THAT(
          std::string(e.what()),
          ::testing::HasSubstr("COMM internal failure:"));
      EXPECT_EQ(e.result(), commResult);
      EXPECT_EQ(e.result(), commResult);
      caughtException = true;
    }

    ASSERT_TRUE(caughtException) << "Expected ctran::utils::Exception";
  }
}
