// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <gmock/gmock.h>
#include "comms/utils/checks.h"
#include "comms/utils/commSpecs.h"

class CommsUtilsCheckTest : public ::testing::Test {};

TEST_F(CommsUtilsCheckTest, FB_CHECKTHROW_Success) {
  EXPECT_NO_THROW(FB_CHECKTHROW(true, "test FB_CHECKTHROW -> NO throw"));
}

TEST_F(CommsUtilsCheckTest, FB_CHECKTHROW_Failure) {
  EXPECT_THROW(
      FB_CHECKTHROW(false, "test FB_CHECKTHROW -> throw"), std::runtime_error);
}

TEST_F(CommsUtilsCheckTest, FB_CHECKTHROW_ExceptionMessage) {
  bool caughtException = false;
  try {
    FB_CHECKTHROW(false, "test FB_CHECKTHROW message");
  } catch (const std::runtime_error& e) {
    auto errMsg = std::string(e.what());
    EXPECT_THAT(errMsg, ::testing::HasSubstr("Check failed:"));
    EXPECT_THAT(errMsg, ::testing::HasSubstr("test FB_CHECKTHROW message"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected std::runtime_error";
}

TEST_F(CommsUtilsCheckTest, FB_ERRORTHROW) {
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
}

TEST_F(CommsUtilsCheckTest, FB_COMMCHECKTHROW_Success) {
  auto dummyFn = []() {
    FB_COMMCHECKTHROW(commSuccess);
    return true;
  };
  EXPECT_NO_THROW(dummyFn());
}

TEST_F(CommsUtilsCheckTest, FB_COMMCHECKTHROW_InProgress) {
  auto dummyFn = []() {
    FB_COMMCHECKTHROW(commInProgress);
    return true;
  };
  EXPECT_NO_THROW(dummyFn());
}

TEST_F(CommsUtilsCheckTest, FB_COMMCHECKTHROW_Failure) {
  auto dummyFn = []() {
    FB_COMMCHECKTHROW(commInternalError);
    return true;
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
}

namespace {
struct MockError {
  int errNum;
  std::string errStr;
};
} // namespace

TEST_F(CommsUtilsCheckTest, FOLLY_EXPECTED_CHECKTHROW_Success) {
  auto successResult = folly::Expected<int, MockError>(42);
  EXPECT_NO_THROW(FOLLY_EXPECTED_CHECKTHROW(successResult));
}

TEST_F(CommsUtilsCheckTest, FOLLY_EXPECTED_CHECKTHROW_Failure) {
  auto errorResult = folly::Expected<int, MockError>(folly::makeUnexpected(
      MockError{
          .errNum = EINVAL,
          .errStr = "mock error message",
      }));

  bool caughtException = false;
  try {
    FOLLY_EXPECTED_CHECKTHROW(errorResult);
  } catch (const std::runtime_error& e) {
    EXPECT_THAT(
        std::string(e.what()), ::testing::HasSubstr("COMM internal failure:"));
    EXPECT_THAT(
        std::string(e.what()), ::testing::HasSubstr("mock error message"));
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "Expected std::runtime_error";
}
