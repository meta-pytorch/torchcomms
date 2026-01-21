// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/utils/AbortException.h"
#include "comms/ctran/utils/Exception.h"

namespace ctran::utils {

// Test that AbortException can be thrown and caught with basic parameters.
// Validates that the context appears in the message and cause is set correctly.
TEST(AbortExceptionTest, BasicThrow) {
  try {
    throw AbortException("test abort context", AbortCause::UNKNOWN);
  } catch (const AbortException& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("test abort context"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("AbortException"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("abortCause: UNKNOWN"));
    EXPECT_EQ(e.cause(), AbortCause::UNKNOWN);
    EXPECT_EQ(e.result(), commRemoteError);
  }
}

// Test AbortException with all optional parameters provided.
// Verifies that rank, commHash, and description are included in the message.
TEST(AbortExceptionTest, ThrowWithFullInfo) {
  const int rank = 5;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "custom description";

  try {
    throw AbortException(
        "full info context", AbortCause::REMOTE_PEER, rank, commHash, desc);
  } catch (const AbortException& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("full info context"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("rank: 5"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("commHash: deadbeef"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("desc: custom description"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("abortCause: REMOTE_PEER"));
    EXPECT_EQ(e.cause(), AbortCause::REMOTE_PEER);
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_EQ(e.desc(), desc);
  }
}

// Test each AbortCause enum value to verify abortCauseToString works correctly.
TEST(AbortExceptionTest, AbortCauseToString) {
  EXPECT_EQ(AbortException::abortCauseToString(AbortCause::UNKNOWN), "UNKNOWN");
  EXPECT_EQ(
      AbortException::abortCauseToString(AbortCause::USER_INITIATED),
      "USER_INITIATED");
  EXPECT_EQ(
      AbortException::abortCauseToString(AbortCause::REMOTE_PEER),
      "REMOTE_PEER");
  EXPECT_EQ(AbortException::abortCauseToString(AbortCause::TIMEOUT), "TIMEOUT");
  EXPECT_EQ(
      AbortException::abortCauseToString(AbortCause::SYSTEM_ERROR),
      "SYSTEM_ERROR");
}

// Test that each AbortCause value appears correctly in the exception message.
TEST(AbortExceptionTest, AllCauseTypesInMessage) {
  const std::vector<std::pair<AbortCause, std::string>> testCases = {
      {AbortCause::UNKNOWN, "UNKNOWN"},
      {AbortCause::USER_INITIATED, "USER_INITIATED"},
      {AbortCause::REMOTE_PEER, "REMOTE_PEER"},
      {AbortCause::TIMEOUT, "TIMEOUT"},
      {AbortCause::SYSTEM_ERROR, "SYSTEM_ERROR"},
  };

  for (const auto& [cause, expectedStr] : testCases) {
    try {
      throw AbortException("cause test", cause);
    } catch (const AbortException& e) {
      EXPECT_EQ(e.cause(), cause);
      EXPECT_THAT(e.what(), ::testing::HasSubstr("abortCause: " + expectedStr));
    }
  }
}

// Test that the result code is always commRemoteError for AbortException.
TEST(AbortExceptionTest, ResultCodeIsCommRemoteError) {
  try {
    throw AbortException("result code test", AbortCause::TIMEOUT);
  } catch (const AbortException& e) {
    EXPECT_EQ(e.result(), commRemoteError);
  }
}

// Test default values for optional parameters when not provided.
TEST(AbortExceptionTest, DefaultOptionalParameters) {
  try {
    throw AbortException("defaults test", AbortCause::SYSTEM_ERROR);
  } catch (const AbortException& e) {
    EXPECT_EQ(e.rank(), -1);
    EXPECT_EQ(e.commHash(), static_cast<uint64_t>(-1));
    EXPECT_EQ(e.desc(), "undefined");
  }
}

} // namespace ctran::utils
