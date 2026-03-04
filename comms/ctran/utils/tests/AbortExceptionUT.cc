// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/utils/AbortException.h"
#include "comms/ctran/utils/Exception.h"

namespace ctran::utils {

// Test default construction with context only.
// Verifies default retriable=false, result code, message content, and default
// optional parameter accessors.
TEST(AbortExceptionTest, BasicThrow) {
  try {
    throw AbortException("test abort context");
  } catch (const AbortException& e) {
    EXPECT_FALSE(e.isRetriable());
    EXPECT_EQ(e.result(), commRemoteError);
    EXPECT_THAT(e.what(), ::testing::HasSubstr("AbortException"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("test abort context"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("retriable: false"));
    EXPECT_EQ(e.rank(), -1);
    EXPECT_EQ(e.commHash(), static_cast<uint64_t>(-1));
    EXPECT_EQ(e.desc(), "undefined");
  }
}

// Test construction with retriable=true.
TEST(AbortExceptionTest, RetriableTrue) {
  try {
    throw AbortException("retriable context", true);
  } catch (const AbortException& e) {
    EXPECT_TRUE(e.isRetriable());
    EXPECT_THAT(e.what(), ::testing::HasSubstr("retriable: true"));
  }
}

// Test with all parameters provided.
TEST(AbortExceptionTest, ThrowWithFullInfo) {
  const int rank = 5;
  const uint64_t commHash = 0xDEADBEEF;
  const std::string desc = "custom description";

  try {
    throw AbortException("full info context", true, rank, commHash, desc);
  } catch (const AbortException& e) {
    EXPECT_TRUE(e.isRetriable());
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_EQ(e.desc(), desc);
    EXPECT_THAT(e.what(), ::testing::HasSubstr("full info context"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("rank: 5"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("commHash: deadbeef"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("desc: custom description"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("retriable: true"));
  }
}

// Test that result code is always commRemoteError regardless of other params.
TEST(AbortExceptionTest, ResultCodeIsCommRemoteError) {
  try {
    throw AbortException("result code test", true, 3, 0xABC);
  } catch (const AbortException& e) {
    EXPECT_EQ(e.result(), commRemoteError);
  }
}

// Test that AbortException can be caught as its base class Exception&,
// confirming the polymorphic inheritance chain works.
TEST(AbortExceptionTest, CatchAsBaseException) {
  try {
    throw AbortException("polymorphic test", true);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("AbortException"));
    EXPECT_THAT(e.what(), ::testing::HasSubstr("polymorphic test"));
    EXPECT_EQ(e.result(), commRemoteError);
  }
}

} // namespace ctran::utils
