// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/utils/Exception.h"

class ExceptionTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(ExceptionTest, BasicThrow) {
  const auto res = commInvalidArgument;

  try {
    throw ctran::utils::Exception("Dummy test failure context", res);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("Dummy test failure context"));
    EXPECT_EQ(e.result(), res);
    EXPECT_EQ(e.rank(), -1);
    EXPECT_EQ(e.commHash(), -1);
  }
}

TEST(ExceptionTest, ThrowWithFullInfo) {
  const auto res = commInvalidArgument;
  const auto rank = 1;
  const auto commHash = 0x1234;

  try {
    throw ctran::utils::Exception(
        "Dummy test failure context", res, rank, commHash);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("Dummy test failure context"));
    EXPECT_EQ(e.result(), res);
    EXPECT_EQ(e.rank(), rank);
    EXPECT_EQ(e.commHash(), commHash);
    EXPECT_EQ(e.desc(), "undefined");
  }
}

TEST(ExceptionTest, ThrowWithDesc) {
  const auto res = commInvalidArgument;
  const std::string desc = "DummyTestDesc";
  try {
    throw ctran::utils::Exception(
        "Dummy test failure context", res, std::nullopt, std::nullopt, desc);
  } catch (const ctran::utils::Exception& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("Dummy test failure context"));
    EXPECT_EQ(e.result(), res);
    EXPECT_EQ(e.desc(), desc);
    EXPECT_EQ(e.rank(), -1); // undefined
    EXPECT_EQ(e.commHash(), -1); // undefined
  }
}
