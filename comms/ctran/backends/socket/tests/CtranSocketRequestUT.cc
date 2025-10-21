// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/backends/socket/CtranSocketBase.h"

class CtranSocketRequestTest : public ::testing::Test {
 public:
  CtranSocketRequestTest() = default;

 protected:
  void SetUp() override {}
};

TEST_F(CtranSocketRequestTest, Complete) {
  CtranSocketRequest req;
  auto res = req.complete();
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(req.isComplete());
}
