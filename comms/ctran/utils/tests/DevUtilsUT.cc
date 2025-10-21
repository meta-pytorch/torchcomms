// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/ctran/utils/DevUtils.cuh"

using namespace ctran::utils;

TEST(Log2iTest, BasicValues) {
  EXPECT_EQ(log2i(1), 0);
  EXPECT_EQ(log2i(2), 1);
  EXPECT_EQ(log2i(3), 1);
  EXPECT_EQ(log2i(4), 2);
  EXPECT_EQ(log2i(7), 2);
  EXPECT_EQ(log2i(8), 3);
  EXPECT_EQ(log2i(0), -1);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
