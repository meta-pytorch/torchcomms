// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <nccl.h>
#include <iostream>

TEST(Version, Code) {
  int ncclVersion{0};
  EXPECT_EQ(ncclGetVersion(&ncclVersion), ncclSuccess);
  EXPECT_EQ(NCCL_VERSION_CODE, ncclVersion);
  std::cout << "NCCL version: " << NCCL_VERSION_CODE << std::endl;
  std::cout << "ncclGetVersion: " << ncclVersion << std::endl;

  // Check same as expected version from BUCK
  EXPECT_EQ(TEST_NCCL_VERSION, ncclVersion);
}

TEST(Version, NoVer) {
  EXPECT_EQ(ncclGetVersion(NULL), ncclInvalidArgument);
}
