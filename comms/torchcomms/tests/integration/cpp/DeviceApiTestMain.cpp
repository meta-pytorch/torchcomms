// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Entry point for DeviceApiStressTest (NCCLx backend).
// Links against the test library which statically registers all TEST_F cases.

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
