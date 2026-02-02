// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string>

#include <cuda.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/utils/CudaChecks.h"

class CommsUtilsCudaChecksTest : public ::testing::Test {};

TEST_F(CommsUtilsCudaChecksTest, FB_CUCHECKTHROW) {
  auto dummyFn = []() {
    CUdeviceptr invalidPtr = 0;
    size_t size = 0;
    FB_CUCHECKTHROW(cuMemGetAddressRange(&invalidPtr, &size, (CUdeviceptr)0x1));
    return true;
  };

  bool caughtException = false;
  try {
    dummyFn();
  } catch (const std::runtime_error& e) {
    auto errMsg = std::string(e.what());
    EXPECT_THAT(errMsg, ::testing::HasSubstr("Cuda failure"));
    caughtException = true;
  }

  ASSERT_TRUE(caughtException) << "Expected std::runtime_error";
}
