// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/Logger.h"

using namespace ctran::utils;

class CudaWrapTest : public ::testing::Test {
 public:
  void SetUp() override {
    ncclCvarInit();
    COMMCHECK_TEST(commCudaLibraryInit());
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {}
};

TEST_F(CudaWrapTest, FB_CUCHECKTHROW) {
  auto dummyFn = []() {
    // This should fail because we're passing an invalid device pointer
    CUdeviceptr invalidPtr = 0;
    size_t size = 0;
    FB_CUCHECKTHROW(cuMemGetAddressRange(&invalidPtr, &size, (CUdeviceptr)0x1));
    return commSuccess;
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
