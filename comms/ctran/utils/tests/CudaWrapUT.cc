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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
