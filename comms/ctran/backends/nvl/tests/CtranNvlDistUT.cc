// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <pthread.h>
#include <stdlib.h>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

#include "comms/testinfra/TestsCuUtils.h"
#if !defined(USE_ROCM)
// needed because we use ncclMemAlloc to test kMemNcclMemAlloc mem type.
// cuMem API is not supported on AMD so we don't test it on AMD.
#include "comms/testinfra/TestUtils.h"
#endif

class CtranNvlTest : public ctran::CtranDistTestFixture {
 public:
  CtranNvlTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    CtranDistTestFixture::SetUp();
    comm_ = makeCtranComm();
    comm = comm_.get();

    // Check epoch lock for the entire test
    NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK = true;

    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    CtranDistTestFixture::TearDown();
  }

 protected:
  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
};

class CtranNvlTestSuite : public CtranNvlTest,
                          public ::testing::WithParamInterface<MemAllocType> {};

TEST_P(CtranNvlTestSuite, NormalInitialize) {
  // Expect CtranNvl to be initialized without internal error
  try {
    auto ctranNvl = std::make_unique<CtranNvl>(this->comm);
  } catch (const std::bad_alloc&) {
    GTEST_SKIP() << "NVL backend failed to allocate. Skip test";
  }
}

INSTANTIATE_TEST_SUITE_P(
    CtranNvlTestInstance,
    CtranNvlTestSuite,
#if !defined(USE_ROCM)
    ::testing::Values(kMemNcclMemAlloc, kMemCudaMalloc));
#else
    ::testing::Values(kMemCudaMalloc));
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
