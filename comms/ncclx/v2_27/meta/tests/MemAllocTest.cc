// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "nccl.h"

class MemAllocTest : public ::testing::TestWithParam<size_t> {
 public:
  MemAllocTest() {
    ncclCvarInit();
    NCCLCHECK_TEST(ncclCudaLibraryInit());
  }
};

TEST_P(MemAllocTest, Alloc) {
  void* buf = nullptr;
  const size_t kTestSize = GetParam();

  CUDACHECK_TEST(cudaSetDevice(0));

  auto res = ncclMemAlloc(&buf, kTestSize);
  EXPECT_EQ(res, ncclSuccess);
  EXPECT_NE(buf, nullptr);

  if (ncclIsCuMemSupported()) {
    CUmemGenericAllocationHandle allocHandle;
    // Check if the buffer is created by cuMem. Otherwise it will fail with
    // error CUDA_ERROR_INVALID_VALUE.
    CUresult cuRes = CUPFN(cuMemRetainAllocationHandle)(&allocHandle, buf);
    EXPECT_EQ(cuRes, CUDA_SUCCESS);
    CUCHECK_TEST(cuMemRelease(allocHandle));
  }

  res = ncclMemFree(buf);
  EXPECT_EQ(res, ncclSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    MemAllocTestWithParamInstantiation,
    MemAllocTest,
    ::testing::Values(
        (size_t)1024 /*small*/,
        (size_t)65535 /* unaligned */,
        (size_t)1073741824 /* large*/),
    [](const ::testing::TestParamInfo<MemAllocTest::ParamType>& info) {
      return std::to_string(info.param) + "bytes";
    });
