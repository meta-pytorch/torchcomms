// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/Logger.h"

using namespace ctran::utils;

class CommAllocTest : public ::testing::Test {
 public:
  void SetUp() override {
    ncclCvarInit();
    COMMCHECK_TEST(commCudaLibraryInit());
    CUDACHECK_TEST(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    gpuName_ = prop.name;
    CLOGF(INFO, "GPU name: {}", gpuName_);
  }

  void TearDown() override {}

 protected:
  std::string gpuName_;
  CommLogData logMetaData_ = {
      .commId = 12345,
      .commHash = 1,
      .commDesc = "dummyCommDesc",
      .rank = 0,
      .nRanks = 1};
};

void ensureDeviceMemNoLeak(const std::function<void()>& test_impl) {
  size_t before_free, after_free, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before_free, &total));

  test_impl();

  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaMemGetInfo(&after_free, &total));
  EXPECT_EQ(before_free, after_free);
}

TEST_F(CommAllocTest, CuMemAllocation) {
  if (!ctran::utils::getCuMemSysSupported()) {
    GTEST_SKIP() << "CuMem not supported, skip CuMemAllocation test";
  }
  // TODO: device attribute query
  // (CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) doesn't give us
  // trustable info about if fabric handle is supported, so we use this
  // hardcoded GPU name to test. See:
  // https://forums.developer.nvidia.com/t/cudevicegetattribute-shows-i-can-use-fabric-handle-but-actually-i-cannot/336426/10
  if (gpuName_.find("GB") == std::string::npos) {
    // on non-GB (Grace Blackwell) platforms, fabric is not supported, we use
    // posix file descriptor for cuda allocation
    ensureDeviceMemNoLeak([this]() {
      void* ptr{nullptr};
      CUmemGenericAllocationHandle handle;
      size_t size = 1024;

      EXPECT_EQ(
          getCuMemAllocHandleType(), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

      commResult_t result = commCuMemAlloc(
          &ptr, &handle, getCuMemAllocHandleType(), size, &logMetaData_, "");

      EXPECT_EQ(result, commSuccess);
      EXPECT_NE(ptr, nullptr);

      result = commCuMemFree(ptr);
      EXPECT_EQ(result, commSuccess);
    });
  } else {
#if !defined(USE_ROCM)
    // for GB (Grace Blackwell) platforms, fabric handle is auto-detected as
    // supported.
    ensureDeviceMemNoLeak([this]() {
      void* ptr{nullptr};
      CUmemGenericAllocationHandle handle;
      size_t size = 1024;

      EXPECT_EQ(
          getCuMemAllocHandleType(),
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC);

      commResult_t result = commCuMemAlloc(
          &ptr, &handle, getCuMemAllocHandleType(), size, &logMetaData_, "");

      EXPECT_EQ(result, commSuccess);
      EXPECT_NE(ptr, nullptr);

      result = commCuMemFree(ptr);
      EXPECT_EQ(result, commSuccess);
    });
#endif
  }
}

TEST_F(CommAllocTest, isCuMemFabricHandleSupported) {
  // getCuMemAllocHandleType() triggers runtime fabric detection internally
  // (alloc/export/import probe), so we cannot wrap it in ensureDeviceMemNoLeak
  // since the CUDA allocator may retain internal memory pools after the probe
  // even though all allocations are freed.
  bool fabricSupported = isCuMemFabricEnabled();
  if (gpuName_.find("GB") == std::string::npos) {
    // on non-GB (Grace Blackwell) platforms, fabric handle should not be
    // supported
    EXPECT_FALSE(fabricSupported);
  } else {
    // for GB (Grace Blackwell) platforms, fabric handle should be auto-detected
    // as supported
    EXPECT_TRUE(fabricSupported);
  }
}

TEST_F(CommAllocTest, CudaAllocation) {
  ensureDeviceMemNoLeak([this]() {
    float* ptr{nullptr};
    size_t size = 1024;

    commResult_t result =
        commCudaMalloc(&ptr, size, &logMetaData_, "testAlloc");

    EXPECT_EQ(result, commSuccess);
    EXPECT_NE(ptr, nullptr);

    result = commCudaFree(ptr);
    EXPECT_EQ(result, commSuccess);
  });
}

TEST_F(CommAllocTest, CudaCallocAsync) {
  if (!ctran::utils::getCuMemSysSupported()) {
    GTEST_SKIP() << "CuMem not supported, skip CudaCallocAsync test";
  }
  ensureDeviceMemNoLeak([this]() {
    char* ptr{nullptr};
    constexpr size_t cnt = 1024;
    cudaStream_t stream;

    CUDACHECK_TEST(cudaStreamCreate(&stream));

    commResult_t result = commCudaCallocAsync(
        &ptr, cnt, stream, &logMetaData_, "testCallocAsync");

    EXPECT_EQ(result, commSuccess);
    EXPECT_NE(ptr, nullptr);

    // Verify memory is zeroed
    char hostPtr[cnt];
    CUDACHECK_TEST(
        cudaMemcpy(hostPtr, ptr, cnt * sizeof(char), cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < cnt; i++) {
      EXPECT_EQ(hostPtr[i], 0);
    }

    result = commCudaFree(ptr);
    EXPECT_EQ(result, commSuccess);

    CUDACHECK_TEST(cudaStreamDestroy(stream));
  });
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
