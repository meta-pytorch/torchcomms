// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <nccl.h>

#include "comms/ctran/memory/Utils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/Logger.h"

class memoryUtilsTest : public ::testing::Test {
 public:
  int cudaDev = 0;
  CommLogData dummyLogData;

  memoryUtilsTest() = default;

 protected:
  void SetUp() override {
    ctran::logGpuMemoryStats(cudaDev);
    CUDACHECK_TEST(cudaSetDevice(cudaDev));
    setenv("NCCL_DEBUG", "INFO", 0);
    setenv("NCCL_DEBUG_SUBSYS", "ALLOC", 0);
    ncclCvarInit();
    ncclCudaLibraryInit();
    initNcclLogger();

    dummyLogData = CommLogData{
        .commId = 0,
        .commHash = 0xfaceb00c12345678,
        .commDesc = "ncclx.ut",
        .rank = 0,
        .nRanks = 1};
  }

  void TearDown() override {
    ctran::logGpuMemoryStats(cudaDev);
    NcclLogger::close();
  }
};

TEST_F(memoryUtilsTest, cudaCallocAsync) {
  float* ptr = nullptr;
  size_t count = 1 << 20; // 1M float = 2MB
  cudaStream_t stream = nullptr;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  size_t before, after, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));

  EXPECT_EQ(
      ncclx::memory::cudaCallocAsync(
          &ptr, count, stream, &dummyLogData, __func__),
      commSuccess);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, count * sizeof(float));

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // can be freed by ncclCudaFree
  EXPECT_EQ(ncclCudaFree(ptr), commSuccess);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before, after);
}

TEST_F(memoryUtilsTest, cudaCallocAsyncSlab) {
  EnvRAII<bool> useSlab(NCCL_MEM_USE_SLAB_ALLOCATOR, true);
  float* ptr = nullptr;
  size_t count = 1 << 18; // 256K float = 1MB
  cudaStream_t stream = nullptr;

  auto allocator = std::make_unique<ncclx::memory::SlabAllocator>();
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  size_t before, after, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));

  EXPECT_EQ(
      ncclx::memory::cudaCallocAsync(
          &ptr, count, stream, &dummyLogData, __func__, allocator.get()),
      commSuccess);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, allocator->getUsedMem());

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  allocator.reset();

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before, after);
}
TEST_F(memoryUtilsTest, allocateShareableBuffer) {
  EnvRAII<size_t> poolSizeGuard(NCCL_MEM_POOL_SIZE, 0);
  auto allocator = ncclx::memory::memCacheAllocator::getInstance();
  void* ptr = nullptr;
  size_t size = 1 << 21; // 2MB

  size_t before, after, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));

  ncclx::memory::allocatorIpcDesc ipcDesc;
  EXPECT_EQ(
      ncclx::memory::allocateShareableBuffer(
          size,
          /*refcount=*/0,
          &ipcDesc,
          &ptr,
          allocator,
          &dummyLogData,
          __func__),
      commSuccess);

  EXPECT_NE(ptr, nullptr);
  EXPECT_TRUE(ipcDesc.udsMemHandle.has_value());

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, size);
  EXPECT_EQ(
      allocator->release(
          {folly::sformat("{}:{:#x}", __func__, dummyLogData.commHash)}),
      commSuccess);
}
