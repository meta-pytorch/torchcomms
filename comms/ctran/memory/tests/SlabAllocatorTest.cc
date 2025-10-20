// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/memory/SlabAllocator.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cstdlib>

#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "strongstream.h"

class SlabAllocatorTest : public ::testing::Test {
 public:
  int cudaDev = 0;

  SlabAllocatorTest() = default;

 protected:
  void SetUp() override {
    ncclCvarInit();
    ncclCudaLibraryInit();
    initEnv();
    logGpuMemoryStats(cudaDev);
    CUDACHECK_TEST(cudaSetDevice(cudaDev));
  }

  void TearDown() override {
    logGpuMemoryStats(cudaDev);
  }

  // helper method to return ground truth allocated bytes
  size_t actualUsedMem(
      ncclx::memory::SlabAllocator* allocator,
      size_t bytes,
      struct ncclStrongStream* ss) {
    size_t before_free, after_free, total;
    CUDACHECK_TEST(cudaMemGetInfo(&before_free, &total));
    void* ptr = nullptr;
    COMMCHECK_TEST(allocator->cuCallocAsync(
        &ptr, bytes, ss->liveStream, "slabAllocatorUT"));
    NCCLCHECK_TEST(ncclStrongStreamSynchronize(ss));
    CUDACHECK_TEST(cudaMemGetInfo(&after_free, &total));
    return before_free - after_free;
  }

  void allocAndCheckMemCpy(
      ncclx::memory::SlabAllocator* allocator,
      size_t bytes,
      struct ncclStrongStream* ss) {
    void* devicebuf = nullptr;
    COMMCHECK_TEST(allocator->cuCallocAsync(
        &devicebuf, bytes, ss->liveStream, "slabAllocatorUT"));
    NCCLCHECK_TEST(ncclStrongStreamSynchronize(ss));

    // Check if the memory is set to 0
    int* hostbuf = (int*)malloc(bytes);
    CUDACHECK_TEST(cudaMemcpy(hostbuf, devicebuf, bytes, cudaMemcpyDefault));

    for (int i = 0; i < bytes / sizeof(int); i++) {
      ASSERT_EQ(hostbuf[i], 0);
      hostbuf[i] = i;
    }

    // Perform memcpy to device
    CUDACHECK_TEST(cudaMemcpy(devicebuf, hostbuf, bytes, cudaMemcpyDefault));
    int* newHostBuf = (int*)malloc(bytes);

    // Copt data back to host and check correctness
    CUDACHECK_TEST(cudaMemcpy(newHostBuf, devicebuf, bytes, cudaMemcpyDefault));
    for (int j = 0; j < bytes / sizeof(int); j++) {
      ASSERT_EQ(newHostBuf[j], j);
    }

    // Check after memcpy values are the same
    free(hostbuf);
    free(newHostBuf);
  }
};

TEST_F(SlabAllocatorTest, ReuseSlabIfPossible) {
  auto allocator = std::make_unique<ncclx::memory::SlabAllocator>();
  EXPECT_THAT(allocator, testing::NotNull());
  // StrongStream construct may have tmp memory costs too, calling it here to
  // mimic prod behavior
  struct ncclStrongStream* ss = nullptr;
  cudaStream_t stream;
  NCCLCHECK_TEST(ncclCalloc(&ss, 1));
  NCCLCHECK_TEST(ncclStrongStreamConstruct(ss));
  NCCLCHECK_TEST(ncclStrongStreamAcquire(
      ncclCudaGraphNone(), ss, /*concurrent=*/false, &stream));
  size_t before_free, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before_free, &total));
  // allocation calls: 1 byte, (2097152 - 16) bytes, 2097152 * 2, 2097152 /2
  // bytes, 2097152/2 bytes
  // acutual allocations: 2097152 (allocate a slab), 0
  // (reuse first slab),  2097152 * 2 (allocate 2 slabs), 2097152 (allocate one
  // slab)
  EXPECT_EQ(actualUsedMem(allocator.get(), 1, ss), 2097152);
  EXPECT_EQ(actualUsedMem(allocator.get(), 2097152 - 16, ss), 0);
  EXPECT_EQ(actualUsedMem(allocator.get(), 2097152 * 2, ss), 2097152 * 2);
  EXPECT_EQ(actualUsedMem(allocator.get(), 2097152 / 2, ss), 2097152);
  NCCLCHECK_TEST(ncclStrongStreamSynchronize(ss));
  size_t after_free;
  CUDACHECK_TEST(cudaMemGetInfo(&after_free, &total));
  EXPECT_EQ(before_free - after_free, allocator->getUsedMem());
  NCCLCHECK_TEST(
      ncclStrongStreamRelease(ncclCudaGraphNone(), ss, /*concurrent=*/false));
  free(ss);
}

TEST_F(SlabAllocatorTest, FreeMemoryUponDestruction) {
  struct ncclStrongStream* ss = nullptr;
  cudaStream_t stream;
  NCCLCHECK_TEST(ncclCalloc(&ss, 1));
  NCCLCHECK_TEST(ncclStrongStreamConstruct(ss));
  NCCLCHECK_TEST(ncclStrongStreamAcquire(
      ncclCudaGraphNone(), ss, /*concurrent=*/false, &stream));
  size_t before_free, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before_free, &total));
  auto allocator = std::make_unique<ncclx::memory::SlabAllocator>();
  EXPECT_THAT(allocator, testing::NotNull());
  EXPECT_EQ(actualUsedMem(allocator.get(), 1, ss), 2097152);
  EXPECT_EQ(actualUsedMem(allocator.get(), 1, ss), 0);
  EXPECT_EQ(actualUsedMem(allocator.get(), 1, ss), 0);
  EXPECT_EQ(actualUsedMem(allocator.get(), 2097152, ss), 2097152);
  EXPECT_EQ(actualUsedMem(allocator.get(), 2097152 * 2, ss), 2097152 * 2);
  NCCLCHECK_TEST(ncclStrongStreamSynchronize(ss));
  NCCLCHECK_TEST(
      ncclStrongStreamRelease(ncclCudaGraphNone(), ss, /*concurrent=*/false));
  free(ss);
  allocator.reset();
  size_t after_free;
  CUDACHECK_TEST(cudaMemGetInfo(&after_free, &total));
  EXPECT_EQ(before_free, after_free);
}

TEST_F(SlabAllocatorTest, CudaMemCpyAsync) {
  auto allocator = std::make_unique<ncclx::memory::SlabAllocator>();
  EXPECT_THAT(allocator, testing::NotNull());
  struct ncclStrongStream* ss = nullptr;
  cudaStream_t stream;
  NCCLCHECK_TEST(ncclCalloc(&ss, 1));
  NCCLCHECK_TEST(ncclStrongStreamConstruct(ss));
  NCCLCHECK_TEST(ncclStrongStreamAcquire(
      ncclCudaGraphNone(), ss, /*concurrent=*/false, &stream));
  allocAndCheckMemCpy(allocator.get(), 4, ss);
  allocAndCheckMemCpy(allocator.get(), 2097152, ss);
  allocAndCheckMemCpy(allocator.get(), 2097152 * 2, ss);
  NCCLCHECK_TEST(ncclStrongStreamSynchronize(ss));
  NCCLCHECK_TEST(
      ncclStrongStreamRelease(ncclCudaGraphNone(), ss, /*concurrent=*/false));
  allocator.reset();
  free(ss);
}
