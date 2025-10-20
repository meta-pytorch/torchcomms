// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <nccl.h>
#include <thread>

#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/Logger.h"

#include "param.h"

class memCacheAllocatorTest : public ::testing::Test {
 public:
  int cudaDev = 0;
  std::string testName;
  CommLogData dummyLogData;

  memCacheAllocatorTest() = default;

 protected:
  void SetUp() override {
    logGpuMemoryStats(cudaDev);
    CUDACHECK_TEST(cudaSetDevice(cudaDev));
    setenv("NCCL_DEBUG", "INFO", 0);
    setenv("NCCL_DEBUG_SUBSYS", "ALLOC", 0);
    // TODO: remove this when memCache does not rely on colltrace
    setenv("NCCL_COLLTRACE", "trace", 1);
    ncclCvarInit();
    ncclCudaLibraryInit();
    initNcclLogger();

    dummyLogData = CommLogData{
        .commDesc = "ncclx.ut",
        .commId = 0,
        .commHash = 0xfaceb00c12345678,
        .nRanks = 1,
        .rank = 0};
    const ::testing::TestInfo* test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    testName =
        folly::sformat("{}.{}", test_info->test_case_name(), test_info->name());
  }

  void TearDown() override {
    logGpuMemoryStats(cudaDev);
    NcclLogger::close();
  }

  std::shared_ptr<ncclx::memory::memRegion> getMemRegByIdWrapper(
      ncclx::memory::memCacheAllocator* allocator,
      const std::string memId) {
    auto hashKey = allocator->genHash(memId);
    return allocator->cachedRegionMap_.at(hashKey);
  }
};

TEST_F(memCacheAllocatorTest, cachedMemGetAndReleaseWithoutPool) {
  EnvRAII<size_t> poolSizeGuard(NCCL_MEM_POOL_SIZE, 0);
  auto allocator = ncclx::memory::memCacheAllocator::getInstance();

  void* ptr = nullptr;
  void* ptr2 = nullptr;
  void* ptr3 = nullptr;
  void* ptr4 = nullptr;

  std::string memId1 = folly::sformat("{}1", testName);
  std::string memId2 = folly::sformat("{}2", testName);
  size_t nBytes = 1 << 25; // 32MB

  size_t before, after, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));

  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId1,
          &ptr,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, nBytes);

  // release by id
  EXPECT_EQ(allocator->release({memId1}), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 0);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  // try to get a buffer with same key from cache allocator again
  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId1,
          &ptr,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_NE(ptr, nullptr);
  // shouldn't have allocated a new region
  EXPECT_EQ(allocator->getNumUsedReg(), 1);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, nBytes);

  // Try getting the same key without release, should return the same ptr
  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId1,
          &ptr2,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_EQ(ptr, ptr2);
  // shouldn't have allocated a new region
  EXPECT_EQ(allocator->getNumUsedReg(), 1);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  // Release memId1, let memId2 use the same region then memId1 should get a
  // different region
  EXPECT_EQ(allocator->release({memId1, memId1}), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 0);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId2,
          &ptr3,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_EQ(ptr2, ptr3);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId1,
          &ptr4,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_NE(ptr3, ptr4);
  EXPECT_EQ(allocator->getNumUsedReg(), 2);
  EXPECT_EQ(allocator->getNumAllocReg(), 2);

  // release both used keys
  EXPECT_EQ(allocator->release({memId1, memId2}), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 0);
  EXPECT_EQ(allocator->reset(), commSuccess);
}

TEST_F(memCacheAllocatorTest, cachedMemByString) {
  EnvRAII<size_t> poolSizeGuard(NCCL_MEM_POOL_SIZE, 0);
  auto allocator = ncclx::memory::memCacheAllocator::getInstance();

  void* ptr = nullptr;

  std::string memId = testName;
  size_t nBytes = 1 << 25; // 32MB

  size_t before, after, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));

  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId,
          &ptr,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, nBytes);

  // release by key
  EXPECT_EQ(allocator->release({memId}), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 0);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  // try to get a buffer with same size from cache allocator again
  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId,
          &ptr,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, nBytes);

  // release by key
  EXPECT_EQ(allocator->release({memId}), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 0);
  EXPECT_EQ(allocator->reset(), commSuccess);
}

TEST_F(memCacheAllocatorTest, cachedMemDoubleGetSameId) {
  EnvRAII<size_t> poolSizeGuard(NCCL_MEM_POOL_SIZE, 0);
  auto allocator = ncclx::memory::memCacheAllocator::getInstance();

  void* ptr = nullptr;
  void* ptr2 = nullptr;
  void* ptr3 = nullptr;

  std::string memId = __func__;
  size_t nBytes = 1 << 25; // 32MB

  size_t before, after, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));

  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId,
          &ptr,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          __func__),
      commSuccess);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);
  auto memRegion = getMemRegByIdWrapper(allocator.get(), memId);
  EXPECT_NE(memRegion, nullptr);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, nBytes);

  // get the buffer using the same key again before releasing it should return
  // the same buffer
  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId,
          &ptr2,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          __func__),
      commSuccess);
  EXPECT_EQ(ptr, ptr2);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);

  // release the key
  EXPECT_EQ(allocator->release({memId, memId}), commSuccess);

  // using the same key should still return the same ptr
  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId,
          &ptr3,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          __func__),
      commSuccess);
  EXPECT_EQ(ptr, ptr2);
  EXPECT_EQ(ptr2, ptr3);
  // Only allocated one region
  EXPECT_EQ(allocator->getNumUsedReg(), 1);

  EXPECT_EQ(allocator->release({memId}), commSuccess);
  EXPECT_EQ(allocator->reset(), commSuccess);
}

TEST_F(memCacheAllocatorTest, cachedMemGetAndReleaseWithPool) {
  EnvRAII<size_t> poolSizeGuard(NCCL_MEM_POOL_SIZE, 1 << 26); // 64MB
  // Check pool initial allocation usage
  size_t before_init, after_init, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before_init, &total));
  auto allocator = ncclx::memory::memCacheAllocator::getInstance();
  CUDACHECK_TEST(cudaMemGetInfo(&after_init, &total));
  EXPECT_EQ(before_init - after_init, NCCL_MEM_POOL_SIZE);

  void* ptr = nullptr;
  void* ptr2 = nullptr;
  void* ptr3 = nullptr;

  std::string memId1 = folly::sformat("{}1", __func__);
  std::string memId2 = folly::sformat("{}2", __func__);
  std::string memId3 = folly::sformat("{}3", __func__);
  size_t nBytes = 1 << 25; // 32MB

  size_t before, after;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));

  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId1,
          &ptr,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  // no additional memory should be allocated since the pool is large enough
  EXPECT_EQ(before, after);

  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId2,
          &ptr2,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_NE(ptr2, nullptr);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  // no additional memory should be allocated since the pool is large enough
  EXPECT_EQ(before, after);

  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId3,
          &ptr3,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_NE(ptr3, nullptr);

  // release the ptr
  EXPECT_EQ(allocator->release({memId1, memId2, memId3}), commSuccess);

  // Should have created 3 regions, none should still be used
  EXPECT_EQ(allocator->getNumUsedReg(), 0);
  EXPECT_EQ(allocator->getNumAllocReg(), 3);
  EXPECT_EQ(allocator->reset(), commSuccess);
}

TEST_F(memCacheAllocatorTest, multiThreadAccess) {
  EnvRAII<size_t> poolSizeGuard(NCCL_MEM_POOL_SIZE, 1 << 26); // 64MB
  auto allocator = ncclx::memory::memCacheAllocator::getInstance();
  size_t nBytes = 1 << 22; // 4MB
  const int numThreads = 20;
  std::vector<std::thread> acquireThreads, releaseThreads;
  std::vector<void*> ptrs(numThreads, nullptr);

  size_t before, after, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));

  // acquire staging buffer from different threads
  auto acquireWorker = [&](int id) {
    std::string memIdPerThread = folly::sformat("{}{}", testName, id);
    EXPECT_EQ(
        allocator->getCachedCuMemById(
            memIdPerThread,
            &ptrs.at(id),
            /*cuHandle=*/nullptr,
            nBytes,
            &dummyLogData,
            testName.c_str()),
        commSuccess);
    EXPECT_NE(ptrs.at(id), nullptr);
  };

  for (int i = 0; i < numThreads; i++) {
    acquireThreads.emplace_back(acquireWorker, i);
  }

  // Join threads
  for (auto& t : acquireThreads) {
    t.join();
  }

  EXPECT_EQ(allocator->getNumUsedReg(), numThreads);

  // release buffer from different thread
  auto releaseWorker = [&](int id) {
    std::string memIdPerThread = folly::sformat("{}{}", testName.c_str(), id);
    EXPECT_EQ(allocator->release({memIdPerThread}), commSuccess);
  };

  for (int i = 0; i < numThreads; i++) {
    releaseThreads.emplace_back(releaseWorker, i);
  }

  // Join threads
  for (auto& t : releaseThreads) {
    t.join();
  }

  EXPECT_EQ(allocator->getNumUsedReg(), 0);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  // Used 4MB * 20 = 80MB total, pool allcoated 64MB so we should have used 16MB
  EXPECT_EQ(before - after, nBytes * numThreads - NCCL_MEM_POOL_SIZE);
  EXPECT_EQ(allocator->reset(), commSuccess);
}

TEST_F(memCacheAllocatorTest, reserve) {
  EnvRAII<size_t> poolSizeGuard(NCCL_MEM_POOL_SIZE, 0);
  auto allocator = ncclx::memory::memCacheAllocator::getInstance();

  std::string memId = testName;
  void* ptr = nullptr;

  size_t nBytes = 1 << 25; // 32MB

  size_t before, after, finish, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));
  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId,
          &ptr,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);

  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, nBytes);

  // release the ptr
  EXPECT_EQ(allocator->release({memId}), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 0);
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  // reserve the region using the same key
  EXPECT_EQ(allocator->reserve(memId), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);

  // release the reserved region
  EXPECT_EQ(allocator->release({memId}), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 0);

  // Only created one region throughout
  EXPECT_EQ(allocator->getNumAllocReg(), 1);

  EXPECT_EQ(allocator->reset(), commSuccess);
  CUDACHECK_TEST(cudaMemGetInfo(&finish, &total));
  EXPECT_EQ(before, finish);
  EXPECT_EQ(allocator->reset(), commSuccess);
}

TEST_F(memCacheAllocatorTest, invalidReserve) {
  EnvRAII<size_t> poolSizeGuard(NCCL_MEM_POOL_SIZE, 0);
  auto allocator = ncclx::memory::memCacheAllocator::getInstance();
  std::string memId = folly::sformat("{}1", testName);
  std::string memId2 = folly::sformat("{}2", testName);
  void* ptr = nullptr;
  void* ptr2 = nullptr;

  size_t nBytes = 1 << 25; // 32MB

  size_t before, after, finish, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));
  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId,
          &ptr,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);

  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(allocator->getNumUsedReg(), 1);

  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, nBytes);

  EXPECT_EQ(allocator->reserve(memId), commSuccess);
  // release the ptr
  EXPECT_EQ(allocator->release({memId, memId}), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 0);

  EXPECT_EQ(
      allocator->getCachedCuMemById(
          memId2,
          &ptr2,
          /*cuHandle=*/nullptr,
          nBytes,
          &dummyLogData,
          testName.c_str()),
      commSuccess);
  EXPECT_EQ(ptr, ptr2);

  // reserve the region after it's been get() by a different key
  // should result in ncclInProgress
  EXPECT_EQ(allocator->reserve(memId), ncclInProgress);

  // reserve the region using a non-existent key without calling get() first
  // should result in ncclInvalidArgument
  EXPECT_EQ(allocator->reserve("nonExistentMemId"), ncclInvalidArgument);

  // release the ptr
  EXPECT_EQ(allocator->release({memId2}), commSuccess);
  EXPECT_EQ(allocator->getNumUsedReg(), 0);

  EXPECT_EQ(allocator->reset(), commSuccess);
  CUDACHECK_TEST(cudaMemGetInfo(&finish, &total));
  EXPECT_EQ(before, finish);
  EXPECT_EQ(allocator->reset(), commSuccess);
}
