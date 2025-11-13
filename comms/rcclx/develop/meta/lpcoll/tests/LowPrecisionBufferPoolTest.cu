// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "meta/lpcoll/low_precision_buffer_pool.h"
#include "meta/lpcoll/low_precision_common.h"
#include "nccl.h"

class LowPrecisionBufferPoolTest : public ::testing::Test {
 public:
  LowPrecisionBufferPoolTest() = default;

  void SetUp() override {
    setenv("RCCL_LOW_PRECISION_ENABLE", "1", 1);

    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    unsetenv("RCCL_LOW_PRECISION_ENABLE");
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(LowPrecisionBufferPoolTest, InitAndDestroy) {
  struct ncclLowPrecisionBufferPool pool;
  memset(&pool, 0, sizeof(pool));

  const size_t maxElements = 1024 * 1024; // 1M elements
  const int maxRanks = 8;

  ncclResult_t result =
      ncclLowPrecisionBufferPoolInit(&pool, maxElements, maxRanks);
  ASSERT_EQ(result, ncclSuccess);
  EXPECT_TRUE(pool.initialized);
  EXPECT_NE(pool.backingBuffer, nullptr);
  EXPECT_GT(pool.maxBufferSize, 0);

  result = ncclLowPrecisionBufferPoolDestroy(&pool);
  ASSERT_EQ(result, ncclSuccess);
  EXPECT_FALSE(pool.initialized);
  EXPECT_EQ(pool.backingBuffer, nullptr);
}

TEST_F(LowPrecisionBufferPoolTest, DoubleInit) {
  struct ncclLowPrecisionBufferPool pool;
  memset(&pool, 0, sizeof(pool));

  const size_t maxElements = 1024 * 1024;
  const int maxRanks = 8;

  ncclResult_t result =
      ncclLowPrecisionBufferPoolInit(&pool, maxElements, maxRanks);
  ASSERT_EQ(result, ncclSuccess);

  // Second init should succeed without re-initializing
  result = ncclLowPrecisionBufferPoolInit(&pool, maxElements, maxRanks);
  ASSERT_EQ(result, ncclSuccess);

  ncclLowPrecisionBufferPoolDestroy(&pool);
}

TEST_F(LowPrecisionBufferPoolTest, GetBuffers) {
  struct ncclLowPrecisionBufferPool pool;
  memset(&pool, 0, sizeof(pool));

  const size_t maxElements = 1024 * 1024;
  const int maxRanks = 8;

  ncclResult_t result =
      ncclLowPrecisionBufferPoolInit(&pool, maxElements, maxRanks);
  ASSERT_EQ(result, ncclSuccess);

  rccl_float8* fp8Phase1Buffer = nullptr;
  rccl_float8* fp8Phase2Buffer = nullptr;
  rccl_float8* fp8AllGatherBuffer = nullptr;
  float* floatReductionBuffer = nullptr;
  float* floatOutputBuffer = nullptr;

  result = ncclLowPrecisionBufferPoolGetBuffers(
      &pool,
      maxElements,
      maxRanks,
      &fp8Phase1Buffer,
      &fp8Phase2Buffer,
      &fp8AllGatherBuffer,
      &floatReductionBuffer,
      &floatOutputBuffer);

  ASSERT_EQ(result, ncclSuccess);
  EXPECT_NE(fp8Phase1Buffer, nullptr);
  EXPECT_NE(fp8Phase2Buffer, nullptr);
  EXPECT_NE(fp8AllGatherBuffer, nullptr);
  EXPECT_NE(floatReductionBuffer, nullptr);
  EXPECT_NE(floatOutputBuffer, nullptr);

  ncclLowPrecisionBufferPoolDestroy(&pool);
}

TEST_F(LowPrecisionBufferPoolTest, GetBuffersUninitialized) {
  struct ncclLowPrecisionBufferPool pool;
  memset(&pool, 0, sizeof(pool));

  rccl_float8* fp8Phase1Buffer = nullptr;
  rccl_float8* fp8Phase2Buffer = nullptr;
  rccl_float8* fp8AllGatherBuffer = nullptr;
  float* floatReductionBuffer = nullptr;
  float* floatOutputBuffer = nullptr;

  ncclResult_t result = ncclLowPrecisionBufferPoolGetBuffers(
      &pool,
      1024,
      8,
      &fp8Phase1Buffer,
      &fp8Phase2Buffer,
      &fp8AllGatherBuffer,
      &floatReductionBuffer,
      &floatOutputBuffer);

  EXPECT_EQ(result, ncclInvalidUsage);
}

TEST_F(LowPrecisionBufferPoolTest, GetBuffersPartial) {
  struct ncclLowPrecisionBufferPool pool;
  memset(&pool, 0, sizeof(pool));

  const size_t maxElements = 1024 * 1024;
  const int maxRanks = 8;

  ncclResult_t result =
      ncclLowPrecisionBufferPoolInit(&pool, maxElements, maxRanks);
  ASSERT_EQ(result, ncclSuccess);

  // Test getting only some buffers
  rccl_float8* fp8Phase1Buffer = nullptr;
  float* floatOutputBuffer = nullptr;

  result = ncclLowPrecisionBufferPoolGetBuffers(
      &pool,
      maxElements,
      maxRanks,
      &fp8Phase1Buffer,
      nullptr,
      nullptr,
      nullptr,
      &floatOutputBuffer);

  ASSERT_EQ(result, ncclSuccess);
  EXPECT_NE(fp8Phase1Buffer, nullptr);
  EXPECT_NE(floatOutputBuffer, nullptr);

  ncclLowPrecisionBufferPoolDestroy(&pool);
}

TEST_F(LowPrecisionBufferPoolTest, EnsureBufferPool) {
  const size_t count = 512 * 1024; // 512K elements
  const int nRanks = 8;

  ncclResult_t result =
      ncclEnsureLowPrecisionBufferPool(this->comm, count, nRanks);
  ASSERT_EQ(result, ncclSuccess);

  struct ncclLowPrecisionBufferPool* pool = &this->comm->lowPrecisionBufferPool;
  EXPECT_TRUE(pool->initialized);
  EXPECT_NE(pool->backingBuffer, nullptr);
}

TEST_F(LowPrecisionBufferPoolTest, EnsureBufferPoolGrowth) {
  // Start with small buffer
  const size_t smallCount = 128 * 1024;
  const int nRanks = 8;

  ncclResult_t result =
      ncclEnsureLowPrecisionBufferPool(this->comm, smallCount, nRanks);
  ASSERT_EQ(result, ncclSuccess);

  struct ncclLowPrecisionBufferPool* pool = &this->comm->lowPrecisionBufferPool;
  size_t initialSize = pool->maxBufferSize;

  // Request larger buffer
  const size_t largeCount = 512 * 1024 * 1024; // 512M elements
  result = ncclEnsureLowPrecisionBufferPool(this->comm, largeCount, nRanks);
  ASSERT_EQ(result, ncclSuccess);

  // Verify buffer grew
  EXPECT_GT(pool->maxBufferSize, initialSize);
}

TEST_F(LowPrecisionBufferPoolTest, BufferAlignment) {
  struct ncclLowPrecisionBufferPool pool;
  memset(&pool, 0, sizeof(pool));

  const size_t maxElements = 1024 * 1024;
  const int maxRanks = 8;

  ncclResult_t result =
      ncclLowPrecisionBufferPoolInit(&pool, maxElements, maxRanks);
  ASSERT_EQ(result, ncclSuccess);

  rccl_float8* fp8Phase1Buffer = nullptr;
  rccl_float8* fp8Phase2Buffer = nullptr;
  rccl_float8* fp8AllGatherBuffer = nullptr;
  float* floatReductionBuffer = nullptr;
  float* floatOutputBuffer = nullptr;

  result = ncclLowPrecisionBufferPoolGetBuffers(
      &pool,
      maxElements,
      maxRanks,
      &fp8Phase1Buffer,
      &fp8Phase2Buffer,
      &fp8AllGatherBuffer,
      &floatReductionBuffer,
      &floatOutputBuffer);

  ASSERT_EQ(result, ncclSuccess);

  // Verify 128-byte alignment for cache optimization
  EXPECT_EQ(reinterpret_cast<uintptr_t>(fp8Phase1Buffer) % 128, 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(floatReductionBuffer) % 128, 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(floatOutputBuffer) % 128, 0);

  ncclLowPrecisionBufferPoolDestroy(&pool);
}

TEST_F(LowPrecisionBufferPoolTest, InitForComm) {
  struct ncclLowPrecisionBufferPool* pool = &this->comm->lowPrecisionBufferPool;

  // Clear any existing initialization
  if (pool->initialized) {
    ncclLowPrecisionBufferPoolDestroy(pool);
  }

  ncclResult_t result =
      ncclInitLowPrecisionBufferPoolForComm(this->comm, this->numRanks);
  ASSERT_EQ(result, ncclSuccess);
  EXPECT_TRUE(pool->initialized);
  EXPECT_NE(pool->backingBuffer, nullptr);
  EXPECT_GT(pool->maxBufferSize, 0);
}

TEST_F(LowPrecisionBufferPoolTest, CalculateKernelConfig) {
  struct ncclLowPrecisionKernelConfig config;

  const size_t totalElements = 1024 * 1024;
  const size_t chunkElements = 256 * 1024;

  ncclResult_t result = ncclCalculateLowPrecisionKernelConfig(
      totalElements, chunkElements, &config);

  ASSERT_EQ(result, ncclSuccess);
  EXPECT_GT(config.blockSize, 0);
  EXPECT_GT(config.maxBlocks, 0);
  EXPECT_GT(config.fullGridSize, 0);
  EXPECT_GT(config.chunkGridSize, 0);
  EXPECT_LE(config.chunkGridSize, config.fullGridSize);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
