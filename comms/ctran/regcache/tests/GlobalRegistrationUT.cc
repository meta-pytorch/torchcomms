// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * Unit tests for global registration API.
 *
 * These tests verify that
 * ctran::globalRegisterWithPtr/ctran::globalDeregisterWithPtr work correctly
 * WITHOUT requiring any communicator or mapper initialization.
 *
 * Key verification points:
 * 1. CtranIbSingleton is lazily initialized on first regMem call
 * 2. Registration succeeds without any comm/mapper
 * 3. Deregistration properly cleans up
 */

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

namespace {

class GlobalRegistrationTest : public ::testing::Test {
 public:
  int cudaDev{0};
  size_t bufSize{1024 * 1024}; // 1MB
  void* buf{nullptr};

 protected:
  void SetUp() override {
    // Initialize environment - but NO comm/mapper creation
    setenv("NCCL_CTRAN_BACKENDS", "ib", 1);
    setenv("NCCL_CTRAN_REGISTER", "async", 1);
    ncclCvarInit();

    meta::comms::logger::initCommLogging();

    // Initialize CUDA library (required for cuMem operations)
    ASSERT_EQ(ctran::utils::commCudaLibraryInit(), commSuccess);

    // Set CUDA device and allocate memory
    CUDACHECK_TEST(cudaSetDevice(cudaDev));
    CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
    CUDACHECK_TEST(cudaMemset(buf, 0, bufSize));
  }

  void TearDown() override {
    if (buf) {
      CUDACHECK_TEST(cudaFree(buf));
      buf = nullptr;
    }

    // Cleanup RegCache for next test
    auto regCache = ctran::RegCache::getInstance();
    if (regCache) {
      EXPECT_EQ(regCache->destroy(), commSuccess);
    }
  }
};

/**
 * Test: Global registration works without any comm/mapper initialization.
 *
 * This test verifies the core assumption that:
 * 1. ctran::globalRegisterWithPtr can be called without a comm
 * 2. CtranIbSingleton is lazily initialized when needed
 * 3. Registration and deregistration succeed
 */
TEST_F(GlobalRegistrationTest, RegisterWithoutCommOrMapper) {
  // Verify NO CtranComm or CtranMapper exists - we're testing global API only
  // There is no comm to check, this test runs standalone

  // Call global registration API directly (same path as CCA integration)
  commResult_t result = ctran::globalRegisterWithPtr(buf, bufSize);
  EXPECT_EQ(result, commSuccess)
      << "Global registration should succeed without comm/mapper";

  // Deregister
  result = ctran::globalDeregisterWithPtr(buf, bufSize);
  EXPECT_EQ(result, commSuccess) << "Global deregistration should succeed";
}

/**
 * Test: Multiple registrations and deregistrations work correctly.
 *
 * This simulates the CCA pattern where multiple memory allocations
 * are registered before a comm is created.
 */
TEST_F(GlobalRegistrationTest, MultipleRegistrationsBeforeComm) {
  constexpr int numBuffers = 4;
  std::vector<void*> buffers(numBuffers, nullptr);

  // Allocate multiple buffers
  for (int i = 0; i < numBuffers; i++) {
    CUDACHECK_TEST(cudaMalloc(&buffers[i], bufSize));
    CUDACHECK_TEST(cudaMemset(buffers[i], i, bufSize));
  }

  // Register all buffers using global API - NO comm exists
  for (int i = 0; i < numBuffers; i++) {
    commResult_t result = ctran::globalRegisterWithPtr(buffers[i], bufSize);
    EXPECT_EQ(result, commSuccess)
        << "Registration " << i << " should succeed without comm";
  }

  // Deregister all buffers
  for (int i = 0; i < numBuffers; i++) {
    commResult_t result = ctran::globalDeregisterWithPtr(buffers[i], bufSize);
    EXPECT_EQ(result, commSuccess)
        << "Deregistration " << i << " should succeed";
  }

  // Cleanup
  for (int i = 0; i < numBuffers; i++) {
    CUDACHECK_TEST(cudaFree(buffers[i]));
  }
}

/**
 * Test: Multi-segment registration via ncclMemAllocDisjoint.
 *
 * This test verifies that cacheSegment correctly handles disjoint memory
 * allocations (multiple physical segments mapped to a contiguous virtual
 * address range). This is the core path used by PyTorch's expandable segments
 * feature in CUDACachingAllocator.
 *
 * The test:
 * 1. Allocates memory with ncclMemAllocDisjoint (creates 2 physical segments)
 * 2. Registers the full virtual range via global registration
 * 3. Verifies that pinRange discovers all physical segments
 * 4. Deregisters and frees the memory
 */
TEST_F(GlobalRegistrationTest, MultiSegmentDisjointRegistration) {
  // Allocate disjoint memory with 2 segments
  constexpr size_t totalSize = 2 * 1024 * 1024; // 2MB total
  std::vector<size_t> segmentSizes = {totalSize / 2, totalSize / 2};
  std::vector<TestMemSegment> segments;
  void* disjointBuf = nullptr;

  NCCLCHECK_TEST(ncclMemAllocDisjoint(&disjointBuf, segmentSizes, segments));
  ASSERT_NE(disjointBuf, nullptr);
  ASSERT_EQ(segments.size(), 2) << "Should have 2 physical segments";

  // Verify segments are contiguous in virtual address space
  uintptr_t seg0End =
      reinterpret_cast<uintptr_t>(segments[0].ptr) + segments[0].size;
  uintptr_t seg1Start = reinterpret_cast<uintptr_t>(segments[1].ptr);
  EXPECT_EQ(seg0End, seg1Start)
      << "Segments should be contiguous in virtual address space";

  // Register the full virtual range using global registration
  commResult_t result =
      ctran::globalRegisterWithPtr(disjointBuf, totalSize, /*forceReg=*/true);
  EXPECT_EQ(result, commSuccess)
      << "Global registration of disjoint memory should succeed";

  // Verify that RegCache has cached the segments via pinRange discovery
  auto regCache = ctran::RegCache::getInstance();
  ASSERT_NE(regCache, nullptr);

  // Check that registration was tracked
  EXPECT_TRUE(regCache->isRegistered(disjointBuf, totalSize))
      << "Full disjoint buffer should be registered";

  // Deregister
  result = ctran::globalDeregisterWithPtr(disjointBuf, totalSize);
  EXPECT_EQ(result, commSuccess)
      << "Global deregistration of disjoint memory should succeed";

  // Verify deregistration
  EXPECT_FALSE(regCache->isRegistered(disjointBuf, totalSize))
      << "Buffer should no longer be registered after deregistration";

  // Free the disjoint memory
  NCCLCHECK_TEST(ncclMemFreeDisjoint(disjointBuf, segmentSizes));
}

/**
 * Test: Multi-segment registration with many segments.
 *
 * This test verifies that cacheSegment handles allocations with many
 * physical segments, similar to large PyTorch allocations that span
 * multiple 20MB chunks in expandable segments mode.
 */
TEST_F(GlobalRegistrationTest, MultiSegmentManyChunks) {
  // Allocate disjoint memory with 5 segments (simulating 5 x 20MB chunks)
  constexpr int numSegments = 5;
  constexpr size_t segmentSize = 512 * 1024; // 512KB per segment for test
  constexpr size_t totalSize = numSegments * segmentSize;

  std::vector<size_t> segmentSizes(numSegments, segmentSize);
  std::vector<TestMemSegment> segments;
  void* disjointBuf = nullptr;

  NCCLCHECK_TEST(ncclMemAllocDisjoint(&disjointBuf, segmentSizes, segments));
  ASSERT_NE(disjointBuf, nullptr);
  ASSERT_EQ(segments.size(), numSegments)
      << "Should have " << numSegments << " physical segments";

  // Register the full virtual range
  commResult_t result =
      ctran::globalRegisterWithPtr(disjointBuf, totalSize, /*forceReg=*/true);
  EXPECT_EQ(result, commSuccess)
      << "Global registration of multi-segment memory should succeed";

  // Verify registration
  auto regCache = ctran::RegCache::getInstance();
  ASSERT_NE(regCache, nullptr);
  EXPECT_TRUE(regCache->isRegistered(disjointBuf, totalSize))
      << "Full multi-segment buffer should be registered";

  // Deregister
  result = ctran::globalDeregisterWithPtr(disjointBuf, totalSize);
  EXPECT_EQ(result, commSuccess);

  // Free the disjoint memory
  NCCLCHECK_TEST(ncclMemFreeDisjoint(disjointBuf, segmentSizes));
}

/**
 * Test: CPU tensor global registration works correctly.
 *
 * This test verifies that ctran::globalRegisterWithPtr handles CPU memory
 * (host memory allocated via malloc) correctly. CPU memory registration
 * proceeds through the normal path, and the IB backend handles it gracefully
 * via exception catching in doRegister, matching CtranMapper behavior.
 */
TEST_F(GlobalRegistrationTest, CpuTensorRegistration) {
  constexpr size_t cpuBufSize = 1024 * 1024; // 1MB
  void* cpuBuf = malloc(cpuBufSize);
  ASSERT_NE(cpuBuf, nullptr) << "CPU memory allocation should succeed";
  memset(cpuBuf, 0, cpuBufSize);

  // Register CPU memory using global API.
  // CPU memory registration proceeds through the normal path, matching
  // CtranMapper behavior. IB backend handles CPU memory gracefully via
  // exception catching in doRegister.
  commResult_t result =
      ctran::globalRegisterWithPtr(cpuBuf, cpuBufSize, /*forceReg=*/true);
  EXPECT_EQ(result, commSuccess) << "CPU memory registration should succeed";

  // Verify that searchRegElem finds the registered element (via isRegistered)
  auto regCache = ctran::RegCache::getInstance();
  ASSERT_NE(regCache, nullptr);
  EXPECT_TRUE(regCache->isRegistered(cpuBuf, cpuBufSize))
      << "CPU buffer should be found by searchRegElem after registration";

  // Deregistration of CPU memory should also succeed gracefully
  result = ctran::globalDeregisterWithPtr(cpuBuf, cpuBufSize);
  EXPECT_EQ(result, commSuccess) << "CPU memory deregistration should succeed";

  free(cpuBuf);
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
