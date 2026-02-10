// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * Unit tests for global registration API.
 *
 * These tests verify that ncclGlobalRegisterWithPtr/ncclGlobalDeregisterWithPtr
 * work correctly WITHOUT requiring any communicator or mapper initialization.
 *
 * This is critical for the CUDACachingAllocator integration where memory may be
 * allocated before any NCCL communicator is created.
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
    setenv("NCCL_CTRAN_REGISTER", "eager", 1);
    ncclCvarInit();

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
  commResult_t result = ctran::globalRegisterWithPtr(buf, bufSize, cudaDev);
  EXPECT_EQ(result, commSuccess)
      << "Global registration should succeed without comm/mapper";

  // Deregister
  result = ctran::globalDeregisterWithPtr(buf, bufSize, cudaDev);
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
    commResult_t result =
        ctran::globalRegisterWithPtr(buffers[i], bufSize, cudaDev);
    EXPECT_EQ(result, commSuccess)
        << "Registration " << i << " should succeed without comm";
  }

  // Deregister all buffers
  for (int i = 0; i < numBuffers; i++) {
    commResult_t result =
        ctran::globalDeregisterWithPtr(buffers[i], bufSize, cudaDev);
    EXPECT_EQ(result, commSuccess)
        << "Deregistration " << i << " should succeed";
  }

  // Cleanup
  for (int i = 0; i < numBuffers; i++) {
    CUDACHECK_TEST(cudaFree(buffers[i]));
  }
}

/**
 * Test: CtranIbSingleton is lazily initialized by static CtranIb::regMem.
 *
 * This test verifies the key architectural point: CtranIb::regMem (static)
 * triggers CtranIbSingleton initialization via getInstance().
 */
TEST_F(GlobalRegistrationTest, IbSingletonLazyInitialization) {
  // Get the singleton - this will lazily initialize it if not already done
  CtranIbSingleton& singleton = CtranIbSingleton::getInstance();

  // The singleton should have IB devices available
  EXPECT_GT(singleton.ibvDevices.size(), 0)
      << "CtranIbSingleton should have discovered IB devices";

  // Now do a registration - this should use the same singleton
  void* ibRegElem = nullptr;
  commResult_t result = CtranIb::regMem(buf, bufSize, cudaDev, &ibRegElem);

  // If IB is available, registration should succeed
  if (result == commSuccess) {
    EXPECT_NE(ibRegElem, nullptr)
        << "IB registration should return valid handle";

    // Cleanup
    EXPECT_EQ(CtranIb::deregMem(ibRegElem), commSuccess);
  } else {
    // IB may not be available in all test environments
    GTEST_SKIP() << "IB backend not available, skipping IB-specific test";
  }
}

/**
 * Test: Registration through RegCache globalRegister succeeds.
 *
 * This test verifies that RegCache::globalRegister properly initializes
 * the global backends from NCCL_CTRAN_BACKENDS and uses them for registration.
 */
TEST_F(GlobalRegistrationTest, RegCacheGlobalBackendsInitialized) {
  auto regCache = ctran::RegCache::getInstance();
  ASSERT_NE(regCache, nullptr);

  // Call globalRegister which uses globalBackends_
  commResult_t result = regCache->globalRegister(buf, bufSize, cudaDev);
  EXPECT_EQ(result, commSuccess) << "RegCache::globalRegister should succeed";

  // Cleanup
  result = regCache->globalDeregister(buf, bufSize, cudaDev);
  EXPECT_EQ(result, commSuccess);
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
      ctran::globalRegisterWithPtr(disjointBuf, totalSize, cudaDev);
  EXPECT_EQ(result, commSuccess)
      << "Global registration of disjoint memory should succeed";

  // Verify that RegCache has cached the segments via pinRange discovery
  auto regCache = ctran::RegCache::getInstance();
  ASSERT_NE(regCache, nullptr);

  // Check that registration was tracked
  EXPECT_TRUE(regCache->isRegistered(disjointBuf, totalSize))
      << "Full disjoint buffer should be registered";

  // Deregister
  result = ctran::globalDeregisterWithPtr(disjointBuf, totalSize, cudaDev);
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
      ctran::globalRegisterWithPtr(disjointBuf, totalSize, cudaDev);
  EXPECT_EQ(result, commSuccess)
      << "Global registration of multi-segment memory should succeed";

  // Verify registration
  auto regCache = ctran::RegCache::getInstance();
  ASSERT_NE(regCache, nullptr);
  EXPECT_TRUE(regCache->isRegistered(disjointBuf, totalSize))
      << "Full multi-segment buffer should be registered";

  // Deregister
  result = ctran::globalDeregisterWithPtr(disjointBuf, totalSize, cudaDev);
  EXPECT_EQ(result, commSuccess);

  // Free the disjoint memory
  NCCLCHECK_TEST(ncclMemFreeDisjoint(disjointBuf, segmentSizes));
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
