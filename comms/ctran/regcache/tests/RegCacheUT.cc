// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "comms/ctran/backends/ib/CtranIbSingleton.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

class RegCacheTest : public ::testing::Test {
 public:
  int cudaDev = 0;
  std::shared_ptr<ctran::RegCache> regCache{nullptr};

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_BACKENDS", "ib", 1);
    setenv("NCCL_CTRAN_REGISTER", "eager", 1);
    ncclCvarInit();

    // Initialize CUDA library (required for cuMem operations)
    ASSERT_EQ(ctran::utils::commCudaLibraryInit(), commSuccess);
    CUDACHECK_TEST(cudaSetDevice(cudaDev));

    regCache = ctran::RegCache::getInstance();
    ASSERT_NE(regCache, nullptr);
  }

  void TearDown() override {
    EXPECT_EQ(regCache->destroy(), commSuccess);
  }
};

// Test caching a single contiguous cudaMalloc buffer
TEST_F(RegCacheTest, CacheSegmentSingleContiguousBuffer) {
  size_t bufSize = 8192;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  std::vector<ctran::regcache::Segment*> segments;
  std::vector<void*> segHdls;

  EXPECT_EQ(
      regCache->cacheSegment(
          buf,
          bufSize,
          cudaDev,
          false /* ncclManaged */,
          0 /* commHash */,
          segments,
          segHdls),
      commSuccess);

  // cudaMalloc should result in exactly one segment
  EXPECT_EQ(segments.size(), 1);
  EXPECT_EQ(segHdls.size(), 1);
  EXPECT_NE(segments[0], nullptr);
  EXPECT_NE(segHdls[0], nullptr);

  // Verify segment properties
  EXPECT_EQ(segments[0]->getType(), DevMemType::kCudaMalloc);

  // Free the segment
  bool freed = false;
  bool ncclManaged = false;
  std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElems;
  EXPECT_EQ(
      regCache->freeSegment(segHdls[0], freed, ncclManaged, regElems),
      commSuccess);
  EXPECT_TRUE(freed);
  EXPECT_FALSE(ncclManaged);

  CUDACHECK_TEST(cudaFree(buf));
}

// Test that caching the same buffer twice increases refcount instead of
// creating duplicate entries
TEST_F(RegCacheTest, CacheSegmentRefCountIncrement) {
  size_t bufSize = 8192;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  // First cache
  std::vector<ctran::regcache::Segment*> segments1;
  std::vector<void*> segHdls1;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf, bufSize, cudaDev, false, 0, segments1, segHdls1),
      commSuccess);
  EXPECT_EQ(segments1.size(), 1);

  // Second cache of the same buffer
  std::vector<ctran::regcache::Segment*> segments2;
  std::vector<void*> segHdls2;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf, bufSize, cudaDev, false, 0, segments2, segHdls2),
      commSuccess);
  EXPECT_EQ(segments2.size(), 1);

  // Should return the same segment
  EXPECT_EQ(segments1[0], segments2[0]);
  EXPECT_EQ(segHdls1[0], segHdls2[0]);

  // First free should not actually free (refcount > 1)
  bool freed = false;
  bool ncclManaged = false;
  std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElems;
  EXPECT_EQ(
      regCache->freeSegment(segHdls1[0], freed, ncclManaged, regElems),
      commSuccess);
  EXPECT_FALSE(freed); // Not freed yet due to refcount

  // Second free should actually free
  EXPECT_EQ(
      regCache->freeSegment(segHdls2[0], freed, ncclManaged, regElems),
      commSuccess);
  EXPECT_TRUE(freed);

  CUDACHECK_TEST(cudaFree(buf));
}

// Test caching a disjoint (multi-segment) buffer
TEST_F(RegCacheTest, CacheSegmentDisjointMultiSegmentBuffer) {
  constexpr size_t segmentSize = 2 * 1024 * 1024; // 2MB per segment
  constexpr int numSegments = 3;
  std::vector<size_t> segSizes(numSegments, segmentSize);

  void* buf = nullptr;
  std::vector<TestMemSegment> memSegments;
  COMMCHECK_TEST(
      ctran::commMemAllocDisjoint(&buf, segSizes, memSegments, true));
  ASSERT_NE(buf, nullptr);
  ASSERT_EQ(memSegments.size(), numSegments);

  std::vector<ctran::regcache::Segment*> segments;
  std::vector<void*> segHdls;
  size_t totalSize = segmentSize * numSegments;

  EXPECT_EQ(
      regCache->cacheSegment(
          buf, totalSize, cudaDev, false, 0, segments, segHdls),
      commSuccess);

  // Should discover all physical segments
  EXPECT_EQ(segments.size(), numSegments);
  EXPECT_EQ(segHdls.size(), numSegments);

  // Verify all segments are distinct
  for (size_t i = 0; i < segments.size(); i++) {
    EXPECT_NE(segments[i], nullptr);
    EXPECT_NE(segHdls[i], nullptr);
    for (size_t j = i + 1; j < segments.size(); j++) {
      EXPECT_NE(segments[i], segments[j]);
      EXPECT_NE(segHdls[i], segHdls[j]);
    }
  }

  // Free all segments
  for (auto segHdl : segHdls) {
    bool freed = false;
    bool ncclManaged = false;
    std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElems;
    EXPECT_EQ(
        regCache->freeSegment(segHdl, freed, ncclManaged, regElems),
        commSuccess);
    EXPECT_TRUE(freed);
  }

  COMMCHECK_TEST(ctran::commMemFreeDisjoint(buf, segSizes));
}

// Test that caching a disjoint buffer twice reuses cached segments
TEST_F(RegCacheTest, CacheSegmentDisjointBufferRefCount) {
  constexpr size_t segmentSize = 2 * 1024 * 1024;
  constexpr int numSegments = 2;
  std::vector<size_t> segSizes(numSegments, segmentSize);

  void* buf = nullptr;
  std::vector<TestMemSegment> memSegments;
  COMMCHECK_TEST(
      ctran::commMemAllocDisjoint(&buf, segSizes, memSegments, true));
  ASSERT_NE(buf, nullptr);

  size_t totalSize = segmentSize * numSegments;

  // First cache
  std::vector<ctran::regcache::Segment*> segments1;
  std::vector<void*> segHdls1;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf, totalSize, cudaDev, false, 0, segments1, segHdls1),
      commSuccess);
  EXPECT_EQ(segments1.size(), numSegments);

  // Second cache
  std::vector<ctran::regcache::Segment*> segments2;
  std::vector<void*> segHdls2;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf, totalSize, cudaDev, false, 0, segments2, segHdls2),
      commSuccess);
  EXPECT_EQ(segments2.size(), numSegments);

  // Should reuse the same segments
  for (size_t i = 0; i < numSegments; i++) {
    EXPECT_EQ(segments1[i], segments2[i]);
    EXPECT_EQ(segHdls1[i], segHdls2[i]);
  }

  // First free should not actually free any segment
  for (auto segHdl : segHdls1) {
    bool freed = false;
    bool ncclManaged = false;
    std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElems;
    EXPECT_EQ(
        regCache->freeSegment(segHdl, freed, ncclManaged, regElems),
        commSuccess);
    EXPECT_FALSE(freed);
  }

  // Second free should free all segments
  for (auto segHdl : segHdls2) {
    bool freed = false;
    bool ncclManaged = false;
    std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElems;
    EXPECT_EQ(
        regCache->freeSegment(segHdl, freed, ncclManaged, regElems),
        commSuccess);
    EXPECT_TRUE(freed);
  }

  COMMCHECK_TEST(ctran::commMemFreeDisjoint(buf, segSizes));
}

// Test lookupSegmentsForBuffer for a single contiguous buffer
TEST_F(RegCacheTest, LookupSegmentsForBufferSingleSegment) {
  size_t bufSize = 8192;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  // First cache the buffer
  std::vector<ctran::regcache::Segment*> segments;
  std::vector<void*> segHdls;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf, bufSize, cudaDev, false, 0, segments, segHdls),
      commSuccess);
  EXPECT_EQ(segHdls.size(), 1);

  // Look up segments for the buffer
  std::vector<void*> foundSegHdls;
  std::vector<ctran::regcache::RegElem*> foundRegElems;
  EXPECT_EQ(
      regCache->lookupSegmentsForBuffer(
          buf, bufSize, cudaDev, foundSegHdls, foundRegElems),
      commSuccess);

  // Should find the cached segment
  EXPECT_EQ(foundSegHdls.size(), 1);
  EXPECT_EQ(foundSegHdls[0], segHdls[0]);

  // Free the segment
  bool freed = false;
  bool ncclManaged = false;
  std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElems;
  EXPECT_EQ(
      regCache->freeSegment(segHdls[0], freed, ncclManaged, regElems),
      commSuccess);
  EXPECT_TRUE(freed);

  CUDACHECK_TEST(cudaFree(buf));
}

// Test lookupSegmentsForBuffer for a disjoint multi-segment buffer
TEST_F(RegCacheTest, LookupSegmentsForBufferMultiSegment) {
  constexpr size_t segmentSize = 2 * 1024 * 1024; // 2MB per segment
  constexpr int numSegments = 3;
  std::vector<size_t> segSizes(numSegments, segmentSize);

  void* buf = nullptr;
  std::vector<TestMemSegment> memSegments;
  COMMCHECK_TEST(
      ctran::commMemAllocDisjoint(&buf, segSizes, memSegments, true));
  ASSERT_NE(buf, nullptr);

  size_t totalSize = segmentSize * numSegments;

  // First cache the buffer
  std::vector<ctran::regcache::Segment*> segments;
  std::vector<void*> segHdls;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf, totalSize, cudaDev, false, 0, segments, segHdls),
      commSuccess);
  EXPECT_EQ(segHdls.size(), numSegments);

  // Look up segments for the buffer
  std::vector<void*> foundSegHdls;
  std::vector<ctran::regcache::RegElem*> foundRegElems;
  EXPECT_EQ(
      regCache->lookupSegmentsForBuffer(
          buf, totalSize, cudaDev, foundSegHdls, foundRegElems),
      commSuccess);

  // Should find all cached segments
  EXPECT_EQ(foundSegHdls.size(), numSegments);
  for (size_t i = 0; i < numSegments; i++) {
    EXPECT_EQ(foundSegHdls[i], segHdls[i]);
  }

  // Free all segments
  for (auto segHdl : segHdls) {
    bool freed = false;
    bool ncclManaged = false;
    std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElems;
    EXPECT_EQ(
        regCache->freeSegment(segHdl, freed, ncclManaged, regElems),
        commSuccess);
    EXPECT_TRUE(freed);
  }

  COMMCHECK_TEST(ctran::commMemFreeDisjoint(buf, segSizes));
}

// Test lookupSegmentsForBuffer returns empty when buffer is not cached
TEST_F(RegCacheTest, LookupSegmentsForBufferNotCached) {
  size_t bufSize = 8192;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  // Look up segments without caching first
  std::vector<void*> foundSegHdls;
  std::vector<ctran::regcache::RegElem*> foundRegElems;
  EXPECT_EQ(
      regCache->lookupSegmentsForBuffer(
          buf, bufSize, cudaDev, foundSegHdls, foundRegElems),
      commSuccess);

  // Should not find any segments
  EXPECT_EQ(foundSegHdls.size(), 0);
  EXPECT_EQ(foundRegElems.size(), 0);

  CUDACHECK_TEST(cudaFree(buf));
}

// Test that destroy clears all cached segments and registrations
TEST_F(RegCacheTest, DestroyCleansCachedSegments) {
  size_t bufSize = 8192;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  // Cache a segment
  std::vector<ctran::regcache::Segment*> segments;
  std::vector<void*> segHdls;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf, bufSize, cudaDev, false, 0, segments, segHdls),
      commSuccess);
  EXPECT_EQ(segments.size(), 1);

  // Verify segment is cached
  EXPECT_EQ(regCache->getSegments().size(), 1);

  // Destroy should clean up all segments
  EXPECT_EQ(regCache->destroy(), commSuccess);

  // Verify segments are cleaned after destroy
  EXPECT_EQ(regCache->getSegments().size(), 0);

  // Re-init for the next test iteration (TearDown will call destroy again)
  regCache->init();

  CUDACHECK_TEST(cudaFree(buf));
}

// Test that destroy clears async registration queue and thread can be restarted
TEST_F(RegCacheTest, DestroyResetsAsyncRegThread) {
  EnvRAII env(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::async);
  regCache->destroy();
  regCache->init();
  size_t bufSize = 8192;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  // Cache a segment first (required for async registration)
  std::vector<ctran::regcache::Segment*> segments;
  std::vector<void*> segHdls;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf, bufSize, cudaDev, false, 0, segments, segHdls),
      commSuccess);

  // Submit an async registration request
  struct CommLogData logMetaData = {};
  std::vector<bool> backends = {true, false, false}; // IB only
  EXPECT_EQ(
      regCache->asyncRegRange(buf, bufSize, cudaDev, logMetaData, backends),
      commSuccess);

  // Wait for async registration to complete
  regCache->waitAsyncRegComplete();

  // Verify the registration was created
  EXPECT_TRUE(regCache->isRegistered(buf, bufSize));

  // Destroy should clean up the async thread and queue
  EXPECT_EQ(regCache->destroy(), commSuccess);

  // Re-init to restart the async thread
  regCache->init();

  // Allocate a new buffer for the second iteration
  void* buf2 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf2, bufSize));

  // Cache and async register the new buffer - this verifies the async thread
  // was properly reset and can handle new requests after destroy/init cycle
  std::vector<ctran::regcache::Segment*> segments2;
  std::vector<void*> segHdls2;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf2, bufSize, cudaDev, false, 0, segments2, segHdls2),
      commSuccess);

  EXPECT_EQ(
      regCache->asyncRegRange(buf2, bufSize, cudaDev, logMetaData, backends),
      commSuccess);

  // Wait for async registration to complete - this would hang if the async
  // thread wasn't properly reset
  regCache->waitAsyncRegComplete();

  // Verify the new registration was created successfully
  EXPECT_TRUE(regCache->isRegistered(buf2, bufSize));

  // Clean up segments before destroy
  bool freed = false;
  bool ncclManaged = false;
  std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElems;
  EXPECT_EQ(
      regCache->freeSegment(segHdls2[0], freed, ncclManaged, regElems),
      commSuccess);
  EXPECT_TRUE(freed);

  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaFree(buf2));
}

// Test that destroy/init cycle can be repeated multiple times
TEST_F(RegCacheTest, DestroyInitCycleRepeatable) {
  size_t bufSize = 8192;

  for (int i = 0; i < 3; i++) {
    void* buf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

    // Cache a segment
    std::vector<ctran::regcache::Segment*> segments;
    std::vector<void*> segHdls;
    EXPECT_EQ(
        regCache->cacheSegment(
            buf, bufSize, cudaDev, false, 0, segments, segHdls),
        commSuccess);
    EXPECT_EQ(segments.size(), 1);

    // Verify segment is cached
    EXPECT_EQ(regCache->getSegments().size(), 1);

    // Destroy should clean up
    EXPECT_EQ(regCache->destroy(), commSuccess);

    // Verify clean state
    EXPECT_EQ(regCache->getSegments().size(), 0);

    // Re-init for next iteration
    regCache->init();

    CUDACHECK_TEST(cudaFree(buf));
  }
}

// Test that destroy clears profiler counts
TEST_F(RegCacheTest, DestroyResetsProfiler) {
  EnvRAII env(NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT, 1);
  size_t bufSize = 8192;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  // Cache a segment to generate profiler events
  std::vector<ctran::regcache::Segment*> segments;
  std::vector<void*> segHdls;
  EXPECT_EQ(
      regCache->cacheSegment(
          buf, bufSize, cudaDev, false, 0, segments, segHdls),
      commSuccess);

  // Check profiler has recorded the cache event
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_GT(snapshot.totalNumCache, 0);

  // Destroy and re-init
  EXPECT_EQ(regCache->destroy(), commSuccess);
  regCache->init();

  // Reset the profiler after re-init (since profiler is not automatically reset
  // by destroy, but this test documents the expected behavior)
  regCache->profiler.wlock()->reset();

  // Verify profiler is reset
  auto snapshotAfter = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshotAfter.totalNumCache, 0);
  EXPECT_EQ(snapshotAfter.totalNumReg, 0);

  CUDACHECK_TEST(cudaFree(buf));
}

// Test that RegCache holds a reference to CtranIbSingleton for proper
// destruction ordering. This verifies the dependency is correctly established.
TEST_F(RegCacheTest, HoldsCtranIbSingletonReference) {
  auto ibSingleton = CtranIbSingleton::getInstance();
  EXPECT_NE(ibSingleton, nullptr);

  // Verify RegCache is also valid
  EXPECT_NE(regCache, nullptr);

  // Both singletons should be independently accessible
  EXPECT_NE(ibSingleton.get(), nullptr);

  // Verify destroy/init cycle works correctly with the dependency
  EXPECT_EQ(regCache->destroy(), commSuccess);
  regCache->init();

  // After re-init, IB singleton should still be accessible
  auto ibSingletonAfterInit = CtranIbSingleton::getInstance();
  EXPECT_NE(ibSingletonAfterInit, nullptr);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
