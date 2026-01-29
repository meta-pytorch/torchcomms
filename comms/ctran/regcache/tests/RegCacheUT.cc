// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
