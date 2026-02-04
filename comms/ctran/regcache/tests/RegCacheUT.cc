// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <thread>

#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

class RegCacheTest : public ::testing::Test {
 public:
  int cudaDev = 0;
  std::shared_ptr<ctran::RegCache> regCache{nullptr};

  // Helper struct for registered buffer management in tests
  struct RegisteredBuffer {
    void* buf{nullptr};
    size_t bufSize{0};
    std::vector<ctran::regcache::Segment*> segments;
    std::vector<void*> segHdls;
    ctran::regcache::RegElem* regElem{nullptr};
  };

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

  // Helper to allocate, cache, and register a buffer in one call.
  // Returns a RegisteredBuffer struct containing all the handles needed
  // for testing and cleanup.
  RegisteredBuffer allocateAndRegister(size_t bufSize) {
    RegisteredBuffer rb;
    rb.bufSize = bufSize;

    // Allocate CUDA memory
    CUDACHECK_TEST(cudaMalloc(&rb.buf, bufSize));

    // Cache the segment
    EXPECT_EQ(
        regCache->cacheSegment(
            rb.buf, bufSize, cudaDev, false, 0, rb.segments, rb.segHdls),
        commSuccess);

    // Register the buffer
    bool didRegister = false;
    std::vector<bool> backends(3, false);
    backends[0] = true; // IB backend
    EXPECT_EQ(
        regCache->regRange(
            rb.buf,
            bufSize,
            cudaDev,
            "test",
            CommLogData{},
            backends,
            didRegister,
            &rb.regElem),
        commSuccess);

    return rb;
  }

  // Helper to free a registered buffer and its associated resources
  void freeRegisteredBuffer(RegisteredBuffer& rb) {
    for (auto segHdl : rb.segHdls) {
      bool freed = false;
      bool ncclManaged = false;
      std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElems;
      EXPECT_EQ(
          regCache->freeSegment(segHdl, freed, ncclManaged, regElems),
          commSuccess);
    }
    if (rb.buf) {
      CUDACHECK_TEST(cudaFree(rb.buf));
      rb.buf = nullptr;
    }
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

// Test RegElemGuard correctly manages inUseCount
TEST_F(RegCacheTest, RegElemGuardIncrementsInUseCount) {
  auto rb = allocateAndRegister(8192);
  ASSERT_NE(rb.regElem, nullptr);

  // Initially inUseCount should be 0
  EXPECT_EQ(rb.regElem->getInUseCount(), 0);
  EXPECT_FALSE(rb.regElem->isInUse());

  // Create a guard - should increment inUseCount
  {
    ctran::regcache::RegElemGuard guard(rb.regElem);
    EXPECT_EQ(rb.regElem->getInUseCount(), 1);
    EXPECT_TRUE(rb.regElem->isInUse());

    // Create another guard - should increment again
    {
      ctran::regcache::RegElemGuard guard2(rb.regElem);
      EXPECT_EQ(rb.regElem->getInUseCount(), 2);
      EXPECT_TRUE(rb.regElem->isInUse());
    }
    // guard2 destroyed - should decrement
    EXPECT_EQ(rb.regElem->getInUseCount(), 1);
    EXPECT_TRUE(rb.regElem->isInUse());
  }
  // guard destroyed - should decrement back to 0
  EXPECT_EQ(rb.regElem->getInUseCount(), 0);
  EXPECT_FALSE(rb.regElem->isInUse());

  freeRegisteredBuffer(rb);
}

// Test that RegElemGuard move semantics work correctly
TEST_F(RegCacheTest, RegElemGuardMoveSemantics) {
  auto rb = allocateAndRegister(8192);
  ASSERT_NE(rb.regElem, nullptr);

  // Test move constructor
  {
    ctran::regcache::RegElemGuard guard1(rb.regElem);
    EXPECT_EQ(rb.regElem->getInUseCount(), 1);

    // Move to guard2
    ctran::regcache::RegElemGuard guard2(std::move(guard1));
    EXPECT_EQ(rb.regElem->getInUseCount(), 1); // Still 1, not 2
    EXPECT_EQ(guard1.get(), nullptr); // guard1 is now empty
    EXPECT_EQ(guard2.get(), rb.regElem);
  }
  EXPECT_EQ(rb.regElem->getInUseCount(), 0);

  // Test move assignment
  {
    ctran::regcache::RegElemGuard guard1(rb.regElem);
    ctran::regcache::RegElemGuard guard2;
    EXPECT_EQ(rb.regElem->getInUseCount(), 1);

    guard2 = std::move(guard1);
    EXPECT_EQ(rb.regElem->getInUseCount(), 1);
    EXPECT_EQ(guard1.get(), nullptr);
    EXPECT_EQ(guard2.get(), rb.regElem);
  }
  EXPECT_EQ(rb.regElem->getInUseCount(), 0);

  freeRegisteredBuffer(rb);
}

// Test demonstrating concurrent access scenario:
// One thread holds a guard (simulating a collective using the registration)
// while another thread attempts to check if it's safe to deregister.
// The isInUse() check prevents deregistration while the collective is
// active.
TEST_F(RegCacheTest, GuardProtectsConcurrentDeregistration) {
  auto rb = allocateAndRegister(8192);
  ASSERT_NE(rb.regElem, nullptr);

  std::atomic<bool> collectiveStarted{false};
  std::atomic<bool> deregChecked{false};
  std::atomic<bool> collectiveDone{false};
  std::atomic<bool> wasInUse{false};

  // Thread 1: Simulates a collective operation holding the guard
  std::thread collectiveThread([&]() {
    ctran::regcache::RegElemGuard guard(rb.regElem);

    // Signal that collective has started (guard is held)
    collectiveStarted.store(true);

    // Wait until the deregistration thread has checked isInUse
    while (!deregChecked.load()) {
      std::this_thread::yield();
    }

    // Guard is still held here, collective is "in progress"
    // When this thread exits, guard destructor decrements inUseCount
    collectiveDone.store(true);
  });

  // Thread 2: Simulates deregistration check
  std::thread deregThread([&]() {
    // Wait for collective to start
    while (!collectiveStarted.load()) {
      std::this_thread::yield();
    }

    // Check if registration is in use - should be true because
    // the collective thread holds the guard
    wasInUse.store(rb.regElem->isInUse());

    // Signal that we've checked
    deregChecked.store(true);
  });

  collectiveThread.join();
  deregThread.join();

  // Verify that isInUse() returned true while the guard was held
  EXPECT_TRUE(wasInUse.load())
      << "isInUse() should have returned true while collective held guard";

  // After both threads complete, the guard should be released
  EXPECT_FALSE(rb.regElem->isInUse());
  EXPECT_EQ(rb.regElem->getInUseCount(), 0);

  freeRegisteredBuffer(rb);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
