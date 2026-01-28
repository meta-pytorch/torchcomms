// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

using namespace ctran;

class RegCacheBackendMismatchTest : public CtranStandaloneFixture {
 public:
  void* buf = nullptr;
  size_t bufSize = 8192;
  std::shared_ptr<RegCache> regCache{nullptr};
  std::vector<TestMemSegment> segments;

 protected:
  void SetUp() override {
    CtranStandaloneFixture::SetUp();

    // Allocate memory using cuMem which supports NVL backend registration
    buf = commMemAlloc(bufSize, kMemCuMemAlloc, segments);
    ASSERT_NE(buf, nullptr) << "Failed to allocate cuMem buffer";

    // Clear the buffer
    CUDACHECK_TEST(cudaMemset(buf, 0, bufSize));

    // Setup regCache pointer
    regCache = RegCache::getInstance();
    ASSERT_NE(regCache, nullptr);
  }

  void TearDown() override {
    if (buf != nullptr) {
      commMemFree(buf, bufSize, kMemCuMemAlloc);
      buf = nullptr;
    }
    segments.clear();

    // Cleanup cached segments in global cache for each test
    EXPECT_EQ(regCache->destroy(), commSuccess);

    CtranStandaloneFixture::TearDown();
  }

  // Helper to create a CtranComm with specific backend configuration
  std::pair<std::unique_ptr<CtranComm>, std::unique_ptr<CtranMapper>>
  makeCtranCommAndMapper(std::vector<CommBackend> backends) {
    auto ctranComm = makeCtranComm();
    // Override the backends configuration
    ctranComm->config_.backends = std::move(backends);
    // Create a new mapper with the overridden configuration
    auto mapper = std::make_unique<CtranMapper>(ctranComm.get());
    return {std::move(ctranComm), std::move(mapper)};
  }
};

/**
 * RegCache test that tests multiple comms with different backends for single
 * buffer.
 *
 * When Comm1 (NVL only) registers a buffer and Comm2 (IB + NVL) looks it up,
 * Comm2 should get a RegElem that has both backends registered.
 */
TEST_F(
    RegCacheBackendMismatchTest,
    DISABLED_CacheHitMissingBackendRegistration) {
  // Create comm without IB backend
  auto [comm1, mapper1] =
      makeCtranCommAndMapper({CommBackend::SOCKET, CommBackend::NVL});

  // Register buffer through Comm1's mapper
  void* segHdl = nullptr;
  void* regHdl1 = nullptr;
  COMMCHECK_TEST(mapper1->regMem(
      buf,
      bufSize,
      &segHdl,
      true /* forceRegist */,
      false /* ncclManaged */,
      &regHdl1));
  ASSERT_NE(segHdl, nullptr) << "Segment handle should not be null";
  ASSERT_NE(regHdl1, nullptr) << "Registration handle should not be null";

  // Create comm with IB + NVL backends
  auto [comm2, mapper2] =
      makeCtranCommAndMapper({CommBackend::IB, CommBackend::NVL});

  // Lookup the same buffer through Comm2's mapper
  void* regHdl2 = nullptr;
  bool dynamicRegist = false;
  COMMCHECK_TEST(
      mapper2->searchRegHandle(buf, bufSize, &regHdl2, &dynamicRegist));
  ASSERT_NE(regHdl2, nullptr) << "Lookup should find a registration";
  EXPECT_FALSE(dynamicRegist) << "Should be a cache hit, not dynamic reg";

  // Verify we got the same RegElem (cache hit)
  auto* regElem1 = reinterpret_cast<regcache::RegElem*>(regHdl1);
  auto* regElem2 = reinterpret_cast<regcache::RegElem*>(regHdl2);
  EXPECT_EQ(regElem1, regElem2)
      << "Should be the same RegElem from cache (demonstrating the bug)";

  EXPECT_NE(regElem2->ibRegElem, nullptr)
      << "RegElem2 should have IB registration";

  // Cleanup
  COMMCHECK_TEST(mapper1->deregMem(segHdl));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
