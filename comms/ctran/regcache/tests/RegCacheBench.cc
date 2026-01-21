// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

class RegCacheBench : public NcclxBaseTest {
 public:
  int cudaDev{0};
  std::shared_ptr<ctran::RegCache> regCache{nullptr};

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    setenv("NCCL_FASTINIT_MODE", "ring_hybrid", 1);
    NcclxBaseTest::SetUp();

    ctran::logGpuMemoryStats(cudaDev);

    commDeprecated_ = createNcclComm(globalRank, numRanks, localRank);
    comm_ = commDeprecated_->ctranComm_.get();

    // Turn on profiler after initialization to track only test registrations
    NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT = 0;

    if (!ctranInitialized(comm_) || !comm_->ctran_->mapper->hasBackend()) {
      GTEST_SKIP()
          << "Ctran is not initialized or backend is not available.  Skip test.";
    }

    // Setup regCache pointer to be used in test
    regCache = ctran::RegCache::getInstance();
    ASSERT_NE(regCache, nullptr);
  }

  void TearDown() override {
    // Report regCache stats including only test registrations
    regCache->profiler.rlock()->reportSnapshot();

    // Turn off profiler to avoid internal in comm destroy.
    NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT = -1;

    NCCLCHECK_TEST(ncclCommDestroy(commDeprecated_));
    ctran::logGpuMemoryStats(cudaDev);
    NcclxBaseTest::TearDown();
  }

 protected:
  ncclComm_t commDeprecated_{nullptr};
  CtranComm* comm_{nullptr};
};

class RegCacheTestParam
    : public RegCacheBench,
      public ::testing::WithParamInterface<std::tuple<size_t, int, size_t>> {};

TEST_P(RegCacheTestParam, RegMemAndSearchRegHandleTime) {
  auto& [offset, numSegments, segmentSize] = GetParam();

  constexpr int numIter = 100;
  std::vector<size_t> segSizes(numSegments, segmentSize);

  // Allocate different buffer for each iteration
  std::vector<void*> allBufs(numIter, nullptr);
  std::vector<TestMemSegment> allSegments;
  for (int iter = 0; iter < numIter; iter++) {
    std::vector<TestMemSegment> segments;
    NCCLCHECK_TEST(ncclMemAllocDisjoint(&allBufs[iter], segSizes, segments));
    allSegments.insert(allSegments.end(), segments.begin(), segments.end());
  }

  XLOG(INFO) << fmt::format(
      "offset={}, numSegments={}, segmentSize={}, allSegments.size={},"
      " allBufs.size={}",
      offset,
      numSegments,
      segmentSize,
      allSegments.size(),
      allBufs.size());

  auto& mapper = comm_->ctran_->mapper;
  EXPECT_THAT(mapper, testing::NotNull());

  auto t0 = std::chrono::steady_clock::now();
  std::vector<void*> allSegHandles;
  for (auto& segment : allSegments) {
    void* hdl = nullptr;
    COMMCHECK_TEST(mapper->regMem(segment.ptr, segment.size, &hdl, false));
    allSegHandles.push_back(hdl);
  }
  auto t1 = std::chrono::steady_clock::now();
  auto regTime =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
      allSegments.size();

  size_t bufSize = segmentSize * numSegments - offset;
  for (int iter = 0; iter < numIter; iter++) {
    void* regHdl = nullptr;
    bool dynamicRegist = false;
    COMMCHECK_TEST(mapper->searchRegHandle(
        reinterpret_cast<char*>(allBufs[iter]) + offset,
        bufSize,
        &regHdl,
        &dynamicRegist));
  }
  auto t2 = std::chrono::steady_clock::now();
  auto searchTime =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
      numIter;

  for (auto& segHdl : allSegHandles) {
    COMMCHECK_TEST(mapper->deregMem(segHdl));
  }
  auto t3 = std::chrono::steady_clock::now();
  auto deregTime =
      std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() /
      numIter;

  for (void* buf : allBufs) {
    NCCLCHECK_TEST(ncclMemFreeDisjoint(buf, segSizes));
  }

  std::cout << "RegMemAndSearchRegHandleTime: offset " << offset
            << ", numSegments " << numSegments << ", segmentSize "
            << segmentSize << ", regMemTime(us) " << regTime
            << ", firstRegTime(us) " << searchTime << ", deregMemTime(us) "
            << deregTime << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    RegCacheBench1,
    RegCacheTestParam,
    ::testing::Combine(
        testing::Values(0),
        testing::Values(1),
        testing::Values(
            20 * 1024 * 1024 * 1,
            20 * 1024 * 1024 * 2,
            20 * 1024 * 1024 * 4,
            20 * 1024 * 1024 * 8,
            20 * 1024 * 1024 * 16,
            20 * 1024 * 1024 * 32)),
    [&](const testing::TestParamInfo<RegCacheTestParam::ParamType>& info) {
      return std::to_string(std::get<0>(info.param)) + "offset_" +
          std::to_string(std::get<1>(info.param)) + "numSeg_" +
          std::to_string(std::get<2>(info.param)) + "SegSize";
    });

INSTANTIATE_TEST_SUITE_P(
    RegCacheBench2,
    RegCacheTestParam,
    ::testing::Combine(
        testing::Values(0),
        testing::Values(1, 2, 4, 8, 16, 32),
        testing::Values(20 * 1024 * 1024)),
    [&](const testing::TestParamInfo<RegCacheTestParam::ParamType>& info) {
      return std::to_string(std::get<0>(info.param)) + "offset_" +
          std::to_string(std::get<1>(info.param)) + "numSeg_" +
          std::to_string(std::get<2>(info.param)) + "SegSize";
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
