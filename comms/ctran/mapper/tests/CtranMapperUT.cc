// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperImpl.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/logger/LogUtils.h"

class CtranMapperTest : public ::testing::Test {
 public:
  std::unique_ptr<ctran::TestCtranCommRAII> commRAII_;
  CtranComm* dummyComm_{nullptr};

  std::unique_ptr<CtranMapper> mapper;
  void *buf, *buf2;
  size_t bufSize = 8192;
  void* hdl = nullptr;
  int cudaDev = 0;
  CtranMapperTest() = default;
  std::shared_ptr<ctran::RegCache> regCache{nullptr};

 protected:
  void SetUp() override {
    ncclCvarInit();
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "true", 1);

    ctran::logGpuMemoryStats(cudaDev);

    commRAII_ = ctran::createDummyCtranComm();
    dummyComm_ = commRAII_->ctranComm.get();
    CUDACHECK_TEST(cudaSetDevice(cudaDev));
    CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
    CUDACHECK_TEST(cudaMalloc(&buf2, bufSize));
    CUDACHECK_TEST(cudaMemset(buf, 0, bufSize));

    // Turn on profiler after initialization to track only test registrations
    NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT = 0;

    // Setup regCache pointer to be used in test
    regCache = ctran::RegCache::getInstance();
    ASSERT_NE(regCache, nullptr);
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaFree(buf));
    CUDACHECK_TEST(cudaFree(buf2));

    // Cleanup cached segments in global cache for each test
    EXPECT_EQ(regCache->destroy(), commSuccess);

    ctran::logGpuMemoryStats(cudaDev);
  }
};
TEST(CtranMapperUT, EnableBackendThroughCVARs) {
  setenv("NCCL_CTRAN_BACKENDS", "ib, nvl, socket", 1);
  ncclCvarInit();
  auto commRAII = ctran::createDummyCtranComm();
  auto dummyComm = commRAII->ctranComm.get();
  auto mapper = std::make_unique<CtranMapper>(dummyComm);
  auto rank = dummyComm->statex_->rank();
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::IB));
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::NVL));
  // Socket is disabled if ib is enabled
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::SOCKET));
}

TEST(CtranMapperUT, EnableBackendThroughCVARsWithoutIB) {
  setenv("NCCL_CTRAN_BACKENDS", "nvl, socket", 1);
  ncclCvarInit();
  auto commRAII = ctran::createDummyCtranComm();
  auto dummyComm = commRAII->ctranComm.get();
  auto mapper = std::make_unique<CtranMapper>(dummyComm);
  auto rank = dummyComm->statex_->rank();
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::IB));
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::NVL));
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::SOCKET));
}
TEST(CtranMapperUT, EnableBackendWithConfigUnset) {
  // Test that when config_.backends contains only UNSET, mapper falls back
  // to using NCCL_CTRAN_BACKENDS CVAR.
  setenv("NCCL_CTRAN_BACKENDS", "nvl, socket", 1);
  ncclCvarInit();
  auto commRAII = ctran::createDummyCtranComm();
  auto dummyComm = commRAII->ctranComm.get();
  dummyComm->config_.backends = {CommBackend::UNSET}; // Test fallback behavior
  auto mapper = std::make_unique<CtranMapper>(dummyComm);
  auto rank = dummyComm->statex_->rank();
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::IB));
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::NVL));
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::SOCKET));
}

TEST(CtranMapperUT, EnableBackendWithExplicitConfigOverride) {
  // Test that when config_.backends is explicitly set,
  // it overrides the NCCL_CTRAN_BACKENDS CVAR.
  setenv("NCCL_CTRAN_BACKENDS", "nvl, socket", 1);
  ncclCvarInit();
  auto commRAII = ctran::createDummyCtranComm();
  auto dummyComm = commRAII->ctranComm.get();
  // Explicitly set config_.backends to IB only.
  dummyComm->config_.backends = {CommBackend::IB};
  auto mapper = std::make_unique<CtranMapper>(dummyComm);
  auto rank = dummyComm->statex_->rank();
  // Verify config_.backends={IB} overrides NCCL_CTRAN_BACKENDS={nvl, socket}
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::IB));
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::NVL));
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::SOCKET));
}

TEST(CtranMapperUT, EnableBackendThroughCVARsWithTCPandIB) {
  setenv("NCCL_CTRAN_BACKENDS", "nvl, ib, socket, tcpdm", 1);
  ncclCvarInit();
  std::optional<std::exception> ex;
  try {
    ctran::createDummyCtranComm();
  } catch (const ctran::utils::Exception& e) {
    ex = e;
  }
  ASSERT_TRUE(ex.has_value());
}

TEST(CtranMapperUT, BackendEnum) {
  for (int i = 0; i < CtranMapperBackend::NUM_BACKENDS; ++i) {
    auto backend = static_cast<CtranMapperBackend>(i);
    if (backend == CtranMapperBackend::UNSET) {
      continue;
    }
    // Check if all valid backends have corresponding string representation
    EXPECT_NE(CtranMapper::backendToStr(backend), "UNKNOWN");
  }
}

TEST_F(CtranMapperTest, Init) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());
}

TEST_F(CtranMapperTest, regMemLazy) {
  EnvRAII env(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::lazy);

  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  // Check profiled registration events.
  // Expect 1 cache entry and no registration
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 1);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 0);
  EXPECT_EQ(snapshot.totalNumDereg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);

  EXPECT_EQ(mapper->deregMem(hdl), commSuccess);

  // Check profiled registration events
  snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 0);
  EXPECT_EQ(snapshot.totalNumDereg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, regHostMemLazy) {
  EnvRAII env(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::lazy);

  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  void* bufH = malloc(bufSize);
  void* segHdl = nullptr;
  auto res = mapper->regMem(bufH, bufSize, &segHdl, false);

  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(segHdl, testing::NotNull());

  EXPECT_EQ(mapper->segmentType(segHdl), DevMemType::kHostUnregistered);

  // Check profiled registration events.
  // Expect 1 cache entry and no registration
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 1);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 0);
  EXPECT_EQ(snapshot.totalNumDereg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);

  // Trigger backend registration
  bool dynamicRegist = false;
  void* regHdl = nullptr;
  EXPECT_EQ(
      mapper->searchRegHandle(bufH, bufSize, &regHdl, &dynamicRegist),
      commSuccess);
  EXPECT_THAT(regHdl, testing::NotNull());
  EXPECT_FALSE(dynamicRegist);

  snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 1);
  EXPECT_EQ(snapshot.currentNumReg, 1);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 1);

  EXPECT_EQ(mapper->deregMem(segHdl), commSuccess);
  free(bufH);

  // Check profiled registration events
  snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 1);
  EXPECT_EQ(snapshot.totalNumDereg, 1);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, segmentBufReturnsOriginalAddr) {
  EnvRAII env(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::lazy);

  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);
  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  EXPECT_EQ(mapper->segmentBuf(hdl), buf);

  EXPECT_EQ(mapper->deregMem(hdl), commSuccess);
}

TEST_F(CtranMapperTest, segmentBufReturnsOriginalAddrHostMem) {
  EnvRAII env(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::lazy);

  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  void* bufH = malloc(bufSize);
  void* segHdl = nullptr;
  auto res = mapper->regMem(bufH, bufSize, &segHdl, false);
  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(segHdl, testing::NotNull());

  EXPECT_EQ(mapper->segmentBuf(segHdl), bufH);

  EXPECT_EQ(mapper->deregMem(segHdl), commSuccess);
  free(bufH);
}

TEST_F(CtranMapperTest, deregMem) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, commSuccess);

  // Check profiled registration events
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 0);
  EXPECT_EQ(snapshot.totalNumDereg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, deregMemNull) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  auto res = mapper->deregMem(hdl);
  EXPECT_EQ(res, commSuccess);

  // Check profiled registration events
  // Expect no cache entry and no registration
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 0);
  EXPECT_EQ(snapshot.totalNumReg, 0);
  EXPECT_EQ(snapshot.totalNumDereg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, doubleRegMem) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  void* hdl2;
  res = mapper->regMem(buf, bufSize, &hdl2, false);

  EXPECT_EQ(res, commSuccess);
  // The same handle should be returned
  EXPECT_EQ(hdl, hdl2);

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, commSuccess);

  EXPECT_EQ(mapper->deregMem(hdl2), commSuccess);

  // Check profiled registration events
  // Expect 1 cache entry and no registration (lazy)
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 0);
  EXPECT_EQ(snapshot.totalNumDereg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, doubleDeregMem) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, commSuccess);

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, commInvalidUsage);

  // Check profiled registration events
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 0);
  EXPECT_EQ(snapshot.totalNumDereg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, regMemEager) {
  EnvRAII env(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::eager);

  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  constexpr int numThreads = 10;
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          void *segHdl = nullptr, *regHdl = nullptr;
          auto res =
              mapper->regMem(buf, bufSize, &segHdl, false, false, &regHdl);
          EXPECT_EQ(res, commSuccess);
          EXPECT_THAT(segHdl, testing::NotNull());
          EXPECT_THAT(regHdl, testing::NotNull());

          res = mapper->deregMem(segHdl);
          EXPECT_EQ(res, commSuccess);
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  // Check profiled registration events.
  // Depending on the order of deregMem calls, we may see >1 cache/reg/dereg
  // records. This is because if a thread calls deregMem before another thread
  // calls regMem, the current cache entry will be released and a new cache
  // entry will be created,
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_GE(snapshot.totalNumCache, 1);
  EXPECT_GE(snapshot.totalNumReg, 1);
  EXPECT_GE(snapshot.totalNumDereg, 1);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, regMemForce) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  constexpr int numThreads = 10;
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          void *segHdl = nullptr, *regHdl = nullptr;
          auto res =
              mapper->regMem(buf, bufSize, &segHdl, true, false, &regHdl);
          EXPECT_EQ(res, commSuccess);
          EXPECT_THAT(segHdl, testing::NotNull());
          EXPECT_THAT(regHdl, testing::NotNull());

          EXPECT_EQ(mapper->deregMem(segHdl), commSuccess);
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  // Check profiled registration events.
  // Depending on the order of deregMem calls, we may see >1 cache/reg/dereg
  // records. This is because if a thread calls deregMem before another thread
  // calls regMem, the current cache entry will be released and a new cache
  // entry will be created,
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_GE(snapshot.totalNumCache, 1);
  EXPECT_GE(snapshot.totalNumReg, 1);
  EXPECT_GE(snapshot.totalNumDereg, 1);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, regMemNMissingDereg) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  constexpr int numThreads = 10;
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          void* hdl = nullptr;
          EXPECT_EQ(mapper->regMem(buf, bufSize, &hdl, false), commSuccess);
          EXPECT_THAT(hdl, testing::NotNull());
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  // Check profiled registration events
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 1);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 0);
  EXPECT_EQ(snapshot.totalNumDereg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, searchRegHandleMiss) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());
  constexpr int numThreads = 10;

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          void* regHdl = nullptr;
          bool dynamicRegist = false;
          EXPECT_EQ(
              mapper->searchRegHandle(buf, bufSize, &regHdl, &dynamicRegist),
              commSuccess);
          EXPECT_THAT(regHdl, testing::NotNull());

          // upon cache miss, the buffer should be registered dynamically
          EXPECT_TRUE(dynamicRegist);

          EXPECT_EQ(mapper->deregDynamic(regHdl), commSuccess);
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  // Check profiled registration events
  // Dynamic registration should be kept thread local, thus we see numThreads
  // sets of records.
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 0);
  EXPECT_EQ(snapshot.totalNumReg, numThreads);
  EXPECT_EQ(snapshot.totalNumDereg, numThreads);
  EXPECT_EQ(snapshot.totalNumDynamicReg, numThreads);
}

TEST_F(CtranMapperTest, searchRegHandleHit) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  constexpr int numThreads = 10;
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          void* regHdl = nullptr;
          bool dynamicRegist = false;
          EXPECT_EQ(
              mapper->searchRegHandle(buf, bufSize, &regHdl, &dynamicRegist),
              commSuccess);
          EXPECT_THAT(regHdl, testing::NotNull());

          EXPECT_FALSE(dynamicRegist);
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  // Check profiled registration events
  // Expect all threads share the same cache entry and corresponding
  // registration.
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 1);
  EXPECT_EQ(snapshot.currentNumReg, 1);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 1);
  EXPECT_EQ(snapshot.totalNumDereg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, commSuccess);

  snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 1);
  EXPECT_EQ(snapshot.totalNumDereg, 1);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, RegMemAndsearchRegHandleHit) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  constexpr int numThreads = 10;
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          // Some threads will register buf2 while the other threads are
          // searching
          void* hdl2 = nullptr;
          if (tid % 2 == 0) {
            EXPECT_EQ(mapper->regMem(buf2, bufSize, &hdl2, false), commSuccess);
          }

          void* regHdl = nullptr;
          bool dynamicRegist = false;
          EXPECT_EQ(
              mapper->searchRegHandle(buf, bufSize, &regHdl, &dynamicRegist),
              commSuccess);
          EXPECT_THAT(regHdl, testing::NotNull());

          EXPECT_FALSE(dynamicRegist);

          if (tid % 2 == 0) {
            EXPECT_EQ(mapper->deregMem(hdl2), commSuccess);
          }
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, commSuccess);
}

TEST_F(CtranMapperTest, RegMemAndsearchRegHandleMiss) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  void* buf3 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf3, bufSize));

  constexpr int numThreads = 10;
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          // Some threads will register buf2 while the other threads are
          // searching
          void* hdl2 = nullptr;
          if (tid % 2 == 0) {
            EXPECT_EQ(mapper->regMem(buf2, bufSize, &hdl2, false), commSuccess);
          }

          // Search buf3, which will be lookup miss and dynamically registered
          void* regHdl = nullptr;
          bool dynamicRegist = false;
          EXPECT_EQ(
              mapper->searchRegHandle(buf3, bufSize, &regHdl, &dynamicRegist),
              commSuccess);
          EXPECT_THAT(regHdl, testing::NotNull());
          EXPECT_TRUE(dynamicRegist);
          EXPECT_EQ(mapper->deregDynamic(regHdl), commSuccess);

          if (tid % 2 == 0) {
            EXPECT_EQ(mapper->deregMem(hdl2), commSuccess);
          }
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, commSuccess);

  CUDACHECK_TEST(cudaFree(buf3));
}

TEST_F(CtranMapperTest, AsyncRegMem) {
  EnvRAII env = EnvRAII(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::async);

  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Reinitialize global cache to enable asyncReg thread
  regCache->init();

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  // Cache the buffer
  EXPECT_EQ(mapper->regMem(buf, bufSize, &hdl, false), commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  // Submit async registration requests multiple times.
  // Expect registration is performed only once for a given buffer and the rest
  // are no-op
  for (int i = 0; i < 5; i++) {
    auto res = mapper->regAsync(buf, bufSize);
    EXPECT_EQ(res, commInProgress);
  }

  // Submit async registration with a uncached buffer.
  // Expect asyncThread to ignore it and later searchRegHandle sees
  // dynamicRegist is set.
  auto res = mapper->regAsync(buf2, bufSize);
  EXPECT_EQ(res, commInProgress);

  // Ensure async thread has finished all requests
  regCache->waitAsyncRegComplete();

  // searchRegHandle is blocked till async registration complete
  void *regHdl1 = nullptr, *regHdl2 = nullptr;
  bool dynamic = false;
  EXPECT_EQ(
      mapper->searchRegHandle(buf, bufSize, &regHdl1, &dynamic), commSuccess);
  EXPECT_THAT(regHdl1, testing::NotNull());
  EXPECT_FALSE(dynamic);

  EXPECT_EQ(
      mapper->searchRegHandle(buf2, bufSize, &regHdl2, &dynamic), commSuccess);
  EXPECT_THAT(regHdl2, testing::NotNull());
  EXPECT_TRUE(dynamic);
  EXPECT_EQ(mapper->deregDynamic(regHdl2), commSuccess);

  EXPECT_EQ(mapper->deregMem(hdl), commSuccess);

  // Check profiled registration events
  // Expect two registrations are recorded and both handled by asyncReg thread
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 2);
  EXPECT_EQ(snapshot.totalNumDereg, 2);
  EXPECT_EQ(snapshot.totalNumAsyncReg, 1);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 1);
}

TEST_F(CtranMapperTest, AsyncRegMemWithErroOnDynamic) {
  EnvRAII env = EnvRAII(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::async);
  EnvRAII envErrOnDynamic = EnvRAII(NCCL_CTRAN_REGISTER_ERROR_ON_DYNAMIC, true);

  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Reinitialize global cache to enable asyncReg thread
  regCache->init();

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  // Cache buffer
  EXPECT_EQ(mapper->regMem(buf, bufSize, &hdl, false), commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  // Submit async registration requests multiple times.
  // Expect registration is performed only once for a given buffer and the rest
  // are no-op
  for (int i = 0; i < 5; i++) {
    auto res = mapper->regAsync(buf, bufSize);
    // With NCCL_CTRAN_REGISTER_ERROR_ON_DYNAMIC, enforce registration happens
    // on the calling thread, thus no async registration is performed.
    EXPECT_EQ(res, commSuccess);
  }

  // Submit async registration with a uncached buffer.
  // Expect fail here
  auto res = mapper->regAsync(buf2, bufSize);
  ASSERT_EQ(res, commInvalidUsage);

  // Ensure async thread has finished all requests
  regCache->waitAsyncRegComplete();
  EXPECT_EQ(mapper->deregMem(hdl), commSuccess);

  // Check profiled registration events
  // Expect two registrations are recorded and both handled by asyncReg thread
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 1);
  EXPECT_EQ(snapshot.totalNumDereg, 1);
  EXPECT_EQ(snapshot.totalNumAsyncReg, 0);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, AsyncRegMemAndSearchRegHandleHit) {
  EnvRAII env = EnvRAII(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::async);

  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Reinitialize global cache to enable asyncReg thread
  regCache->init();

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  // Cache the buffer
  auto res = mapper->regMem(buf, bufSize, &hdl, false);
  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  // Submit async registration request multiple times.
  // Expect registration is performed only once and the rest are no-op
  for (int i = 0; i < 5; i++) {
    res = mapper->regAsync(buf, bufSize);
    EXPECT_EQ(res, commInProgress);
  }

  // While async registration is in progress, launch test threads to search the
  // buffer
  constexpr int numThreads = 5;
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          void* regHdl = nullptr;
          bool dynamicRegist = false;
          EXPECT_EQ(
              mapper->searchRegHandle(buf, bufSize, &regHdl, &dynamicRegist),
              commSuccess);
          EXPECT_THAT(regHdl, testing::NotNull());
          EXPECT_FALSE(dynamicRegist);
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, commSuccess);

  // Check profiled registration events
  // Expect the registration is handled by either asyncReg thread or the first
  // searchRegHandle.
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 1);
  EXPECT_EQ(snapshot.totalNumReg, 1);
  EXPECT_EQ(snapshot.totalNumDereg, 1);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}

TEST_F(CtranMapperTest, icopy) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  cudaStream_t stream = 0;
  CtranMapperRequest* req = nullptr;
  char* srcBuf;
  CUDACHECK_TEST(cudaMalloc(&srcBuf, bufSize));
  CUDACHECK_TEST(cudaMemset(srcBuf, 1, bufSize));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  auto res = mapper->icopy(buf, srcBuf, bufSize, stream, &req);
  EXPECT_EQ(res, commSuccess);

  bool isComplete = false;
  do {
    mapper->testRequest(req, &isComplete);
  } while (!isComplete);

  std::vector<char> observedVals(bufSize);
  CUDACHECK_TEST(
      cudaMemcpy(observedVals.data(), buf, bufSize, cudaMemcpyDefault));
  EXPECT_THAT(observedVals, testing::Each(1));

  CUDACHECK_TEST(cudaFree(srcBuf));
  delete req;
}

TEST_F(CtranMapperTest, icopyNoReq) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  cudaStream_t stream = 0;
  char* srcBuf;
  CUDACHECK_TEST(cudaMalloc(&srcBuf, bufSize));
  CUDACHECK_TEST(cudaMemset(srcBuf, 1, bufSize));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  auto res = mapper->icopy(buf, srcBuf, bufSize, stream);
  EXPECT_EQ(res, commSuccess);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<char> observedVals(bufSize);
  CUDACHECK_TEST(
      cudaMemcpy(observedVals.data(), buf, bufSize, cudaMemcpyDefault));
  EXPECT_THAT(observedVals, testing::Each(1));

  CUDACHECK_TEST(cudaFree(srcBuf));
}

TEST_F(CtranMapperTest, ReqInit) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  const auto rank = this->dummyComm_->statex_->rank();
  CtranMapperRequest* req =
      new CtranMapperRequest(CtranMapperRequest::ReqType::COPY, rank);
  EXPECT_THAT(req, testing::NotNull());

  delete req;
}

TEST_F(CtranMapperTest, ReqTest) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  for (auto type :
       {CtranMapperRequest::ReqType::SEND_CTRL,
        CtranMapperRequest::ReqType::IB_PUT,
        CtranMapperRequest::ReqType::COPY}) {
    const auto rank = this->dummyComm_->statex_->rank();
    CtranMapperRequest* req = new CtranMapperRequest(type, rank);
    EXPECT_THAT(req, testing::NotNull());

    // IB_PUT or CTRL request should NOT complete before internal IB
    // request completes
    if (type != CtranMapperRequest::ReqType::COPY) {
      bool isComplete = false;
      auto res = mapper->testRequest(req, &isComplete);
      ;
      EXPECT_EQ(res, commSuccess);
      EXPECT_FALSE(isComplete);
    }

    // Complete internal IB request
    req->ibReq.complete();

    // Check mapper request completes
    bool isComplete = false;
    auto res = mapper->testRequest(req, &isComplete);
    EXPECT_EQ(res, commSuccess);
    EXPECT_TRUE(isComplete);

    delete req;
  }
}

TEST_F(CtranMapperTest, ReqWait) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  for (auto type :
       {CtranMapperRequest::ReqType::SEND_CTRL,
        CtranMapperRequest::ReqType::IB_PUT,
        CtranMapperRequest::ReqType::COPY}) {
    const auto rank = this->dummyComm_->statex_->rank();
    CtranMapperRequest* req = new CtranMapperRequest(type, rank);
    EXPECT_THAT(req, testing::NotNull());

    // Complete internal IB request
    req->ibReq.complete();

    // Wait mapper request completes
    auto res = mapper->waitRequest(req);
    EXPECT_EQ(res, commSuccess);

    delete req;
  }
}

void runTestSome(
    std::unique_ptr<CtranMapper>& mapper,
    const int numReqs,
    bool recordTime,
    std::vector<std::unique_ptr<CtranMapperRequest>>& reqs,
    std::vector<CtranMapperTimestampPoint>& tps) {
  commResult_t res;
  for (int i = 0; i < numReqs; i++) {
    int peer = i % 2;
    reqs.push_back(
        std::make_unique<CtranMapperRequest>(
            CtranMapperRequest::ReqType::SEND_CTRL, peer));
  }

  int completed = 0;
  while (!reqs.empty()) {
    // Complete internal IB request for the first remainiing request
    reqs.front()->ibReq.complete();
    completed++;

    // Expect completed request to be erased
    if (recordTime) {
      res = mapper->testSomeRequests(reqs, tps);
    } else {
      res = mapper->testSomeRequests(reqs);
    }
    EXPECT_EQ(res, commSuccess);

    // Check expected number of remaining requests
    EXPECT_EQ(numReqs - completed, reqs.size());
  }
}

TEST_F(CtranMapperTest, ReqTestSome) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  std::vector<std::unique_ptr<CtranMapperRequest>> reqs;
  std::vector<CtranMapperTimestampPoint> tps;

  constexpr int numReqs = 10;
  runTestSome(mapper, numReqs, false, reqs, tps);
  EXPECT_TRUE(reqs.empty());
}

TEST_F(CtranMapperTest, ReqTestSomeWithRecordTime) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);

  constexpr int numReqs = 10;
  std::vector<std::unique_ptr<CtranMapperRequest>> reqs;
  std::vector<CtranMapperTimestampPoint> tps;
  CtranMapperTimestampPoint startTp = CtranMapperTimestampPoint(0);

  runTestSome(mapper, numReqs, true, reqs, tps);
  EXPECT_TRUE(reqs.empty());

  EXPECT_EQ(tps.size(), numReqs);
  CtranMapperTimestampPoint* prevTp = &startTp;
  for (int i = 0; i < numReqs; i++) {
    EXPECT_EQ(tps[i].peer, i % 2);
    EXPECT_GT(tps[i].now, prevTp->now);
    prevTp = &tps[i];
  }
}

TEST_F(CtranMapperTest, getNumSegments) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  void *segHdl = nullptr, *regHdl = nullptr;
  auto res = mapper->regMem(
      buf, bufSize, &segHdl, true /* force register */, false, &regHdl);
  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(segHdl, testing::NotNull());
  EXPECT_THAT(regHdl, testing::NotNull());
  EXPECT_EQ(((ctran::regcache::RegElem*)regHdl)->numSegments(), 1);
}

TEST_F(CtranMapperTest, getRegElems) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  // Test regElem backed with single segment, should return only 1 regElem
  void* segHdl = nullptr;
  void *regHdl0 = nullptr, *regHdl1 = nullptr;
  auto res = mapper->regMem(
      buf, bufSize, &segHdl, true /* force register */, false, &regHdl0);
  EXPECT_EQ(res, commSuccess);
  EXPECT_THAT(segHdl, testing::NotNull());
  EXPECT_THAT(regHdl0, testing::NotNull());

  // - Search should not trigger another registration, and return the same
  // regHdl
  bool dynamicReg = false;
  EXPECT_EQ(
      mapper->searchRegHandle(buf, bufSize, &regHdl1, &dynamicReg),
      commSuccess);
  EXPECT_EQ(regHdl0, regHdl1);
  EXPECT_FALSE(dynamicReg);

  // - Expect only 1 regElem for the given segHdl
  auto regElems = regCache->getRegElems(segHdl);
  EXPECT_EQ(regElems.size(), 1);
  EXPECT_EQ(regElems.at(0), regHdl0);

  EXPECT_EQ(mapper->deregMem(segHdl), commSuccess);

  // - After deregMem, expect no regElem for the given segHdl
  regElems = regCache->getRegElems(segHdl);
  EXPECT_EQ(regElems.size(), 0);
}

TEST_F(CtranMapperTest, getMultiSegRegElems) {
  // Test regElem backed with multiple segments, which may cause multiple
  // regElems per segHdl
  if (!ctran::utils::getCuMemSysSupported()) {
    GTEST_SKIP() << "CuMem not supported, skip multi-segment test";
  }

  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  void* buf_ = nullptr;
  std::vector<TestMemSegment> segments;
  std::vector<size_t> segSizes(3, 1048576);
  EXPECT_EQ(
      ctran::commMemAllocDisjoint(&buf_, segSizes, segments), commSuccess);

  std::vector<void*> segHdls;
  for (auto& seg : segments) {
    void* hdl = nullptr;
    EXPECT_EQ(mapper->regMem(seg.ptr, seg.size, &hdl), commSuccess);
    segHdls.push_back(hdl);
  }

  void *regHdl0 = nullptr, *regHdl1 = nullptr, *regHdl2 = nullptr;

  // - Before search, expect no regElem for the given segHdl
  auto regElems = regCache->getRegElems(segHdls.at(0));
  EXPECT_EQ(regElems.size(), 0);

  // - register buf_ with different segments, which may cause multiple regElems.
  //   Use the actual size of each segment to ensure we touch multiple segments
  //   per registration, since the actual allocated size may be greater than the
  //   requested size in segSizes

  // - First regElem backed with segments[0]
  size_t size_ = segments.at(0).size;
  bool dynamicReg = false;
  EXPECT_EQ(
      mapper->searchRegHandle(buf_, size_, &regHdl0, &dynamicReg), commSuccess);
  EXPECT_THAT(regHdl0, testing::NotNull());
  EXPECT_FALSE(dynamicReg);

  // - Second regElem backed with segments[0] and segments[1]
  size_ = segments.at(0).size + segments.at(1).size;
  EXPECT_EQ(
      mapper->searchRegHandle(buf_, size_, &regHdl1, &dynamicReg), commSuccess);
  EXPECT_THAT(regHdl1, testing::NotNull());
  EXPECT_FALSE(dynamicReg);

  // - Third regElem backed with segments[0], segments[1] and segments[2]
  size_ = segments.at(0).size + segments.at(1).size + segments.at(2).size;
  EXPECT_EQ(
      mapper->searchRegHandle(buf_, size_, &regHdl2, &dynamicReg), commSuccess);
  EXPECT_THAT(regHdl2, testing::NotNull());
  EXPECT_FALSE(dynamicReg);

  // - Expect 3 regElem for the given segHdl
  regElems = regCache->getRegElems(segHdls.at(0));
  EXPECT_EQ(regElems.size(), 3);
  EXPECT_EQ(regElems.at(0), regHdl0);
  EXPECT_EQ(regElems.at(1), regHdl1);
  EXPECT_EQ(regElems.at(2), regHdl2);

  for (auto& hdl : segHdls) {
    EXPECT_EQ(mapper->deregMem(hdl), commSuccess);
  }
  EXPECT_EQ(ctran::commMemFreeDisjoint(buf_, segSizes), commSuccess);
}

TEST_F(CtranMapperTest, RemoteAccessKeyToString) {
  CtranMapperRemoteAccessKey rkey1 = {.backend = CtranMapperBackend::IB};
  for (auto i = 0; i < CTRAN_MAX_IB_DEVICES_PER_RANK; i++) {
    rkey1.ibKey.rkeys[i] = 291 + i;
  }
  rkey1.ibKey.nKeys = CTRAN_MAX_IB_DEVICES_PER_RANK;
  rkey1.nvlKey.peerId = "host1:1234";
  rkey1.nvlKey.basePtr = (void*)0x4567890;
  EXPECT_EQ(rkey1.toString(), "backend=IB, ibKey=[291, 292]");

  CtranMapperRemoteAccessKey rkey2 = rkey1;
  rkey2.backend = CtranMapperBackend::NVL;
  EXPECT_EQ(
      rkey2.toString(),
      "backend=NVL, nvlKey=[peerId: host1:1234, basePtr: 0x4567890, uid: 0]");

  CtranMapperRemoteAccessKey rkey3 = rkey1;
  rkey3.backend = CtranMapperBackend::UNSET;
  EXPECT_EQ(rkey3.toString(), "backend=UNKNOWN");
}

TEST_F(CtranMapperTest, ExportRegCache) {
  std::unique_ptr<ctran::ExportRegCache> cache =
      std::make_unique<ctran::ExportRegCache>();
  const ctran::regcache::RegElem* dummyRegElem0 =
      reinterpret_cast<ctran::regcache::RegElem*>(0x12345);
  const std::vector<int> peers = {0, 1, 2, 3};

  for (auto peer : peers) {
    cache->record(dummyRegElem0, peer);
  }

  // Except dump gives full copy of the cache
  const auto dump = cache->dump();
  EXPECT_EQ(dump.size(), 1);
  auto it = dump.begin();
  EXPECT_EQ(it->first, dummyRegElem0);
  EXPECT_EQ(it->second.size(), peers.size());

  const ctran::regcache::RegElem* dummyRegElem1 =
      reinterpret_cast<ctran::regcache::RegElem*>(0x12346);

  // Expect return empty vector for non-existing regElem
  auto cachedPeers = cache->remove(dummyRegElem1);
  EXPECT_EQ(cachedPeers.size(), 0);

  // Expect return cached peers for existing regElem
  cachedPeers = cache->remove(dummyRegElem0);
  EXPECT_EQ(cachedPeers.size(), peers.size());
  for (auto peer : peers) {
    EXPECT_EQ(cachedPeers.count(peer), 1);
  }

  // Expect empty dump after remove
  const auto dump1 = cache->dump();
  EXPECT_EQ(dump1.size(), 0);
}

class CtranMapperTestDisjoint : public ::testing::Test {
 public:
  std::unique_ptr<ctran::TestCtranCommRAII> commRAII_;
  CtranComm* dummyComm_{nullptr};

  std::unique_ptr<CtranMapper> mapper;
  void *bufBase, *buf;
  // Use larger buffer size to ensure we span multiple segments
  const size_t bufSize = 2097152 * 2; // 4MB to span segments
  // use offset to test registration of unaligned memory, on T20 (H100) host
  // page size is 4K so offset must be less than that
  const size_t offset = 128;
  const size_t bufBaseSize = bufSize + offset;
  void* hdl = nullptr;
  int cudaDev = 0;
  CtranMapperTestDisjoint() = default;
  std::shared_ptr<ctran::RegCache> regCache{nullptr};

  // Store segment sizes for proper cleanup of disjoint allocation
  std::vector<size_t> disjointSegSizes;
  bool usedDisjointAllocation = false;

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    ncclCvarInit();
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "true", 1);

    ctran::logGpuMemoryStats(cudaDev);

    commRAII_ = ctran::createDummyCtranComm();
    dummyComm_ = commRAII_->ctranComm.get();
    CUDACHECK_TEST(cudaSetDevice(cudaDev));

    // Skip test if cuMem is not supported since we need disjoint allocations
    if (!ctran::utils::getCuMemSysSupported()) {
      GTEST_SKIP() << "CuMem not supported, skip test";
    }

    // Create a buffer that spans multiple cuMem allocations to test dynamic
    // registration
    std::vector<TestMemSegment> segments;
    disjointSegSizes = {bufBaseSize / 2, bufBaseSize / 2};
    auto result =
        ctran::commMemAllocDisjoint(&bufBase, disjointSegSizes, segments);
    if (result != commSuccess) {
      GTEST_SKIP()
          << "Disjoint allocation failed, cannot test multiple allocation scenario";
    }
    usedDisjointAllocation = true;

    CLOGF(
        INFO,
        "Disjoint allocation created {} segments: seg0[{}, {}], seg1[{}, {}]",
        segments.size(),
        segments[0].ptr,
        segments[0].size,
        segments[1].ptr,
        segments[1].size);

    buf = (char*)bufBase + offset;

    CLOGF(INFO, "bufBase: {}, buf: {}", bufBase, buf);

    // Turn on profiler after initialization to track only test registrations
    NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT = 0;

    // Setup regCache pointer to be used in test
    regCache = ctran::RegCache::getInstance();
    ASSERT_NE(regCache, nullptr);
  }

  void TearDown() override {
    // Free disjoint allocation properly if it was used
    if (usedDisjointAllocation) {
      EXPECT_EQ(
          ctran::commMemFreeDisjoint(bufBase, disjointSegSizes), commSuccess);
    }

    // Cleanup cached segments in global cache for each test
    EXPECT_EQ(regCache->destroy(), commSuccess);

    ctran::logGpuMemoryStats(cudaDev);
  }
};

TEST_F(CtranMapperTestDisjoint, dynamicReg) {
  mapper = std::make_unique<CtranMapper>(dummyComm_);
  EXPECT_THAT(mapper, testing::NotNull());

  // Cleanup profiling record
  regCache->profiler.wlock()->reset();

  void* sendHdl = nullptr;
  bool isDynamicReg = false;
  // searchRegHandle has a side effect to perform dynamic registration if a
  // buffer is not registered, if dynamic registration happend isDynamicReg
  // will be set to true and user has to call deregDynamic to deregister the
  // buffer. This is typical path how is't done in collectives, e.g. see
  // SendRecv.cc.
  COMMCHECK_TEST(
      mapper->searchRegHandle(buf, bufSize, &sendHdl, &isDynamicReg));
  ASSERT_NE(sendHdl, nullptr);
  ASSERT_EQ(isDynamicReg, true);

  COMMCHECK_TEST(mapper->deregDynamic(sendHdl));

  // Check profiled registration events
  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);
  EXPECT_EQ(snapshot.totalNumCache, 0);
  EXPECT_EQ(snapshot.totalNumReg, 1);
  EXPECT_EQ(snapshot.totalNumDereg, 1);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 1);
}
